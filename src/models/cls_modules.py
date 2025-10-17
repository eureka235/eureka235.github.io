import os, math, random, argparse, json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys

from demo.models_new.fusion import AttentionFusion, GatedFusion, TransformerFusion


class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", trainable=True, 
                 use_temporal=True, max_seq_len=512):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size
        self.use_temporal = use_temporal
        
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False
                
        if use_temporal:
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden, nhead=8, 
                    dim_feedforward=self.hidden*2, dropout=0.1
                ), num_layers=2
            )
            
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_temporal:
            hidden_states = out.last_hidden_state  # (B, T, H)
            
            seq_len = hidden_states.size(1)
            hidden_states = hidden_states.transpose(0, 1)  # (T, B, H)
            
            # attention mask for transformer
            key_padding_mask = ~attention_mask.bool()  # (B, T)
            
            temporal_out = self.temporal_encoder(
                hidden_states, 
                src_key_padding_mask=key_padding_mask
            )  # (T, B, H)
            
            # masked mean pooling
            temporal_out = temporal_out.transpose(0, 1)  # (B, T, H)
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            masked_hidden = temporal_out * mask
            seq_lens = attention_mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            h_txt = masked_hidden.sum(dim=1) / seq_lens  # (B, H)
        else:
            # only [CLS]
            h_txt = out.last_hidden_state[:, 0, :]
            
        return h_txt



class WavLMAdapter(nn.Module):
    def __init__(self, in_dim: Optional[int] = None, out_dim: int = 64, 
                 use_temporal=True, num_layers=2):
        super().__init__()
        self.use_temporal = use_temporal
        
        if in_dim is not None:
            if use_temporal:
                self.temporal_conv = nn.Sequential(
                    nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                
                # LSTM for temporal dynamics
                self.temporal_lstm = nn.LSTM(
                    out_dim, out_dim//2, num_layers=num_layers, 
                    batch_first=True, bidirectional=True, dropout=0.1
                )
            else:
                self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = None
            
    def forward(self, wavlm_frames=None, wavlm_lens=None, wavlm_mean=None):
        if wavlm_mean is not None:
            if self.proj is not None:
                return self.proj(wavlm_mean)
            return wavlm_mean
            
        elif wavlm_frames is not None and wavlm_lens is not None:
            if self.use_temporal and hasattr(self, 'temporal_conv'):
                B, T, D = wavlm_frames.shape
                
                # Conv1d processing
                x = wavlm_frames.transpose(1, 2)  # (B, D, T)
                x = self.temporal_conv(x)  # (B, out_dim, T)
                x = x.transpose(1, 2)  # (B, T, out_dim)
                
                # LSTM processing
                lengths = wavlm_lens.cpu()
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_out, _ = self.temporal_lstm(packed_x)
                unpacked_out, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True
                )  # (B, T, out_dim)
                
                # temporal weigted pooling
                device = wavlm_frames.device
                lens = wavlm_lens.to(device).unsqueeze(-1)  # (B, 1)
                mask = torch.arange(T, device=device)[None, :] < wavlm_lens[:, None]
                
                # attn + mean pooling
                attn_weights = torch.softmax(
                    torch.sum(unpacked_out, dim=-1), dim=1
                )  # (B, T)
                attn_weights = attn_weights * mask.float()
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
                
                x = torch.sum(unpacked_out * attn_weights.unsqueeze(-1), dim=1)  # (B, out_dim)
            else:
                # mask mean pooling
                B, T, D = wavlm_frames.shape
                device = wavlm_frames.device
                lens = wavlm_lens.to(device).unsqueeze(-1)
                mask = torch.arange(T, device=device)[None, :] < wavlm_lens[:, None]
                mask = mask.float().unsqueeze(-1)
                s = (wavlm_frames * mask).sum(dim=1)
                x = s / torch.clamp(lens, min=1)
                
                if self.proj is not None:
                    x = self.proj(x)
            
            return x
        else:
            return None


##########################################################################
# SWM MODULES
class TemporalToMModule(nn.Module):
    def __init__(self, txt_hidden: int, z_dim: int = 88, 
                 wavlm_out_dim: Optional[int] = None,
                 tom_hidden: int = 128, num_emotions: int = 7,
                 use_temporal_attn: bool = True):
        super().__init__()
        
        self.use_temporal_attn = use_temporal_attn
        base_dim = txt_hidden + z_dim + (wavlm_out_dim or 0) #### fusion-256
        
        if use_temporal_attn:
            # MHA
            self.temporal_attn = nn.MultiheadAttention(
                base_dim, num_heads=8, dropout=0.1
            )
            self.norm = nn.LayerNorm(base_dim)
            
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, tom_hidden), 
            nn.GELU(), 
            nn.Dropout(0.3),  # dropout
            nn.Linear(tom_hidden, tom_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(tom_hidden // 2, num_emotions)
        )

    def forward(self, h_txt, z_prosody, wavlm_vec=None):
        feats = [h_txt, z_prosody]
        if wavlm_vec is not None:
            feats.append(wavlm_vec)
        x = torch.cat(feats, dim=-1)  # (B, D)
        
        if self.use_temporal_attn:
            x_seq = x.unsqueeze(1)  # (B, 1, D)
            x_seq = x_seq.transpose(0, 1)  # (1, B, D)
            
            attn_out, _ = self.temporal_attn(x_seq, x_seq, x_seq)
            x = self.norm(attn_out.squeeze(0) + x)
        
        logits = self.mlp(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs



    
# NEED TO DECIDE: wma_hidden, num_wma
class WMAModule(nn.Module):
    def __init__(self, txt_hidden: int,
                 wavlm_out_dim: Optional[int] = None,
                 wma_hidden: int = 128, num_wma: int = 7,
                 use_temporal_attn: bool = True):
        super().__init__()
        
        self.use_temporal_attn = use_temporal_attn
        base_dim = txt_hidden + (wavlm_out_dim or 0) # 768
        
        if use_temporal_attn:
            # MHA
            self.temporal_attn = nn.MultiheadAttention(
                base_dim, num_heads=8, dropout=0.1
            )
            self.norm = nn.LayerNorm(base_dim)
            
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, wma_hidden), 
            nn.GELU(), 
            nn.Dropout(0.3),  # dropout
            nn.Linear(wma_hidden, wma_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(wma_hidden // 2, num_wma)
        )

    def forward(self, h_txt, wavlm_vec=None):
        feats = [h_txt]
        if wavlm_vec is not None:
            feats.append(wavlm_vec)
        x = torch.cat(feats, dim=-1)  # (B, D)
        
        if self.use_temporal_attn:
            x_seq = x.unsqueeze(1)  # (B, 1, D)
            x_seq = x_seq.transpose(0, 1)  # (1, B, D)
            
            attn_out, _ = self.temporal_attn(x_seq, x_seq, x_seq)
            x = self.norm(attn_out.squeeze(0) + x)
        
        logits = self.mlp(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs



class SAModule(nn.Module):
    def __init__(self, txt_hidden: int, num_emotions: int, num_wma: int,
                 sa_hidden: int = 256, num_sa: int = 24, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        
        input_dim = txt_hidden + num_emotions + num_wma
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, sa_hidden), 
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.Linear(sa_hidden, sa_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(sa_hidden // 2, num_sa)
        )
        
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, num_sa)
          

    def forward(self, h_txt, tom_probs, wma_probs):
        feats = [h_txt]

        if tom_probs is not None:
            feats.append(tom_probs)

        if wma_probs is not None:
            feats.append(wma_probs)
        
        x = torch.cat(feats, dim=-1)  # (B, D)
        
        logits = self.mlp(x)
        
        if self.use_residual:
            residual = self.residual_proj(x)
            logits = logits + residual
            
        return logits



# NEED TO DECIDE: prag_hidden, num_prag
class PragModule(nn.Module):
    def __init__(self, txt_hidden: int, num_emotions: int, num_wma: int, num_sa: int,
                 prag_hidden: int = 256, num_prag: int = 24, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        
        input_dim = txt_hidden + num_emotions + num_wma + num_sa
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, prag_hidden), 
            nn.GELU(), 
            nn.Dropout(0.3),
            nn.Linear(prag_hidden, prag_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(prag_hidden // 2, num_prag)
        )
        
        if use_residual:
            self.residual_proj = nn.Linear(input_dim, num_prag)
            
    def forward(self, h_txt, tom_probs, wma_probs, sa_probs):
        feats = [h_txt]

        if tom_probs is not None:
            feats.append(tom_probs)

        if wma_probs is not None:
            feats.append(wma_probs)
        
        if sa_probs is not None:
            feats.append(sa_probs)

        x = torch.cat(feats, dim=-1)  # (B, D)
        
        logits = self.mlp(x)
        
        if self.use_residual:
            residual = self.residual_proj(x)
            logits = logits + residual
            
        return logits