import os, math, random, argparse, json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys


class AttentionFusion(nn.Module):
    def __init__(self, txt_dim, prosody_dim, wavlm_dim, hidden_dim=256):
        super().__init__()
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim) if wavlm_dim > 0 else None
        
        # cross-attn
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h_txt, z_prosody, wavlm_vec=None):
        txt_feat = self.txt_proj(h_txt).unsqueeze(1)  # (B, 1, H)
        prosody_feat = self.prosody_proj(z_prosody).unsqueeze(1)  # (B, 1, H)
        
        features = [txt_feat, prosody_feat]
        if wavlm_vec is not None and self.wavlm_proj is not None:
            wavlm_feat = self.wavlm_proj(wavlm_vec).unsqueeze(1)  # (B, 1, H)
            features.append(wavlm_feat)
        
        x = torch.cat(features, dim=1)  # (B, N, H)
        
        x = x.transpose(0, 1)  # (N, B, H)
        attn_out, _ = self.cross_attn(x, x, x)
        x = self.norm(attn_out + x)
        x = x.transpose(0, 1)  # (B, N, H)
        
        fused = x.mean(dim=1)  # (B, H)
        return fused


class GatedFusion(nn.Module):
    def __init__(self, txt_dim, prosody_dim, wavlm_dim, hidden_dim=256):
        super().__init__()
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim) if wavlm_dim > 0 else None
        
        total_dim = hidden_dim * (3 if wavlm_dim > 0 else 2)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 if wavlm_dim > 0 else 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, h_txt, z_prosody, wavlm_vec=None):
        txt_feat = self.txt_proj(h_txt)
        prosody_feat = self.prosody_proj(z_prosody)
        
        features = [txt_feat, prosody_feat]
        if wavlm_vec is not None and self.wavlm_proj is not None:
            wavlm_feat = self.wavlm_proj(wavlm_vec)
            features.append(wavlm_feat)
        
        concat_feat = torch.cat(features, dim=-1)
        gates = self.gate(concat_feat)  # (B, N)
        
        fused = sum(gate.unsqueeze(-1) * feat for gate, feat in zip(gates.unbind(-1), features))
        return fused


class TransformerFusion(nn.Module):
    def __init__(self, txt_dim, prosody_dim, wavlm_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)
        self.prosody_proj = nn.Linear(prosody_dim, hidden_dim)
        self.wavlm_proj = nn.Linear(wavlm_dim, hidden_dim) if wavlm_dim > 0 else None
        
        max_modalities = 3
        self.pos_embed = nn.Parameter(torch.randn(max_modalities, hidden_dim))
        

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, 
            dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def forward(self, h_txt, z_prosody, wavlm_vec=None):
        features = []
        
        txt_feat = self.txt_proj(h_txt) + self.pos_embed[0]
        prosody_feat = self.prosody_proj(z_prosody) + self.pos_embed[1]
        features = [txt_feat, prosody_feat]
        
        if wavlm_vec is not None and self.wavlm_proj is not None:
            wavlm_feat = self.wavlm_proj(wavlm_vec) + self.pos_embed[2]
            features.append(wavlm_feat)
        
        x = torch.stack(features, dim=1)  # (B, N, H)
        x = x.transpose(0, 1)  # (N, B, H)
        x = self.transformer(x)  # (N, B, H)
        
        fused = x.mean(dim=0)  # (B, H)
        return fused