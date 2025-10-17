import os, math, random, argparse, json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys

from demo.models.cls_modules import TextEncoder, WavLMAdapter, TemporalToMModule, WMAModule, SAModule, PragModule
from demo.models.fusion import AttentionFusion, GatedFusion, TransformerFusion


class SWMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.text = TextEncoder(
            config.text_model, 
            trainable=not config.freeze_text,
            use_temporal=config.get('use_text_temporal', True)
        )
        
        self.wavlm = None
        wavlm_out = 0
        if config.use_wavlm:
            self.wavlm = WavLMAdapter(
                in_dim=config.wavlm_in_dim, 
                out_dim=config.wavlm_out_dim,
                use_temporal=config.get('use_wavlm_temporal', True)
            )
            wavlm_out = config.wavlm_out_dim if config.wavlm_in_dim is not None else 0
        
        self.is_fusion = config.get('pre_fusion', True)
        self.sa_is_fusion = config.get('sa_fusion', True)
        self.prag_is_fusion = config.get('prag_fusion', True)

        fusion_type = config.get('fusion_type', 'attention')
        
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(
                txt_dim=self.text.hidden,
                prosody_dim=config.z_dim,
                wavlm_dim=wavlm_out,
                hidden_dim=config.get('fusion_hidden', 256)
            )
            fusion_out_dim = config.get('fusion_hidden', 256)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(
                txt_dim=self.text.hidden,
                prosody_dim=config.z_dim,
                wavlm_dim=wavlm_out,
                hidden_dim=config.get('fusion_hidden', 256)
            )
            fusion_out_dim = config.get('fusion_hidden', 256)
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(
                txt_dim=self.text.hidden,
                prosody_dim=config.z_dim,
                wavlm_dim=wavlm_out,
                hidden_dim=config.get('fusion_hidden', 256),
                num_layers=config.get('fusion_layers', 2)
            )
            fusion_out_dim = config.get('fusion_hidden', 256)
        else:
            self.fusion = None
            fusion_out_dim = self.text.hidden + config.z_dim + wavlm_out

        # WMA module
        self.wma = WMAModule(
            txt_hidden=self.text.hidden,
            wavlm_out_dim=wavlm_out if config.use_wavlm else None,
            wma_hidden=config.wma_hidden,
            num_wma=config.num_wma,
            use_temporal_attn=config.get('use_wma_temporal_attn', True)
        )

        # ToM module
        self.tom = TemporalToMModule(
            txt_hidden=fusion_out_dim if self.is_fusion else self.text.hidden,
            z_dim=0 if self.fusion else config.z_dim,
            wavlm_out_dim=0 if self.fusion else (wavlm_out if config.use_wavlm else None),
            tom_hidden=config.tom_hidden,
            num_emotions=config.num_emotions,
            use_temporal_attn=config.get('use_tom_temporal_attn', True)
        )
        
        # SA module
        self.sa = SAModule(
            txt_hidden=fusion_out_dim if self.sa_is_fusion else self.text.hidden,
            num_emotions=config.num_emotions,
            num_wma=config.num_wma,
            sa_hidden=config.sa_hidden,
            num_sa=config.num_sa,
            use_residual=config.get('sa_use_residual', True)
        )

        # Prag module
        self.prag = PragModule(
            txt_hidden=fusion_out_dim if self.prag_is_fusion else self.text.hidden,
            num_emotions=config.num_emotions,
            num_wma=config.num_wma,
            num_sa=config.num_sa,
            prag_hidden=config.prag_hidden,
            num_prag=config.num_prag,
            use_residual=config.get('prag_use_residual', True)
        )
        
        self.tom_teacher_force_p = config.tom_teacher_force_p # for every edge
        self.wma_teacher_force_p = config.wma_teacher_force_p # for every edge
        self.sa_teacher_force_p = config.sa_teacher_force_p # for every edge

        self.fusion_type = fusion_type


    def forward(self, batch, teacher_forcing: bool = False):
        h_txt = self.text(batch["input_ids"], batch["attention_mask"])

        wavlm_vec = None
        if self.wavlm is not None:
            if "wavlm_mean" in batch:
                wavlm_vec = self.wavlm(wavlm_mean=batch["wavlm_mean"])
            else:
                wavlm_vec = self.wavlm(batch.get("wavlm_frames"), batch.get("wavlm_lens"))

        # WMA: text + wavlm
        wma_logits, wma_probs = self.wma(h_txt, wavlm_vec)

        # ToM: text + prosody + wavlm (with or without fusion)
        if self.is_fusion:
            fused_features = self.fusion(h_txt, batch["z_prosody"], wavlm_vec)
            tom_logits, tom_probs = self.tom(fused_features, torch.zeros_like(batch["z_prosody"][:, :0]), None)
        else:
            tom_logits, tom_probs = self.tom(h_txt, batch["z_prosody"], wavlm_vec)

        
        # Teacher forcing for downstream tasks
        if self.training and teacher_forcing and random.random() < self.tom_teacher_force_p:
            # Teacher forcing for ToM
            B, E = tom_probs.shape
            tom_onehot = torch.zeros_like(tom_probs)
            tom_onehot[torch.arange(B, device=tom_probs.device), batch["y_tom"]] = 1.0
            tom_input = tom_onehot
        else:
            tom_input = tom_probs.detach()
        

        if self.training and teacher_forcing and random.random() < self.wma_teacher_force_p:
            # Teacher forcing for WMA
            B, W = wma_probs.shape
            wma_onehot = torch.zeros_like(wma_probs)
            wma_onehot[torch.arange(B, device=wma_probs.device), batch["y_wma"]] = 1.0
            wma_input = wma_onehot
        else:
            wma_input = wma_probs.detach()
        

        # SA: text + WMA + ToM
        if self.sa_is_fusion:
            sa_logits = self.sa(fused_features, tom_input, wma_input)
        else:
            sa_logits = self.sa(h_txt, tom_input, wma_input)

        # Get SA probs for Prag module
        sa_probs = torch.softmax(sa_logits, dim=-1) 
        
        # Teacher forcing for SA
        if self.training and teacher_forcing and random.random() < self.sa_teacher_force_p:
            B, S = sa_probs.shape
            sa_onehot = torch.zeros_like(sa_probs)
            sa_onehot[torch.arange(B, device=sa_probs.device), batch["y_sa"]] = 1.0
            sa_input = sa_onehot
        else:
            sa_input = sa_probs.detach()

        # Prag: text + WMA + ToM + SA
        if self.prag_is_fusion:
            prag_logits = self.prag(fused_features, tom_input, wma_input, sa_input)
        else:
            prag_logits = self.prag(h_txt, tom_input, wma_input, sa_input)
            
        return wma_logits, tom_logits, sa_logits, prag_logits