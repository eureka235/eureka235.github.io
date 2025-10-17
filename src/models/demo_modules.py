import os, math, random, argparse, json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys



class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", trainable=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # only extract [CLS] token
        h_txt = out.last_hidden_state[:, 0, :]   # (B, H)
        return h_txt


# Mask mean-pool
class WavLMAdapter(nn.Module):
    def __init__(self, in_dim: Optional[int] = None, out_dim: int = 64):
        super().__init__()
        self.proj = None
        if in_dim is not None:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, wavlm_frames=None, wavlm_lens=None, wavlm_mean=None):
        if wavlm_mean is not None:
            x = wavlm_mean  # (B, D)
        elif wavlm_frames is not None and wavlm_lens is not None:
            # mask mean
            B, T, D = wavlm_frames.shape
            device = wavlm_frames.device
            lens = wavlm_lens.to(device).unsqueeze(-1)  # (B,1)
            mask = torch.arange(T, device=device)[None, :] < wavlm_lens[:, None]
            mask = mask.float().unsqueeze(-1)           # (B,T,1)
            s = (wavlm_frames * mask).sum(dim=1)        # (B,D)
            x = s / torch.clamp(lens, min=1)            # (B,D)
        else:
            return None
        if self.proj is not None:
            x = self.proj(x)
        return x  # (B, D)



# ToM: text + z_prosody (+ wavlm) -> emotion logits
class ToMModule(nn.Module):
    def __init__(self, txt_hidden: int, z_dim: int = 88, wavlm_out_dim: Optional[int] = None,
                 tom_hidden: int = 128, num_emotions: int = 7):
        super().__init__()
        in_dim = txt_hidden + z_dim + (wavlm_out_dim or 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, tom_hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(tom_hidden, num_emotions)
        )

    def forward(self, h_txt, z_prosody, wavlm_vec=None):
        feats = [h_txt, z_prosody]
        if wavlm_vec is not None:
            feats.append(wavlm_vec)
        x = torch.cat(feats, dim=-1)
        logits = self.mlp(x)  # (B, E)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs



# speech act: text + ToM -> prag logits
class PragModule(nn.Module):
    def __init__(self, txt_hidden: int, num_emotions: int, prag_hidden: int = 256, num_prag: int = 24):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(txt_hidden + num_emotions, prag_hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(prag_hidden, num_prag)
        )

    def forward(self, h_txt, tom_probs):
        x = torch.cat([h_txt, tom_probs], dim=-1) # [B, P] + [B, E=7]
        logits = self.mlp(x)  # (B, P)
        return logits


class ToM2PragModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text = TextEncoder(config.text_model, trainable=not config.freeze_text)
        self.wavlm = None
        wavlm_out = 0
        if config.use_wavlm:
            self.wavlm = WavLMAdapter(in_dim=config.wavlm_in_dim, out_dim=config.wavlm_out_dim)
            wavlm_out = (config.wavlm_out_dim if config.wavlm_in_dim is not None else 0)

        self.tom = ToMModule(
            txt_hidden=self.text.hidden,
            z_dim=config.z_dim,
            wavlm_out_dim=wavlm_out if config.use_wavlm else None,
            tom_hidden=config.tom_hidden,
            num_emotions=config.num_emotions
        )
        self.prag = PragModule(
            txt_hidden=self.text.hidden,
            num_emotions=config.num_emotions,
            prag_hidden=config.prag_hidden,
            num_prag=config.num_prag
        )
        self.teacher_force_p = config.teacher_force_p

    def forward(self, batch, teacher_forcing: bool = False):
        # text encoding
        h_txt = self.text(batch["input_ids"], batch["attention_mask"])  # (B,H)

        # wavlm adapt (optional)
        wavlm_vec = None
        if self.wavlm is not None:
            if "wavlm_mean" in batch:
                wavlm_vec = self.wavlm(wavlm_mean=batch["wavlm_mean"])
            else:
                wavlm_vec = self.wavlm(batch.get("wavlm_frames"), batch.get("wavlm_lens"))

        # ToM
        tom_logits, tom_probs = self.tom(h_txt, batch["z_prosody"], wavlm_vec)

        # prag, teacher_force: weight (GT tom one-hot, tom_probs)
        if self.training and teacher_forcing and random.random() < self.teacher_force_p:
            B, E = tom_probs.shape
            onehot = torch.zeros_like(tom_probs)
            onehot[torch.arange(B, device=tom_probs.device), batch["y_emotion"]] = 1.0
            tom_input = onehot
        else:
            tom_input = tom_probs.detach()

        prag_logits = self.prag(h_txt, tom_input)
        return tom_logits, prag_logits