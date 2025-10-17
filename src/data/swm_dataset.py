import os, json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def wav_to_wavlm_path(wav_path: str, src_dir="wav", dst_dir="wavlms_large") -> str:
    p = wav_path.replace(f"/{src_dir}/", f"/{dst_dir}/").replace(f"\\{src_dir}\\", f"\\{dst_dir}\\")
    return p[:-4] + ".npy" if p.endswith(".wav") else p

def wav_to_prosody_path(wav_path: str, src_dir="wav", dst_dir="prosody") -> str:
    p = wav_path.replace(f"/{src_dir}/", f"/{dst_dir}/").replace(f"\\{src_dir}\\", f"\\{dst_dir}\\")
    return p[:-4] + ".npy" if p.endswith(".wav") else p

def load_wavlm_npy(path: str) -> np.ndarray:
    """Return (T,D) float32; auto-transpose if saved as (D,T) or (D,) pooled."""
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]  # (1,D) pooled
    return arr.astype(np.float32)

def load_prosody_npy(path: str) -> np.ndarray:
    """Return (88,) float32 prosody vector."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 1 or arr.shape[0] != 88:
        raise ValueError(f"Prosody shape {arr.shape} != (88,)")
    return arr



class SWMDataset(Dataset):
    def __init__(
        self,
        csv_paths: List[str],
        label_json_path: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 64,
        pool_wavlm: bool = False,
        wav_src_dir: str = "wav",
        wavlm_dir: str = "wavlms_large",
        prosody_dir: str = "prosody",
        text_col: str = "transcription",
        wav_col: str = "wav_path",
        tom_col: str = "ToM",
        sa_col: str = "SA",
        wma_col: str = "WMA",
        prag_col: str = "Prag"
    ):
        super().__init__()

        dfs = [pd.read_csv(csv_path) for csv_path in csv_paths]
        self.df = pd.concat(dfs, ignore_index=True)

        with open(label_json_path,"r",encoding="utf-8") as f:
            label = json.load(f)
        self.tom2id: Dict[str,int]   = label["ToM"]
        self.sa2id: Dict[str,int]    = label["SA"]
        self.wma2id: Dict[str,int]   = label["WMA"]
        self.prag2id: Dict[str,int]  = label["Prag"]

        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.pool_wavlm = pool_wavlm

        # columns & dirs
        self.text_col, self.wav_col = text_col, wav_col
        self.tom_col, self.sa_col, self.wma_col, self.prag_col = tom_col, sa_col, wma_col, prag_col
        self.wav_src_dir, self.wavlm_dir, self.prosody_dir = wav_src_dir, wavlm_dir, prosody_dir
        
        self.wavlm_path_col, self.prosody_path_col = "wavlm_path", "prosody_path"

        # unknown label check
        for name, col, table in [
            ("tom", self.tom_col, self.tom2id),
            ("sa", self.sa_col, self.sa2id),
            ("wma", self.wma_col, self.wma2id),
            ("prag", self.prag_col, self.prag2id),
        ]:
            unknown = sorted(set(map(str, self.df[col].unique())) - set(table.keys()))
            if unknown:
                print(f"[WARN] Unknown {name} in CSV not found in label.json: {unknown}")

    def __len__(self): return len(self.df)

    def _lab(self, key: str, table: Dict[str,int], name:str) -> int:
        s = str(key)
        if s not in table:
            return -1
        return table[s]

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        
        y_tom = self._lab(row[self.tom_col], self.tom2id, "ToM")
        y_sa = self._lab(row[self.sa_col], self.sa2id, "SA")
        y_wma = self._lab(row[self.wma_col], self.wma2id, "WMA")
        y_prag = self._lab(row[self.prag_col], self.prag2id, "Prag")

        if y_tom == -1 or y_sa == -1 or y_wma == -1 or y_prag == -1:
            return None 

        # tokenizer
        enc = self.tok(
            str(row[self.text_col]),
            max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # prosody -> (88,)
        pros_path = row.get(self.prosody_path_col)
        if pd.isna(pros_path): 
            wav_path = str(row[self.wav_col])
            pros_path = wav_to_prosody_path(wav_path, self.wav_src_dir, self.prosody_dir)
        else:
            pros_path = str(pros_path)
        
        if not os.path.exists(pros_path):
            raise FileNotFoundError(f"prosody npy not found: {pros_path}")
        z_prosody = torch.tensor(load_prosody_npy(pros_path), dtype=torch.float32)

        # wavLM -> (T,D) / pooled (D,)
        wavlm_path = row.get(self.wavlm_path_col) 
        if pd.isna(wavlm_path): 
            wav_path = str(row[self.wav_col])
            wavlm_path = wav_to_wavlm_path(wav_path, self.wav_src_dir, self.wavlm_dir)
        else:
            wavlm_path = str(wavlm_path)

        if not os.path.exists(wavlm_path):
            raise FileNotFoundError(f"wavLM npy not found: {wavlm_path}")
        wavlm_frames = torch.from_numpy(load_wavlm_npy(wavlm_path))

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "z_prosody": z_prosody,
            "y_tom":   torch.tensor(y_tom, dtype=torch.long),
            "y_sa": torch.tensor(y_sa, dtype=torch.long),
            "y_wma": torch.tensor(y_wma, dtype=torch.long),
            "y_prag": torch.tensor(y_prag, dtype=torch.long)
        }

        if self.pool_wavlm:
            sample["wavlm_mean"] = wavlm_frames.mean(dim=0)
        else:
            sample["wavlm_frames"] = wavlm_frames
            sample["wavlm_len"] = torch.tensor(wavlm_frames.shape[0], dtype=torch.long)
        return sample


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batch = [b for b in batch if b is not None]
    if not batch: 
        return {}
    
    out: Dict[str, torch.Tensor] = {}

    for k in ["input_ids","attention_mask","z_prosody","y_tom","y_sa","y_wma", "y_prag"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    if "wavlm_mean" in batch[0]:
        out["wavlm_mean"] = torch.stack([b["wavlm_mean"] for b in batch], dim=0)
    else:
        lens = torch.stack([b["wavlm_len"] for b in batch], dim=0)
        max_len = int(lens.max().item())
        feat_dim = batch[0]["wavlm_frames"].shape[1]
        B = len(batch)
        padded = torch.zeros(B, max_len, feat_dim, dtype=batch[0]["wavlm_frames"].dtype)
        for i,b in enumerate(batch):
            T = b["wavlm_frames"].shape[0]
            padded[i, :T] = b["wavlm_frames"]
        out["wavlm_frames"] = padded
        out["wavlm_lens"] = lens
    return out


def build_dataloader(
    csv_paths: List,
    label_json_path: str,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 64,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pool_wavlm: bool = False,
    wav_src_dir: str = "wav",
    wavlm_dir: str = "wavlms_large",
    prosody_dir: str = "prosody",
):
    ds = SWMDataset(
        csv_paths=csv_paths,
        label_json_path=label_json_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        pool_wavlm=pool_wavlm,
        wav_src_dir=wav_src_dir,
        wavlm_dir=wavlm_dir,
        prosody_dir=prosody_dir,
        text_col="transcription",
        wav_col="wav_path",
        tom_col="ToM",
        sa_col="SA",
        wma_col="WMA",
        prag_col="Prag"
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch
    )
    return ds, dl




if __name__ == "__main__":
    CSV = ["xxx.csv"]
    LABEL = "label.json"
    ds, dl = build_dataloader(
        csv_paths=CSV,
        label_json_path=LABEL,
        tokenizer_name="distilbert-base-uncased",
        max_length=64,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pool_wavlm=False,
        wav_src_dir="wav",
        wavlm_dir="wavlms_large", ###16k
        prosody_dir="prosody"
    )
    print("len(ds) =", len(ds))
    b = next(iter(dl))
    for k,v in b.items():
        print(k, tuple(v.shape))
