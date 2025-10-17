import os, math, random, argparse, json
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm
from datetime import datetime


from demo.data.swm_dataset_final import build_dataloader
from demo.models.full_graph import SWMModel

import setproctitle
process_name = "swm_complete"
setproctitle.setproctitle(process_name)


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()



class SWMTrainer:
    def __init__(self, model: SWMModel, num_train_steps: int, cfg):
        self.model = model.to(cfg.device)
        self.cfg = cfg

        # lr
        enc_params = list(self.model.text.parameters())
        head_params = [p for n,p in self.model.named_parameters() if not n.startswith("text.")]
        self.opt = torch.optim.AdamW([
            {"params": enc_params, "lr": cfg.lr_text},
            {"params": head_params, "lr": cfg.lr_head},
        ], weight_decay=cfg.weight_decay)

        self.ce = nn.CrossEntropyLoss()
        self.sched = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=int(0.06*num_train_steps),
            num_training_steps=num_train_steps
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    def step_batch(self, batch, train=True):
        
        torch.autograd.set_detect_anomaly(True)

        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=self.cfg.amp):
            wma_logits, tom_logits, sa_logits, prag_logits = self.model(batch, teacher_forcing=train)
            
            loss_wma = self.ce(wma_logits, batch["y_wma"])
            loss_tom = self.ce(tom_logits, batch["y_tom"])
            loss_sa = self.ce(sa_logits, batch["y_sa"])
            loss_prag = self.ce(prag_logits, batch["y_prag"])
            
            loss = (loss_wma * self.cfg.lambda_wma + 
                   loss_tom * self.cfg.lambda_tom + 
                   loss_sa * self.cfg.lambda_sa + 
                   loss_prag * self.cfg.lambda_prag)

        if train:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()

        with torch.no_grad():
            wma_acc = (wma_logits.argmax(-1) == batch["y_wma"]).float().mean()
            tom_acc = (tom_logits.argmax(-1) == batch["y_tom"]).float().mean()
            sa_acc = (sa_logits.argmax(-1) == batch["y_sa"]).float().mean()
            prag_acc = (prag_logits.argmax(-1) == batch["y_prag"]).float().mean()
            
        return (loss.item(), loss_wma.item(), loss_tom.item(), loss_sa.item(), loss_prag.item(),
                wma_acc.item(), tom_acc.item(), sa_acc.item(), prag_acc.item())

    def fit(self, train_loader: DataLoader, dev_loader: Optional[DataLoader]=None):
        best = -1.0  # mean acc
        gstep = 0
        
        for ep in range(1, self.cfg.epochs+1):
            self.model.train()

            m_loss = m_loss_wma = m_loss_tom = m_loss_sa = m_loss_prag = 0
            m_wma = m_tom = m_sa = m_prag = 0
            n = 0
            
            for batch in tqdm(train_loader, desc=f"Training Epoch {ep}", leave=False):
                gstep += 1
                (loss, loss_wma, loss_tom, loss_sa, loss_prag, 
                 wma_acc, tom_acc, sa_acc, prag_acc) = self.step_batch(batch, train=True)
                
                m_loss += loss
                m_loss_wma += loss_wma; m_loss_tom += loss_tom
                m_loss_sa += loss_sa; m_loss_prag += loss_prag
                m_wma += wma_acc; m_tom += tom_acc
                m_sa += sa_acc; m_prag += prag_acc
                n += 1
                
                if gstep % self.cfg.log_every == 0:
                    print(f"[ep{ep} step{gstep}] loss={m_loss/n:.3f} "
                          f"wma={m_wma/n:.3f} tom={m_tom/n:.3f} sa={m_sa/n:.3f} prag={m_prag/n:.3f}")
                    print(f"  losses: wma={m_loss_wma/n:.3f} tom={m_loss_tom/n:.3f} "
                          f"sa={m_loss_sa/n:.3f} prag={m_loss_prag/n:.3f}")


            if dev_loader:
                self.model.eval()
                d_loss = d_loss_wma = d_loss_tom = d_loss_sa = d_loss_prag = 0
                d_wma = d_tom = d_sa = d_prag = 0
                dn = 0
                
                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc=f"Validating Epoch {ep}", leave=False):
                        (loss, loss_wma, loss_tom, loss_sa, loss_prag, 
                         wma_acc, tom_acc, sa_acc, prag_acc) = self.step_batch(batch, train=False)
                        
                        d_loss += loss
                        d_loss_wma += loss_wma; d_loss_tom += loss_tom
                        d_loss_sa += loss_sa; d_loss_prag += loss_prag
                        d_wma += wma_acc; d_tom += tom_acc
                        d_sa += sa_acc; d_prag += prag_acc
                        dn += 1
                
                avg_loss = d_loss/dn
                avg_wma = d_wma/dn; avg_tom = d_tom/dn
                avg_sa = d_sa/dn; avg_prag = d_prag/dn
                avg_acc = (avg_wma + avg_tom + avg_sa + avg_prag) / 4 
                
                print(f"[DEV ep{ep}] loss={avg_loss:.3f} avg_acc={avg_acc:.3f}")
                print(f"  accuracies: wma={avg_wma:.3f} tom={avg_tom:.3f} sa={avg_sa:.3f} prag={avg_prag:.3f}")
                print(f"  losses: wma={d_loss_wma/dn:.3f} tom={d_loss_tom/dn:.3f} "
                      f"sa={d_loss_sa/dn:.3f} prag={d_loss_prag/dn:.3f}")
                
                if avg_acc > best:
                    best = avg_acc
                    torch.save(self.model.state_dict(), self.cfg.out_ckpt)
                    print(f">> saved best model (avg_acc={best:.3f}) to {self.cfg.out_ckpt}")



from demo.models.config import Cfg_SWM
from demo.eval import EnhancedSWMTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", 
                       default=["xxx.csv"])
    parser.add_argument("--dev_csv",
                       default=["yyy.csv"])
    
    parser.add_argument("--label_json", default="data/label.json")
    
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_text", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    
    parser.add_argument("--tom_teacher_force_p", type=float, default=0.3)
    parser.add_argument("--wma_teacher_force_p", type=float, default=0.3)
    parser.add_argument("--sa_teacher_force_p", type=float, default=0.3)
    
    parser.add_argument("--use_wavlm", action="store_true")
    parser.add_argument("--freeze_text", action="store_true") ### freeze text encoder
    
    parser.add_argument("--fusion_type", choices=['attention', 'gated', 'transformer'], default="gated")
    
    parser.add_argument("--lambda_wma", type=float, default=1.0)
    parser.add_argument("--lambda_tom", type=float, default=1.0)
    parser.add_argument("--lambda_sa", type=float, default=1.0)
    parser.add_argument("--lambda_prag", type=float, default=1.0)
    
    parser.add_argument("--run_name", default="full/baseline-tf0.3-gated-tomxsa")
    args = parser.parse_args()


    cfg = Cfg_SWM()
    
    cfg.train_csv = args.train_csv
    cfg.dev_csv = args.dev_csv
    cfg.label_json = args.label_json
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.device = args.device
    cfg.lr_text = args.lr_text
    cfg.lr_head = args.lr_head
    cfg.use_wavlm = args.use_wavlm
    cfg.freeze_text = args.freeze_text
    cfg.fusion_type = args.fusion_type
    cfg.tom_teacher_force_p = args.tom_teacher_force_p
    cfg.wma_teacher_force_p = args.wma_teacher_force_p
    cfg.sa_teacher_force_p = args.sa_teacher_force_p
    cfg.lambda_wma = args.lambda_wma
    cfg.lambda_tom = args.lambda_tom
    cfg.lambda_sa = args.lambda_sa
    cfg.lambda_prag = args.lambda_prag
    # cfg.out_ckpt = args.out_ckpt


    # logs
    base_dir = "saved_models"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name.strip() or f"run-{ts}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)

    log_path = os.path.join(run_dir, "train.log")
    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    print(f">> Run directory created: {run_dir}")
    print(f">> Logging to: {log_path}")

    cfg.out_ckpt = os.path.join(run_dir, "best.pt")

    cfg_json_path = os.path.join(run_dir, "config.json")
    try:
        cfg_dict = cfg.__dict__.copy()
        cfg_dict["out_ckpt"] = cfg.out_ckpt
        with open(cfg_json_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        print(f">> Saved config to: {cfg_json_path}")
    except Exception as e:
        print(f"!! Failed to save config: {e}")


    train_ds, train_loader = build_dataloader(
        csv_paths=cfg.train_csv, label_json_path=cfg.label_json,
        batch_size=cfg.batch_size, shuffle=True, pool_wavlm=not cfg.use_wavlm,
    )
    dev_ds, dev_loader = build_dataloader(
        csv_paths=cfg.dev_csv, label_json_path=cfg.label_json,
        batch_size=cfg.batch_size, shuffle=False, pool_wavlm=not cfg.use_wavlm,
    )

    print(f"Training samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")


    steps_per_epoch = math.ceil(len(train_ds)/cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"Total training steps: {total_steps}")

    model = SWMModel(cfg)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # trainer = SWMTrainer(model, total_steps, cfg)
    trainer = EnhancedSWMTrainer(model, total_steps, cfg)
    trainer.fit(train_loader, dev_loader)


    print(f">> Finished. Best checkpoint (if saved) at: {cfg.out_ckpt}")
    print(f">> Logs available at: {log_path}")


if __name__ == "__main__":
    main()