"""
Training loop, Section 6.3 (paper-exact hyperparameters):

  "We used the standard Adam optimiser with a fixed learning rate of 0.0001
  for experiments without the wav2vec 2.0 front-end. Since fine-tuning
  demands high GPU computation, experiments with wav2vec 2.0 were performed
  with a smaller batch size of 14 and a lower learning rate of 10^-6...
  All models were trained for 100 epochs on a single GeForce RTX 3090 GPU...
  we performed each experiment with three runs using different random seeds."

  Section 6.3: "It is performed using a weighted cross entropy objective
  function to minimize the training loss."
"""
import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from data import ASVspoofDataset, collate_fn
from rawboost import RAWBOOST_CONFIGS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(dataset, device):
    labels = [lbl for _, lbl in dataset.items]
    n_bonafide = sum(labels)
    n_spoof = len(labels) - n_bonafide
    total = len(labels)
    # inverse-frequency weighting -> "weighted cross entropy" (Section 6.3)
    w_spoof = total / (2 * max(n_spoof, 1))
    w_bonafide = total / (2 * max(n_bonafide, 1))
    return torch.tensor([w_spoof, w_bonafide], dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=[
        "sinc_no_sa_no_da", "wav2vec_no_sa_no_da", "sinc_sa_no_da",
        "wav2vec_sa_no_da", "sinc_sa_da", "wav2vec_sa_da",
        "wav2vec_sa_da_df", "sinc_sa_da_df"])
    ap.add_argument("--train_protocol",
                     default="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
    ap.add_argument("--train_dir", default="data/LA/ASVspoof2019_LA_train/flac")
    ap.add_argument("--dev_protocol",
                     default="data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")
    ap.add_argument("--dev_dir", default="data/LA/ASVspoof2019_LA_dev/flac")
    ap.add_argument("--epochs", type=int, default=100)  # Section 6.3
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None,
                     help="Defaults per-config: 14 for wav2vec2 (Section 6.3), 32 for sinc")
    ap.add_argument("--lr", type=float, default=None,
                     help="Defaults per-config: 1e-6 wav2vec2 / 1e-4 sinc (Section 6.3)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--subset_frac", type=float, default=1.0,
                     help="Run on a fraction of the data first, to validate the pipeline cheaply")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_optimizer", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, da_key = build_model(args.config)
    model.to(device)

    is_wav2vec = "wav2vec" in args.config
    lr = args.lr or (1e-6 if is_wav2vec else 1e-4)  # Section 6.3
    batch_size = args.batch_size or (14 if is_wav2vec else 32)  # Sec 6.3: batch 14 for SSL

    rawboost_fn = RAWBOOST_CONFIGS.get(da_key) if da_key else None

    train_ds = ASVspoofDataset(args.train_protocol, args.train_dir, train=True,
                                rawboost_fn=rawboost_fn, subset_frac=args.subset_frac,
                                seed=args.seed)
    dev_ds = ASVspoofDataset(args.dev_protocol, args.dev_dir, train=False,
                              subset_frac=args.subset_frac, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate_fn,
                               drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    class_weights = get_class_weights(train_ds, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Section 6.3: Adam
    scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    run_dir = os.path.join(args.out_dir, args.config, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    best_dev_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{args.config} seed{args.seed}] epoch {epoch+1}/{args.epochs}")
        for wavs, labels, _ in pbar:
            wavs, labels = wavs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=args.fp16):
                logits = model(wavs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # Validation
        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for wavs, labels, _ in dev_loader:
                wavs, labels = wavs.to(device), labels.to(device)
                logits = model(wavs)
                dev_loss += criterion(logits, labels).item()
        dev_loss /= max(1, len(dev_loader))
        print(f"epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f} dev_loss={dev_loss:.4f}")

        ckpt = {"model_state": model.state_dict(), "epoch": epoch, "config": args.config}
        if args.save_optimizer:
            ckpt["optimizer_state"] = optimizer.state_dict()
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    print(f"Done. Best checkpoint: {os.path.join(run_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
