"""
Evaluation script: runs a checkpoint over an eval protocol, computes pooled EER and min t-DC.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import SSLAASIST
from data import ASVspoofDataset, collate_fn
from metrics import compute_eer, compute_min_tdcf


TABLE2_LA = {
    "sinc_no_sa_no_da":   {"eer": (11.47, 11.95), "tdcf": (0.5081, 0.5139)},
    "wav2vec_no_sa_no_da": {"eer": (6.15, 6.46),   "tdcf": (0.3577, 0.3587)},
    "sinc_sa_no_da":      {"eer": (8.73, 11.61),  "tdcf": (0.4285, 0.5203)},
    "wav2vec_sa_no_da":   {"eer": (4.48, 6.15),   "tdcf": (0.3094, 0.3482)},
    "sinc_sa_da":         {"eer": (7.65, 7.87),   "tdcf": (0.3894, 0.3960)},
    "wav2vec_sa_da":      {"eer": (0.82, 1.00),   "tdcf": (0.2066, 0.2120)},
}

TABLE3_DF = {
    "sinc_no_sa_no_da":    {"eer": (21.06, 22.11)},
    "wav2vec_no_sa_no_da": {"eer": (7.69, 9.48)},
    "sinc_sa_no_da":       {"eer": (23.22, 25.08)},
    "wav2vec_sa_no_da":    {"eer": (4.57, 7.70)},
    "sinc_sa_da_df":       {"eer": (24.42, 25.38)},
    "wav2vec_sa_da_df":    {"eer": (2.85, 3.69)},
}


def get_scores(model, loader, device):
    model.eval()
    bonafide_scores, spoof_scores = [], []
    with torch.no_grad():
        for wavs, labels, _ in loader:
            wavs = wavs.to(device)
            logits = model(wavs)
            # score = bonafide-class logit (index 1) minus spoof-class logit
            # — higher means more bonafide-like, matching compute_eer's convention
            scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            for s, lbl in zip(scores, labels.numpy()):
                (bonafide_scores if lbl == 1 else spoof_scores).append(s)
    return np.array(bonafide_scores), np.array(spoof_scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=[
        "sinc_no_sa_no_da", "wav2vec_no_sa_no_da", "sinc_sa_no_da",
        "wav2vec_sa_no_da", "sinc_sa_da", "wav2vec_sa_da",
        "wav2vec_sa_da_df", "sinc_sa_da_df"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--eval_protocol")
    ap.add_argument("--eval_dir")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--compare_table", choices=["table2", "table3"],
                     help="Skip inference; just print the paper's target table")
    args = ap.parse_args()

    if args.compare_table:
        table = TABLE2_LA if args.compare_table == "table2" else TABLE3_DF
        print(f"\nPaper targets ({'Table 2 — ASVspoof2021 LA' if args.compare_table=='table2' else 'Table 3 — ASVspoof2021 DF'}):")
        print(f"{'config':<22} {'EER best (avg)':<20}" + ("min t-DCF best (avg)" if args.compare_table == "table2" else ""))
        for k, v in table.items():
            eer_str = f"{v['eer'][0]:.2f} ({v['eer'][1]:.2f})"
            tdcf_str = f"{v['tdcf'][0]:.4f} ({v['tdcf'][1]:.4f})" if "tdcf" in v else ""
            print(f"{k:<22} {eer_str:<20} {tdcf_str}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    from model import build_model
    model, _ = build_model(args.config)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)

    ds = ASVspoofDataset(args.eval_protocol, args.eval_dir, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    bonafide_scores, spoof_scores = get_scores(model, loader, device)
    eer, threshold = compute_eer(bonafide_scores, spoof_scores)
    tdcf = compute_min_tdcf(bonafide_scores, spoof_scores)

    print(f"\n=== {args.config} ===")
    print(f"Pooled EER:      {eer:.2f}%  (n_bonafide={len(bonafide_scores)}, n_spoof={len(spoof_scores)})")
    print(f"min t-DCF (approx, nominal ASV op-point — see metrics.py note): {tdcf:.4f}")

    if args.config in TABLE2_LA:
        target = TABLE2_LA[args.config]
        print(f"Paper target:    {target['eer'][0]:.2f}% (best) / {target['eer'][1]:.2f}% (avg)")
        print(f"Delta from best: {eer - target['eer'][0]:+.2f} points")


if __name__ == "__main__":
    main()
