"""
evaluate.py
-----------
Two evaluation modes:

  1) Re-evaluate a saved checkpoint on a chosen split (top-1, per-class,
     confusion matrix, conformal coverage if a val_loader is provided).

  2) Counterfactual geometry-swap test — falsification check.
     For each test sample, also predict with the WRONG bearing geometry
     vector substituted in. If the geometry-invariant path is doing its
     job (Path B), predictions should NOT change much when we swap the
     geometry. If they change a lot, the model is leaking geometry into
     its fault prediction.

The geometry-swap test is the headline plot of the paper: it lets the
reader inspect how well disentanglement actually worked.

Usage:
    python evaluate.py --protocol P2 --train_ds CWRU MFPT Paderborn --test_ds HUST
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, DEVICE, NUM_WORKERS, N_CLASSES, RESULTS_DIR,
    BEARING_GEOMETRY, geometry_vector, CONFORMAL_ALPHA,
)
from unified_dataset import build_index, BearingDataset, collate
from splits import (
    split_P1_within_dataset, split_P2_cross_dataset,
    split_P3_hust_cross_geometry, split_summary,
)
from physgen_model import PhysGenBearing
from conformal import mondrian_calibrate, mondrian_predict_sets, conformal_metrics


def make_loader(samples, domain_map, shuffle=False):
    ds = BearingDataset(samples)
    def _collate(batch):
        out = collate(batch)
        dom_ids = [domain_map.get((b["dataset"], b["bearing_geom_name"]), 0) for b in batch]
        out["domain_idx"] = torch.tensor(dom_ids, dtype=torch.long)
        return out
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, collate_fn=_collate,
                      pin_memory=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_all, lab_all, geom_all = [], [], []
    for batch in loader:
        signal   = batch["signal"].to(device)
        fs       = batch["fs"].to(device)
        rpm      = batch["rpm"].to(device)
        geom_vec = batch["geom_vec"].to(device)
        out = model(signal, fs, rpm, geom_vec, adv_lambda=0.0)
        probs = F.softmax(out["logits_main"], dim=-1).cpu().numpy()
        probs_all.append(probs)
        lab_all.append(batch["fault"].numpy())
        geom_all.append(batch["geom_vec"].numpy())
    return np.concatenate(probs_all), np.concatenate(lab_all), np.concatenate(geom_all)


@torch.no_grad()
def predict_with_swapped_geom(model, loader, swap_geom_vec, device):
    """
    Run model on every sample but substitute `swap_geom_vec` for the true geom_vec.
    swap_geom_vec: (5,) numpy array.
    """
    model.eval()
    probs_all = []
    swap_t = torch.from_numpy(swap_geom_vec.astype(np.float32))
    for batch in loader:
        signal = batch["signal"].to(device)
        fs     = batch["fs"].to(device)
        rpm    = batch["rpm"].to(device)
        B = signal.size(0)
        gv = swap_t.unsqueeze(0).expand(B, -1).to(device)
        out = model(signal, fs, rpm, gv, adv_lambda=0.0)
        probs_all.append(F.softmax(out["logits_main"], dim=-1).cpu().numpy())
    return np.concatenate(probs_all)


def confusion_and_per_class(probs, labels, n_classes=N_CLASSES):
    preds = probs.argmax(axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    per_class_acc = []
    for c in range(n_classes):
        denom = cm[c].sum()
        per_class_acc.append(float(cm[c, c] / denom) if denom > 0 else float("nan"))
    return cm, per_class_acc


def main(args):
    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    samples = build_index(datasets=tuple(args.use_datasets),
                          paderborn_max_files=args.paderborn_max_files)

    if args.protocol == "P1":
        train_s, val_s, test_s = split_P1_within_dataset(samples, dataset=args.dataset)
        tag = f"P1_{args.dataset}"
    elif args.protocol == "P2":
        train_s, val_s, test_s = split_P2_cross_dataset(
            samples, train_datasets=set(args.train_ds), test_datasets=set(args.test_ds))
        tag = f"P2_train-{'+'.join(args.train_ds)}_test-{'+'.join(args.test_ds)}"
    else:
        train_s, val_s, test_s = split_P3_hust_cross_geometry(samples)
        tag = "P3_HUST_xgeom"
    split_summary(tag, train_s, val_s, test_s)

    ckpt_path = RESULTS_DIR / f"{tag}_best.pt"
    if not ckpt_path.exists():
        print(f"No checkpoint at {ckpt_path}. Train first with train.py.")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    domain_map = ckpt["domain_map"]
    n_domains = max(domain_map.values()) + 1
    model = PhysGenBearing(n_domains=n_domains).to(device)
    model.load_state_dict(ckpt["model"])

    val_loader  = make_loader(val_s, domain_map) if len(val_s) > 0 else None
    test_loader = make_loader(test_s, domain_map)

    # Plain test metrics
    probs, labels, _ = predict(model, test_loader, device)
    top1 = (probs.argmax(1) == labels).mean()
    cm, pca = confusion_and_per_class(probs, labels)
    print(f"\n[eval] top-1 = {top1:.4f}")
    print(f"[eval] per-class acc = {pca}")
    print(f"[eval] confusion matrix:\n{cm}")

    # Conformal
    conf_metrics = None
    if val_loader is not None and len(val_s) >= 50:
        thr = mondrian_calibrate(model, val_loader, N_CLASSES,
                                 alpha=CONFORMAL_ALPHA, device=device)
        pred_sets, probs2, labels2 = mondrian_predict_sets(model, test_loader, thr, device=device)
        conf_metrics = conformal_metrics(pred_sets, probs2, labels2, N_CLASSES)
        print(f"[conformal] {json.dumps(conf_metrics, indent=2)}")

    # ─────────────────────────────────────────────────────────────────────
    # Counterfactual geometry-swap test
    # For each candidate target geometry, run the model with that geom
    # substituted for the true geom — and measure how often predictions agree.
    # Higher agreement = better disentanglement.
    # ─────────────────────────────────────────────────────────────────────
    print("\n[geom-swap] running counterfactual geometry-swap test")
    base_preds = probs.argmax(axis=1)
    swap_results = {}
    for geom_name, geom in BEARING_GEOMETRY.items():
        gv = np.array(geometry_vector(geom), dtype=np.float32)
        swap_probs = predict_with_swapped_geom(model, test_loader, gv, device)
        swap_preds = swap_probs.argmax(axis=1)
        agreement = float((swap_preds == base_preds).mean())
        acc_under_swap = float((swap_preds == labels).mean())
        swap_results[geom_name] = {
            "prediction_agreement": agreement,
            "accuracy_under_swap":  acc_under_swap,
        }
        print(f"  geom={geom_name:10s}  agreement={agreement:.4f}  acc={acc_under_swap:.4f}")

    summary = {
        "tag": tag,
        "test_top1": float(top1),
        "per_class_acc": pca,
        "confusion_matrix": cm.tolist(),
        "conformal": conf_metrics,
        "geom_swap": swap_results,
    }
    out_path = RESULTS_DIR / f"{tag}_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["P1", "P2", "P3"], default="P1")
    ap.add_argument("--dataset", default="CWRU")
    ap.add_argument("--train_ds", nargs="+", default=["CWRU", "MFPT", "Paderborn"])
    ap.add_argument("--test_ds", nargs="+", default=["HUST"])
    ap.add_argument("--use_datasets", nargs="+",
                    default=["CWRU", "HUST", "MFPT", "Paderborn"])
    ap.add_argument("--paderborn_max_files", type=int, default=20)
    args = ap.parse_args()
    main(args)
