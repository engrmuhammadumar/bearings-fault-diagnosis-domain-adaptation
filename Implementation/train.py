"""
train.py
--------
Main training script. Given a (train, val, test) split, train PhysGen-Bearing
with the three losses and save the best checkpoint by validation accuracy.

Usage examples:
    python train.py --protocol P1 --dataset CWRU
    python train.py --protocol P2 --train_ds CWRU MFPT Paderborn --test_ds HUST
    python train.py --protocol P3
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, LAMBDA_PHYS, LAMBDA_ADV,
    DEVICE, SEED, NUM_WORKERS, N_CLASSES, RESULTS_DIR, CONFORMAL_ALPHA,
)
from unified_dataset import build_index, BearingDataset, collate
from splits import (
    split_P1_within_dataset, split_P2_cross_dataset,
    split_P3_hust_cross_geometry, split_summary,
)
from physgen_model import PhysGenBearing
from losses import classification_loss, physics_consistency_loss, adversarial_domain_loss
from conformal import mondrian_calibrate, mondrian_predict_sets, conformal_metrics


# ─────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────
def build_domain_labels(samples):
    """
    Assign an integer domain id to each (dataset, geometry) combination.
    Used as the adversarial classifier's target.
    Returns: dict {(dataset, geom_name): int}, and an int n_domains.
    """
    pairs = sorted({(s["dataset"], s["bearing_geom_name"]) for s in samples})
    domain_map = {p: i for i, p in enumerate(pairs)}
    return domain_map, len(domain_map)


# ─────────────────────────────────────────────────────────────────────────
def make_loader(samples, domain_map, batch_size=BATCH_SIZE, shuffle=True):
    """Wrap a sample list as a DataLoader, attaching domain_idx to each batch."""
    ds = BearingDataset(samples)

    def _collate(batch):
        out = collate(batch)
        # Attach domain_idx
        dom_ids = [
            domain_map.get((b["dataset"], b["bearing_geom_name"]), 0)
            for b in batch
        ]
        out["domain_idx"] = torch.tensor(dom_ids, dtype=torch.long)
        return out

    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=NUM_WORKERS, collate_fn=_collate, drop_last=False,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────
def one_epoch(model, loader, optimizer, device, adv_lambda=1.0, train=True):
    model.train(mode=train)
    totals = defaultdict(float)
    n = 0
    correct = 0
    for batch in loader:
        signal   = batch["signal"].to(device)
        fs       = batch["fs"].to(device)
        rpm      = batch["rpm"].to(device)
        geom_vec = batch["geom_vec"].to(device)
        labels   = batch["fault"].to(device)
        dom_idx  = batch["domain_idx"].to(device)

        out = model(signal, fs, rpm, geom_vec, adv_lambda=adv_lambda)
        loss_cls = classification_loss(out["logits_main"], labels)
        loss_aux = classification_loss(out["logits_aux"],  labels)
        loss_phys = physics_consistency_loss(out["attn"], geom_vec, labels)
        loss_dom = adversarial_domain_loss(out["logits_domain"], dom_idx)
        loss = loss_cls + 0.3 * loss_aux + LAMBDA_PHYS * loss_phys + LAMBDA_ADV * loss_dom

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = labels.size(0)
        totals["loss"]     += loss.item() * bs
        totals["loss_cls"] += loss_cls.item() * bs
        totals["loss_phys"] += loss_phys.item() * bs
        totals["loss_dom"] += loss_dom.item() * bs
        correct += (out["logits_main"].argmax(dim=-1) == labels).sum().item()
        n += bs
    return {k: v / max(1, n) for k, v in totals.items()} | {"acc": correct / max(1, n)}


# ─────────────────────────────────────────────────────────────────────────
def run(args):
    set_seed(SEED)
    device = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    # 1. Load samples
    samples = build_index(force_reload=args.force_reload,
                          datasets=tuple(args.use_datasets),
                          paderborn_max_files=args.paderborn_max_files)

    # 2. Build split
    if args.protocol == "P1":
        train_s, val_s, test_s = split_P1_within_dataset(samples, dataset=args.dataset)
        tag = f"P1_{args.dataset}"
    elif args.protocol == "P2":
        train_s, val_s, test_s = split_P2_cross_dataset(
            samples,
            train_datasets=set(args.train_ds),
            test_datasets=set(args.test_ds),
        )
        tag = f"P2_train-{'+'.join(args.train_ds)}_test-{'+'.join(args.test_ds)}"
    elif args.protocol == "P3":
        train_s, val_s, test_s = split_P3_hust_cross_geometry(samples)
        tag = "P3_HUST_xgeom"
    else:
        raise ValueError(args.protocol)

    split_summary(tag, train_s, val_s, test_s)
    if len(train_s) == 0 or len(test_s) == 0:
        print("Empty split — abort.")
        return

    # 3. Domain map built from full pool (incl. test) so model has a fixed head size
    domain_map, n_domains = build_domain_labels(samples)
    print(f"[domains] n_domains = {n_domains}")

    train_loader = make_loader(train_s, domain_map, shuffle=True)
    val_loader   = make_loader(val_s,   domain_map, shuffle=False) if len(val_s) > 0 else None
    test_loader  = make_loader(test_s,  domain_map, shuffle=False)

    # 4. Model + optimizer
    model = PhysGenBearing(n_domains=n_domains).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # 5. Train
    best_val = -1.0
    best_ckpt = RESULTS_DIR / f"{tag}_best.pt"
    history = []
    for ep in range(EPOCHS):
        adv_lambda = min(1.0, ep / max(1, EPOCHS // 3))   # ramp up adversary
        tr_log = one_epoch(model, train_loader, opt, device, adv_lambda=adv_lambda, train=True)
        sched.step()
        if val_loader is not None:
            va_log = one_epoch(model, val_loader, opt, device, adv_lambda=0.0, train=False)
        else:
            va_log = {"acc": tr_log["acc"], "loss": tr_log["loss"]}
        history.append({"epoch": ep, "train": tr_log, "val": va_log})
        msg = (f"[ep {ep:03d}] train acc={tr_log['acc']:.4f} loss={tr_log['loss']:.4f}"
               f" | val acc={va_log['acc']:.4f}")
        print(msg)

        if va_log["acc"] > best_val:
            best_val = va_log["acc"]
            torch.save({"model": model.state_dict(),
                        "domain_map": domain_map,
                        "epoch": ep, "val_acc": best_val}, best_ckpt)
            print(f"  ↳ saved best ({best_val:.4f}) → {best_ckpt.name}")

    print(f"\n[done] best val acc = {best_val:.4f}")

    # 6. Test with best model
    print("\n[test] loading best checkpoint...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_log = one_epoch(model, test_loader, opt, device, adv_lambda=0.0, train=False)
    print(f"[test] top-1 accuracy = {te_log['acc']:.4f}")

    # 7. Conformal calibration on val_loader
    metrics = {"test_top1": te_log["acc"], "best_val": best_val}
    if val_loader is not None and len(val_s) >= 50:
        print("[conformal] calibrating on val set...")
        thr = mondrian_calibrate(model, val_loader, N_CLASSES,
                                 alpha=CONFORMAL_ALPHA, device=device)
        print(f"  thresholds per class: {thr}")
        pred_sets, probs, labels = mondrian_predict_sets(model, test_loader, thr, device=device)
        cm = conformal_metrics(pred_sets, probs, labels, N_CLASSES)
        print(f"[conformal] {json.dumps(cm, indent=2)}")
        metrics["conformal"] = cm
        metrics["thresholds"] = thr.tolist()

    # 8. Save full results
    out_json = RESULTS_DIR / f"{tag}_results.json"
    with open(out_json, "w") as f:
        json.dump({"tag": tag,
                   "metrics": metrics,
                   "history": history,
                   "args": vars(args)}, f, indent=2)
    print(f"[results] saved → {out_json}")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["P1", "P2", "P3"], default="P1")
    ap.add_argument("--dataset", default="CWRU",
                    help="for P1: which dataset to use")
    ap.add_argument("--train_ds", nargs="+",
                    default=["CWRU", "MFPT", "Paderborn"],
                    help="for P2")
    ap.add_argument("--test_ds", nargs="+",
                    default=["HUST"], help="for P2")
    ap.add_argument("--use_datasets", nargs="+",
                    default=["CWRU", "HUST", "MFPT", "Paderborn"],
                    help="datasets to load into the master index")
    ap.add_argument("--paderborn_max_files", type=int, default=20)
    ap.add_argument("--force_reload", action="store_true")
    args = ap.parse_args()
    run(args)
