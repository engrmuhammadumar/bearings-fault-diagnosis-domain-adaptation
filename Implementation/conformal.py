"""
conformal.py
------------
Mondrian (class-conditional) split conformal prediction.

After training, we hold out a calibration set, compute per-class
nonconformity scores (1 - softmax of true class), and at test time
output a PREDICTION SET that contains the true label with probability
≥ 1 - alpha.

Why Mondrian: marginal split conformal coverage is only guaranteed
ON AVERAGE across classes. In bearing fault diagnosis, healthy class
is over-represented and the "inner" / "outer" / "ball" classes are
what we actually care about. Mondrian gives per-class coverage.

Reference:
  Vovk, Gammerman, Shafer — "Algorithmic Learning in a Random World"
  Angelopoulos & Bates — "Conformal Prediction: A Gentle Introduction"
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def _softmax_probs(model, loader, device):
    """Run model on a loader and return (probs, labels) as numpy arrays."""
    model.eval()
    probs_all, lab_all = [], []
    with torch.no_grad():
        for batch in loader:
            signal   = batch["signal"].to(device)
            fs       = batch["fs"].to(device)
            rpm      = batch["rpm"].to(device)
            geom_vec = batch["geom_vec"].to(device)
            labels   = batch["fault"].to(device)
            out = model(signal, fs, rpm, geom_vec, adv_lambda=0.0)
            probs = F.softmax(out["logits_main"], dim=-1)
            probs_all.append(probs.cpu().numpy())
            lab_all.append(labels.cpu().numpy())
    return np.concatenate(probs_all), np.concatenate(lab_all)


def mondrian_calibrate(model, calib_loader, n_classes, alpha=0.1, device="cuda"):
    """
    Compute class-conditional nonconformity thresholds.

    Returns:
        thresholds: np.ndarray of length n_classes — qhat per class.
    """
    probs, labels = _softmax_probs(model, calib_loader, device)
    # nonconformity score s_i = 1 - p(true class | x_i)
    s = 1.0 - probs[np.arange(len(labels)), labels]

    thresholds = np.full(n_classes, np.inf)
    for c in range(n_classes):
        sc = s[labels == c]
        if len(sc) == 0:
            continue
        n = len(sc)
        # Finite-sample-adjusted quantile
        q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
        thresholds[c] = float(np.quantile(sc, q_level, method="higher"))
    return thresholds


def mondrian_predict_sets(model, test_loader, thresholds, device="cuda"):
    """
    For each test sample, output the set of classes c such that
    p(c | x) >= 1 - thresholds[c].

    Returns:
        pred_sets:  list of sets (one per sample) — each is a Python set of int class ids
        probs:      (N, C) numpy array of softmax probabilities
        labels:     (N,)   numpy array of true labels
    """
    probs, labels = _softmax_probs(model, test_loader, device)
    pred_sets = []
    for i in range(len(probs)):
        s = set()
        for c, t in enumerate(thresholds):
            if probs[i, c] >= 1.0 - t:
                s.add(c)
        if not s:
            # fall back to argmax to guarantee non-empty set
            s.add(int(probs[i].argmax()))
        pred_sets.append(s)
    return pred_sets, probs, labels


def conformal_metrics(pred_sets, probs, labels, n_classes):
    """
    Report:
      - top-1 accuracy (argmax)
      - marginal coverage   = fraction of i where labels[i] ∈ pred_set[i]
      - per-class coverage  = same but conditional on label
      - average set size
      - selective accuracy at singleton sets (the model "knew" the answer)
    """
    top1 = (probs.argmax(axis=1) == labels).mean()
    in_set = np.array([labels[i] in pred_sets[i] for i in range(len(labels))])
    marg_cov = in_set.mean()
    per_class_cov = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            per_class_cov.append(float("nan"))
        else:
            per_class_cov.append(float(in_set[mask].mean()))

    set_sizes = np.array([len(s) for s in pred_sets])
    avg_set = float(set_sizes.mean())

    singleton_mask = set_sizes == 1
    if singleton_mask.sum() > 0:
        sing_acc = in_set[singleton_mask].mean()
        sing_frac = float(singleton_mask.mean())
    else:
        sing_acc = float("nan")
        sing_frac = 0.0

    return {
        "top1_acc": float(top1),
        "marginal_coverage": float(marg_cov),
        "per_class_coverage": per_class_cov,
        "avg_set_size": avg_set,
        "singleton_fraction": sing_frac,
        "singleton_accuracy": float(sing_acc) if not math.isnan(sing_acc) else None,
    }
