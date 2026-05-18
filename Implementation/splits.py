"""
splits.py
---------
Leakage-free splitting protocols.

We implement three nested protocols (described in the proposal):

  P1 — Within-dataset, bearing-wise leave-out split.
       For a single dataset, train and test on DIFFERENT physical bearings.
       Eliminates the "same bearing in train+test" leakage exposed by
       Vieira et al. (arXiv 2509.22267).

  P2 — Cross-dataset (leave-one-dataset-out).
       Train on, e.g., {CWRU, MFPT, Paderborn-Artificial},
       test on {Paderborn-Real, HUST}.
       Tests true cross-machine generalization.

  P3 — Cross-geometry (within HUST, leave-one-bearing-size-out).
       Train on bearings 6204, 6205, 6207, test on 6206, 6208.
       The hardest task — tests generalization to bearing geometries the
       model has NEVER seen during training.

All splits return (train_samples, val_samples, test_samples) — lists of
sample dicts as produced by unified_dataset.build_index().
"""

import random
from collections import defaultdict


def _stratified_bearing_split(samples, train_frac=0.7, val_frac=0.15, seed=42):
    """
    Split samples by bearing_id such that no bearing appears in two splits.
    Stratified to keep class balance roughly the same across splits.
    """
    rng = random.Random(seed)

    # Group bearing_ids by their (dataset, fault) so we stratify properly
    bearings_by_strat = defaultdict(set)
    for s in samples:
        key = (s["dataset"], s["fault"])
        bearings_by_strat[key].add(s["bearing_id"])

    train_bids, val_bids, test_bids = set(), set(), set()
    for key, bids in bearings_by_strat.items():
        bids = sorted(bids)
        rng.shuffle(bids)
        n = len(bids)
        if n == 1:
            # Can't split a single bearing — give it to train and hope another
            # class provides the val/test signal. Should be very rare.
            train_bids.update(bids)
            continue
        if n == 2:
            train_bids.add(bids[0])
            test_bids.add(bids[1])
            continue
        # n >= 3: try to give at least 1 bearing to each split
        n_test  = max(1, int(round(n * (1.0 - train_frac - val_frac))))
        n_val   = max(1, int(round(n * val_frac)))
        n_train = n - n_val - n_test
        if n_train < 1:
            # ran out — fall back to a sensible default
            n_train = max(1, n - 2)
            n_val   = max(0, (n - n_train) // 2)
            n_test  = max(1, n - n_train - n_val)
        train_bids.update(bids[:n_train])
        val_bids.update(bids[n_train:n_train + n_val])
        test_bids.update(bids[n_train + n_val:])

    train = [s for s in samples if s["bearing_id"] in train_bids]
    val   = [s for s in samples if s["bearing_id"] in val_bids]
    test  = [s for s in samples if s["bearing_id"] in test_bids]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────
# P1
# ─────────────────────────────────────────────────────────────────────────
def split_P1_within_dataset(samples, dataset="CWRU", seed=42):
    """P1: bearing-wise split within ONE dataset."""
    pool = [s for s in samples if s["dataset"] == dataset]
    return _stratified_bearing_split(pool, seed=seed)


# ─────────────────────────────────────────────────────────────────────────
# P2
# ─────────────────────────────────────────────────────────────────────────
def split_P2_cross_dataset(samples, train_datasets, test_datasets,
                            val_frac_within_train=0.15, seed=42):
    """
    P2: cross-dataset split.
      train_datasets:  iterable of dataset names used for training+validation
      test_datasets:   iterable of dataset names used for test
    Validation is held out from the training pool (bearing-wise).
    """
    train_pool = [s for s in samples if s["dataset"] in train_datasets]
    test_pool  = [s for s in samples if s["dataset"] in test_datasets]

    # Bearing-wise val split inside the training pool
    rng = random.Random(seed)
    bearings_by_strat = defaultdict(set)
    for s in train_pool:
        bearings_by_strat[(s["dataset"], s["fault"])].add(s["bearing_id"])

    train_bids, val_bids = set(), set()
    for key, bids in bearings_by_strat.items():
        bids = sorted(bids)
        rng.shuffle(bids)
        n = len(bids)
        n_val = max(1, int(n * val_frac_within_train)) if n >= 2 else 0
        val_bids.update(bids[:n_val])
        train_bids.update(bids[n_val:])

    train = [s for s in train_pool if s["bearing_id"] in train_bids]
    val   = [s for s in train_pool if s["bearing_id"] in val_bids]
    test  = test_pool
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────
# P3
# ─────────────────────────────────────────────────────────────────────────
def split_P3_hust_cross_geometry(samples, train_geoms=("6204", "6205", "6207"),
                                  test_geoms=("6206", "6208"),
                                  val_frac=0.15, seed=42):
    """
    P3: cross-geometry split within HUST.
      Train on subset of HUST bearing sizes, test on the rest.
      This is the hardest test — geometry the model has never seen.
    """
    pool = [s for s in samples if s["dataset"] == "HUST"]
    train_pool = [s for s in pool if s["bearing_geom_name"] in train_geoms]
    test_pool  = [s for s in pool if s["bearing_geom_name"] in test_geoms]

    # Bearing-wise val split inside train_pool
    rng = random.Random(seed)
    bearings_by_strat = defaultdict(set)
    for s in train_pool:
        bearings_by_strat[(s["bearing_geom_name"], s["fault"])].add(s["bearing_id"])

    val_bids = set()
    train_bids = set()
    for key, bids in bearings_by_strat.items():
        bids = sorted(bids)
        rng.shuffle(bids)
        n = len(bids)
        n_val = max(1, int(n * val_frac)) if n >= 2 else 0
        val_bids.update(bids[:n_val])
        train_bids.update(bids[n_val:])

    train = [s for s in train_pool if s["bearing_id"] in train_bids]
    val   = [s for s in train_pool if s["bearing_id"] in val_bids]
    test  = test_pool
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────
# Pretty-print split stats
# ─────────────────────────────────────────────────────────────────────────
def split_summary(name, train, val, test):
    def stats(samples):
        by_class = defaultdict(int)
        bearings = set()
        for s in samples:
            by_class[s["fault"]] += 1
            bearings.add(s["bearing_id"])
        return len(samples), len(bearings), dict(by_class)

    print(f"\n--- Split: {name} ---")
    for tag, S in [("train", train), ("val", val), ("test", test)]:
        n, nb, byc = stats(S)
        print(f"  {tag:5s}: {n:7d} windows | {nb:4d} bearings | classes={byc}")


if __name__ == "__main__":
    from unified_dataset import build_index
    samples = build_index()
    print(f"\nTotal samples: {len(samples)}")

    tr, va, te = split_P1_within_dataset(samples, dataset="CWRU")
    split_summary("P1 within CWRU", tr, va, te)

    tr, va, te = split_P2_cross_dataset(
        samples,
        train_datasets={"CWRU", "MFPT", "Paderborn"},
        test_datasets={"HUST"},
    )
    split_summary("P2 (train=CWRU+MFPT+Paderborn → test=HUST)", tr, va, te)

    tr, va, te = split_P3_hust_cross_geometry(samples)
    split_summary("P3 HUST cross-geometry (train 6204/05/07 → test 6206/08)", tr, va, te)
