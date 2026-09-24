"""
Focused variable-speed experiment for Reviewer 1.

HOW TO RUN
----------
1. Run the original notebook/code through the cell that defines:
      extract_features_v3, char_orders, fast_len_leq
2. In the SAME Jupyter kernel run:
      %run comment1_variable_speed_experiment.py

The script deliberately reuses the manuscript's frozen v3 feature extractor.
It writes CSV results plus response_values.txt in:
    results_v2/comment1_variable_speed/

Important: do not change parameters after inspecting the results. If a change
is scientifically necessary, document it and rerun the complete experiment.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# 0. Frozen experiment configuration
# ---------------------------------------------------------------------------
CFG = {
    "seed": 4555951,
    "bearing_key": "PADERBORN_6203",
    "fs_hz": 12_000.0,
    "revolutions": 50,
    "speed_low_hz": 15.0,       # 900 rpm
    "speed_high_hz": 25.0,      # 1500 rpm
    "samples_per_revolution": 256,
    "profiles": ["increasing", "decreasing", "inc_dec", "dec_inc"],
    "speed_errors_pct": [0.0, 0.5, 1.0, 2.0],
    "fault_orders": ["BPFI", "BPFO"],
    "fault_snr_db": -2.0,
    "healthy_trials_per_profile": 40,
    "fault_trials_per_profile_class": 30,
    "calibration_fraction": 0.5,
    "alpha": 0.05,
    "n_null": 99,
}

OUT = Path("results_v2") / "comment1_variable_speed"
OUT.mkdir(parents=True, exist_ok=True)

required = ["extract_features_v3", "char_orders", "fast_len_leq"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the original analysis cells first. Missing: " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# 1. Variable-speed signal simulation
# ---------------------------------------------------------------------------
def _speed_profile(name, u, lo, hi):
    """Smooth positive shaft-frequency trajectory in Hz, u in [0, 1]."""
    smooth = 0.5 - 0.5 * np.cos(np.pi * u)
    if name == "increasing":
        return lo + (hi - lo) * smooth
    if name == "decreasing":
        return hi - (hi - lo) * smooth
    if name == "inc_dec":
        return lo + (hi - lo) * np.sin(np.pi * u) ** 2
    if name == "dec_inc":
        return hi - (hi - lo) * np.sin(np.pi * u) ** 2
    raise ValueError(name)


def _unit_rms(x):
    x = np.asarray(x, float)
    return x / (np.sqrt(np.mean(x * x)) + 1e-12)


def simulate_variable_speed(seed, profile, fault_order=None, snr_db=None):
    """Generate vibration plus the exact instantaneous speed/shaft phase."""
    rng = np.random.default_rng(int(seed))
    fs = float(CFG["fs_hz"])
    lo, hi = CFG["speed_low_hz"], CFG["speed_high_hz"]

    # Use a preliminary duration, then truncate at the requested revolutions.
    duration = 1.20 * CFG["revolutions"] / ((lo + hi) / 2.0)
    n0 = int(np.ceil(duration * fs))
    t0 = np.arange(n0) / fs
    u0 = np.clip(t0 / t0[-1], 0.0, 1.0)
    fr0 = _speed_profile(profile, u0, lo, hi)
    q0 = np.concatenate([[0.0], np.cumsum((fr0[:-1] + fr0[1:]) * 0.5 / fs)])
    stop = int(np.searchsorted(q0, CFG["revolutions"], side="right"))
    if stop >= n0:
        raise RuntimeError("Preliminary signal was too short")

    t = t0[:stop]
    u = np.linspace(0.0, 1.0, stop)
    fr = _speed_profile(profile, u, lo, hi)
    q = np.concatenate([[0.0], np.cumsum((fr[:-1] + fr[1:]) * 0.5 / fs)])
    theta = 2.0 * np.pi * q

    noise = _unit_rms(rng.normal(size=stop))
    shaft = sum(
        (0.18 / h ** 0.8) * np.sin(h * theta + rng.uniform(0, 2*np.pi))
        for h in range(1, 6)
    )
    background = noise + shaft
    background -= background.mean()

    component = np.zeros(stop, float)
    if fault_order is not None:
        order = float(char_orders(CFG["bearing_key"])[fault_order])
        # One impact whenever order*q crosses an integer.
        event_phase = order * q
        event_numbers = np.arange(1, int(np.floor(event_phase[-1])) + 1)
        indices = np.searchsorted(event_phase, event_numbers)
        indices = indices[indices < stop]
        impulses = np.zeros(stop, float)
        impulses[indices] = rng.lognormal(mean=0.0, sigma=0.18, size=len(indices))

        resonance = rng.uniform(1800.0, 4200.0)
        decay = rng.uniform(300.0, 650.0)
        kt = np.arange(int(0.025 * fs)) / fs
        kernel = np.exp(-decay * kt) * np.sin(2*np.pi*resonance*kt)
        component = fftconvolve(impulses, kernel, mode="full")[:stop]
        component = _unit_rms(component)
        scale = 10.0 ** (float(snr_db) / 20.0)
        component *= scale * np.sqrt(np.mean(background**2))

    signal = background + component
    signal -= signal.mean()
    return signal, fr, q


# ---------------------------------------------------------------------------
# 2. Two processing routes
# ---------------------------------------------------------------------------
def angular_resample(signal, q_true, relative_speed_error, samples_per_rev):
    """Resample on estimated shaft angle; error is applied to speed/phase."""
    q_est = q_true * (1.0 + float(relative_speed_error))
    grid = np.arange(0.0, q_est[-1], 1.0 / samples_per_rev)
    return np.interp(grid, q_est, signal), grid


def extract_one(signal, fr_true, q_true, method, error_pct, seed_str):
    if method == "instantaneous_angle":
        x, _ = angular_resample(
            signal, q_true, error_pct / 100.0,
            CFG["samples_per_revolution"]
        )
        # Angle-domain units: fs=samples/revolution and shaft frequency=1 order.
        fs_used = float(CFG["samples_per_revolution"])
        fr_used = 1.0
    elif method == "record_mean":
        x = signal
        fs_used = float(CFG["fs_hz"])
        fr_used = float(np.mean(fr_true))
    else:
        raise ValueError(method)

    return extract_features_v3(
        sig=x,
        fs=fs_used,
        fr=fr_used,
        bearing_key=CFG["bearing_key"],
        seed_str=seed_str,
        n_null=CFG["n_null"],
        alpha=CFG["alpha"],
        return_detail=False,
    )


# ---------------------------------------------------------------------------
# 3. Locked trial registry and execution
# ---------------------------------------------------------------------------
def build_registry():
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for profile in CFG["profiles"]:
        for i in range(CFG["healthy_trials_per_profile"]):
            rows.append(dict(
                trial_id=f"H0_{profile}_{i:03d}",
                seed=int(rng.integers(0, 2**32 - 1)), profile=profile,
                state="healthy", target_order="NONE"
            ))
        for target in CFG["fault_orders"]:
            for i in range(CFG["fault_trials_per_profile_class"]):
                rows.append(dict(
                    trial_id=f"H1_{target}_{profile}_{i:03d}",
                    seed=int(rng.integers(0, 2**32 - 1)), profile=profile,
                    state="fault", target_order=target
                ))
    reg = pd.DataFrame(rows)
    # Fixed per-profile healthy split; faulty trials are never calibration data.
    reg["calibration"] = False
    for profile in CFG["profiles"]:
        idx = reg.index[(reg.profile == profile) & (reg.state == "healthy")]
        n_cal = int(np.floor(len(idx) * CFG["calibration_fraction"]))
        reg.loc[idx[:n_cal], "calibration"] = True
    return reg


def run_experiment():
    registry = build_registry()
    registry.to_csv(OUT / "trial_registry.csv", index=False)
    with open(OUT / "configuration.json", "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2)

    rows, failures = [], []
    started = time.time()
    for j, trial in enumerate(registry.itertuples(index=False), 1):
        fault = None if trial.state == "healthy" else trial.target_order
        snr = None if fault is None else CFG["fault_snr_db"]
        signal, fr, q = simulate_variable_speed(
            trial.seed, trial.profile, fault, snr
        )
        routes = [("record_mean", 0.0)] + [
            ("instantaneous_angle", e) for e in CFG["speed_errors_pct"]
        ]
        for method, error_pct in routes:
            try:
                feat = extract_one(
                    signal, fr, q, method, error_pct,
                    f"C1::{trial.trial_id}::{method}::{error_pct:.3f}"
                )
                rows.append(dict(
                    trial_id=trial.trial_id, profile=trial.profile,
                    state=trial.state, target_order=trial.target_order,
                    calibration=trial.calibration, method=method,
                    speed_error_pct=error_pct,
                    max_z=float(feat["max_z"]),
                    predicted_order=str(feat["argmax_order"]),
                    BPFI_z=float(feat.get("BPFI_z", np.nan)),
                    BPFO_z=float(feat.get("BPFO_z", np.nan)),
                ))
            except Exception as exc:
                failures.append(dict(
                    trial_id=trial.trial_id, method=method,
                    speed_error_pct=error_pct,
                    error_type=type(exc).__name__, error_message=str(exc)
                ))
        if j % 10 == 0 or j == len(registry):
            pd.DataFrame(rows).to_csv(OUT / "trial_features_checkpoint.csv", index=False)
            print(f"{j:4d}/{len(registry)} records; elapsed {(time.time()-started)/60:.1f} min")

    features = pd.DataFrame(rows)
    features.to_csv(OUT / "trial_features.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} extraction failures; inspect failures.csv")
    return features


# ---------------------------------------------------------------------------
# 4. Split-conformal decisions and exact requested metrics
# ---------------------------------------------------------------------------
def add_decisions(features):
    output = []
    for (method, err), group in features.groupby(["method", "speed_error_pct"]):
        bank = group[(group.state == "healthy") & group.calibration].max_z.to_numpy(float)
        if len(bank) < 19:
            raise RuntimeError(f"Calibration bank too small: {method}, {err}, n={len(bank)}")
        test = group[~group.calibration].copy()
        test["conformal_p"] = [
            (1.0 + np.sum(bank >= score)) / (len(bank) + 1.0)
            for score in test.max_z.to_numpy(float)
        ]
        test["alarm"] = test.conformal_p <= CFG["alpha"]
        test["localization_correct"] = (
            (test.state == "fault") &
            (test.predicted_order == test.target_order)
        )
        test["correct_diagnosis"] = test.alarm & test.localization_correct
        output.append(test)
    decisions = pd.concat(output, ignore_index=True)
    decisions.to_csv(OUT / "test_decisions.csv", index=False)
    return decisions


def _rate(frame, column):
    return 100.0 * float(frame[column].mean()) if len(frame) else np.nan


def summarise(decisions):
    rows = []
    for (method, err, profile), g in decisions.groupby(
        ["method", "speed_error_pct", "profile"]
    ):
        h = g[g.state == "healthy"]
        for target in CFG["fault_orders"]:
            f = g[(g.state == "fault") & (g.target_order == target)]
            rows.append(dict(
                method=method, speed_error_pct=err, profile=profile,
                target_order=target, healthy_n=len(h), fault_n=len(f),
                healthy_false_alarm_pct=_rate(h, "alarm"),
                detection_pct=_rate(f, "alarm"),
                localization_pct=_rate(f, "localization_correct"),
                correct_diagnosis_pct=_rate(f, "correct_diagnosis"),
            ))
    by_profile = pd.DataFrame(rows)
    by_profile.to_csv(OUT / "metrics_by_profile.csv", index=False)

    overall = []
    for (method, err), g in decisions.groupby(["method", "speed_error_pct"]):
        h = g[g.state == "healthy"]
        row = dict(
            method=method, speed_error_pct=err,
            maximum_profile_false_alarm_pct=float(
                h.groupby("profile").alarm.mean().max() * 100.0
            ),
            overall_false_alarm_pct=_rate(h, "alarm"),
        )
        for target in CFG["fault_orders"]:
            f = g[(g.state == "fault") & (g.target_order == target)]
            row[f"{target}_detection_pct"] = _rate(f, "alarm")
            row[f"{target}_localization_pct"] = _rate(f, "localization_correct")
            row[f"{target}_correct_diagnosis_pct"] = _rate(f, "correct_diagnosis")
        overall.append(row)
    overall = pd.DataFrame(overall).sort_values(["method", "speed_error_pct"])
    overall.to_csv(OUT / "metrics_overall.csv", index=False)

    angle0 = overall[(overall.method == "instantaneous_angle") &
                     (overall.speed_error_pct == 0.0)].iloc[0]
    mean = overall[overall.method == "record_mean"].iloc[0]
    lines = [
        "VALUES FOR REVIEWER COMMENT 1 (percent)",
        "",
        "Instantaneous-angle processing, 0% speed error:",
        f"  Maximum profile healthy false-alarm rate: {angle0.maximum_profile_false_alarm_pct:.1f}%",
        f"  BPFI/inner-race detection:               {angle0.BPFI_detection_pct:.1f}%",
        f"  BPFO/outer-race detection:               {angle0.BPFO_detection_pct:.1f}%",
        f"  BPFI/inner-race localization:            {angle0.BPFI_localization_pct:.1f}%",
        f"  BPFO/outer-race localization:            {angle0.BPFO_localization_pct:.1f}%",
        f"  BPFI/inner-race correct diagnosis:       {angle0.BPFI_correct_diagnosis_pct:.1f}%",
        f"  BPFO/outer-race correct diagnosis:       {angle0.BPFO_correct_diagnosis_pct:.1f}%",
        "",
        "Record-mean-speed processing:",
        f"  Maximum profile healthy false-alarm rate: {mean.maximum_profile_false_alarm_pct:.1f}%",
        f"  BPFI/inner-race detection:               {mean.BPFI_detection_pct:.1f}%",
        f"  BPFO/outer-race detection:               {mean.BPFO_detection_pct:.1f}%",
        f"  BPFI/inner-race localization:            {mean.BPFI_localization_pct:.1f}%",
        f"  BPFO/outer-race localization:            {mean.BPFO_localization_pct:.1f}%",
        "",
        "Use metrics_overall.csv for the 0%, 0.5%, 1%, and 2% error comparison.",
    ]
    text = "\n".join(lines)
    (OUT / "response_values.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    return by_profile, overall


FEATURES = run_experiment()
DECISIONS = add_decisions(FEATURES)
METRICS_BY_PROFILE, METRICS_OVERALL = summarise(DECISIONS)
print(f"\nAll outputs written to: {OUT.resolve()}")
