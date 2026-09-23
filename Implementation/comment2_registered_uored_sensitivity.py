"""Post-review sensitivity of the REGISTERED UORED v5 pipeline.

Run in the original notebook kernel after the final amended Cells 15B and 16.
    %run comment2_registered_uored_sensitivity.py

Requires UORED_ROOT, UORED_SEGMENT_FEATURES (the 300 amended rows),
UORED_UNIT_STATE_STATISTICS, UORED_MANIFEST, UORED_BEARING_KEY,
extract_features_v3, usable_orders_v3, fast_len_leq and band_list.

The script verifies that reaggregation reproduces the registered baseline,
then varies one parameter at a time. It does not alter frozen outputs.
Every new setting uses the same 20 physical bearings, the same source CSVs,
new features where required, and a recalculated 19-unit healthy bank.
"""

from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

NEEDED = ["UORED_ROOT", "UORED_MANIFEST", "UORED_SEGMENT_FEATURES",
          "UORED_UNIT_STATE_STATISTICS", "UORED_BEARING_KEY",
          "extract_features_v3", "usable_orders_v3", "fast_len_leq", "band_list"]
missing = [name for name in NEEDED if name not in globals()]
if missing:
    raise RuntimeError("Run the final amended UORED extraction cells first: "
                       + ", ".join(missing))

OUT = Path("results_v2/comment2_registered_uored_sensitivity")
OUT.mkdir(parents=True, exist_ok=True)
BASE = pd.DataFrame(UORED_SEGMENT_FEATURES).copy()
ORIGINAL = pd.DataFrame(UORED_UNIT_STATE_STATISTICS).copy()
MANIFEST = pd.DataFrame(UORED_MANIFEST).copy()
if len(BASE) != 300 or BASE.record_id.nunique() != 60 or \
        not BASE.groupby("record_id").size().eq(5).all():
    raise RuntimeError("Expected the amended 300-row/60-state/5-segment UORED cache")
if len(ORIGINAL) != 60 or len(MANIFEST) != 60:
    raise RuntimeError("Expected the final amended 60-row state and manifest tables")
if not {"max_z", "BPFI_z", "BPFO_z", "record_id", "unit", "state",
        "target_order", "primary_test_state"}.issubset(BASE.columns):
    raise RuntimeError("Amended segment feature cache lacks required columns")
if "relative_path" not in MANIFEST or "record_id" not in MANIFEST:
    raise RuntimeError("Manifest requires record_id and relative_path")
BASE_TRIAL_ID = BASE.set_index(["record_id", "segment_index"])["trial_id"].to_dict()

# These are sensitivity probes, not replacements for the frozen method.
REF = dict(revolutions=54, order_tolerance=0.03, carrier_bank="full",
           n_null=199, aggregation_q=0.90)
LEVELS = {
    "revolutions": [45, 50, 54, 60],
    "order_tolerance": [0.02, 0.03, 0.04],
    "carrier_bank": ["full", "remove_low_quartile", "remove_high_quartile",
                     "every_second"],
    "n_null": [99, 199, 399],
    "aggregation_q": [0.75, 0.90, 1.00],
}
settings = [dict(setting="reference", factor="reference", level="reference", **REF)]
for factor, choices in LEVELS.items():
    for level in choices:
        if level == REF[factor]:
            continue
        varied = dict(REF)
        varied[factor] = level
        settings.append(dict(setting=f"{factor}__{level}", factor=factor,
                             level=str(level), **varied))
SETTINGS = pd.DataFrame(settings)
SETTINGS.to_csv(OUT / "settings.csv", index=False)
pd.DataFrame([dict(parameter=k, levels=str(v), reference=REF[k])
              for k, v in LEVELS.items()]).to_csv(OUT / "protocol.csv", index=False)


def digest_file(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            sha.update(block)
    return sha.hexdigest()


def check_source(row):
    path = Path(UORED_ROOT) / str(row.relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if hasattr(row, "file_sha256") and pd.notna(row.file_sha256):
        if digest_file(path) != str(row.file_sha256):
            raise RuntimeError(f"Source hash changed: {path}")
    return path


def audit_setting(setting):
    for row in MANIFEST.itertuples(index=False):
        fs = float(getattr(row, "fs", 42000.0))
        if not np.isfinite(fs): fs = 42000.0
        rpm = (float(row.rpm_median) if getattr(row, "rpm_source", "") ==
               "csv_hall_effect_rpm" and pd.notna(getattr(row, "rpm_median", np.nan))
               else 1750.0)
        fr = rpm / 60.0
        length = int(fast_len_leq(int(setting.revolutions * fs / fr)))
        kept, _ = usable_orders_v3(
            bearing_key=UORED_BEARING_KEY, fr=fr,
            df_resolution=fs / length,
            order_tolerance=float(setting.order_tolerance),
            min_collision_bins=2.0, collision_guard_bins=1.0,
        )
        if not {"BPFI", "BPFO"}.issubset(kept):
            return False, f"BPFI/BPFO unresolved for {row.record_id}; kept={sorted(kept)}"
        if int(row.row_count) // length < 4:
            return False, f"fewer than four complete segments for {row.record_id}"
    return True, "admissible"


def selected_bands(fs, variant, original):
    bands = list(original(fs))
    if variant == "full": return bands
    centres = np.asarray([(lo+hi)/2 for lo, hi in bands])
    loq, hiq = np.quantile(centres, [0.25, 0.75])
    if variant == "remove_low_quartile":
        return [b for b, c in zip(bands, centres) if c >= loq]
    if variant == "remove_high_quartile":
        return [b for b, c in zip(bands, centres) if c <= hiq]
    if variant == "every_second": return bands[::2]
    raise ValueError(variant)


def extract_variant(signal, fs, fr, seed, setting):
    # Patch the exact global namespace used by the existing extractor.
    namespace = extract_features_v3.__globals__
    native = namespace["band_list"]
    namespace["band_list"] = lambda sampling_rate, *a, **k: selected_bands(
        sampling_rate, setting.carrier_bank, native)
    try:
        return extract_features_v3(
            sig=signal, fs=fs, fr=fr, bearing_key=UORED_BEARING_KEY,
            seed_str=seed, n_null=int(setting.n_null), alpha=0.05,
            order_tolerance=float(setting.order_tolerance), return_detail=False)
    finally:
        namespace["band_list"] = native


def rows_for_setting(setting):
    if setting.setting == "reference" or setting.factor == "aggregation_q":
        return BASE.copy()
    path = OUT / f"features_{setting.setting}.csv"
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(cache.trial_id.astype(str)) if len(cache) else set()
    output = cache.to_dict("records") if len(cache) else []
    expected_count = 0
    for number, m in enumerate(MANIFEST.itertuples(index=False), 1):
        src = check_source(m)
        fs = 42000.0
        rpm = (float(m.rpm_median) if getattr(m, "rpm_source", "") ==
               "csv_hall_effect_rpm" and pd.notna(getattr(m, "rpm_median", np.nan))
               else 1750.0)
        fr = rpm / 60.0
        length = int(fast_len_leq(int(setting.revolutions * fs / fr)))
        raw = pd.read_csv(src, sep=str(m.csv_delimiter),
                          encoding=str(m.csv_encoding),
                          usecols=[str(m.vibration_column)], low_memory=False)
        vibration = pd.to_numeric(raw[str(m.vibration_column)],
                                  errors="raise").to_numpy(float)
        if len(vibration) != int(m.row_count) or not np.isfinite(vibration).all():
            raise RuntimeError(f"Invalid source signal {m.record_id}")
        starts = np.arange(0, len(vibration)-length+1, length)
        expected_count += len(starts)
        for index, start in enumerate(starts):
            trial_id = f"{m.record_id}::{int(setting.revolutions)}::{index}::{start}"
            if trial_id in done: continue
            try:
                original_trial = BASE_TRIAL_ID.get((m.record_id, index))
                null_seed = (f"UORED-V5::{original_trial}"
                             if int(setting.revolutions) == 54 and original_trial
                             else f"UORED-V5-SENS::{trial_id}")
                feature = extract_variant(
                    vibration[int(start):int(start)+length], fs, fr,
                    null_seed, setting)
                row = dict(setting=setting.setting, trial_id=trial_id,
                           record_id=m.record_id, unit=m.unit,
                           state=m.state, fault_label=m.fault_label,
                           target_order=m.target_order,
                           primary_test_state=m.primary_test_state,
                           segment_index=index, max_z=float(feature["max_z"]),
                           BPFI_z=float(feature["BPFI_z"]),
                           BPFO_z=float(feature["BPFO_z"]))
                if not np.isfinite([row["max_z"],row["BPFI_z"],row["BPFO_z"]]).all():
                    raise ValueError("Nonfinite feature")
                output.append(row)
                done.add(trial_id)
            except Exception as exc:
                pd.DataFrame(output).to_csv(path, index=False)
                raise RuntimeError(f"{setting.setting}, {trial_id}: {exc}") from exc
        pd.DataFrame(output).to_csv(path, index=False)
        print(f"{setting.setting}: {number}/60 states processed")
    frame = pd.DataFrame(output)
    if len(frame) != expected_count or frame.trial_id.duplicated().any():
        raise RuntimeError(f"Incomplete cache for {setting.setting}")
    return frame


def states_from_segments(features, q):
    rows = []
    for record_id, g in features.groupby("record_id", sort=True):
        one = g.iloc[0]
        rows.append(dict(record_id=record_id, unit=one.unit,
                         state=str(one.state).lower(),
                         fault_label=one.fault_label,
                         target_order=one.target_order,
                         primary_test_state=one.primary_test_state,
                         n_segments=len(g),
                         T=float(np.quantile(g.max_z, q, method="linear")),
                         BPFI_z_med=float(np.median(g.BPFI_z)),
                         BPFO_z_med=float(np.median(g.BPFO_z))))
    states = pd.DataFrame(rows)
    if len(states) != 60 or states.unit.nunique() != 20:
        raise RuntimeError("Expected exactly 60 states from 20 physical bearings")
    return states


def decisions(states):
    healthy = states[states.state == "healthy"].set_index("unit")
    if len(healthy) != 20: raise RuntimeError("Expected 20 healthy bearing banks")
    out=[]
    for row in states.itertuples(index=False):
        bank = healthy.drop(index=row.unit)["T"].to_numpy(float)
        p = (1 + int(np.sum(bank >= row.T))) / 20.0
        tie = bool(np.isclose(row.BPFI_z_med, row.BPFO_z_med))
        pred = ("INDETERMINATE" if tie else
                "BPFI" if row.BPFI_z_med > row.BPFO_z_med else "BPFO")
        out.append(dict(**row._asdict(), p_conformal=p,
                        alarm=int(p <= 0.05), predicted_order=pred,
                        localization_correct=int(row.state != "healthy" and
                                                 pred == row.target_order),
                        correct_diagnosis=int(row.state != "healthy" and
                                              p <= 0.05 and pred == row.target_order)))
    return pd.DataFrame(out)


def ci_exact(k, n):
    a=.025
    return (0.0 if k==0 else float(beta.ppf(a,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(1-a,k+1,n-k)))


def metrics(table, setting):
    rows=[]
    groups=[("Healthy", table[table.state=="healthy"], "alarm", "False alarm")]
    for label in ("IR", "OR"):
        for state in ("developing", "faulty"):
            g=table[(table.fault_label==label)&(table.state==state)&
                    (table.primary_test_state.astype(str).str.lower().isin(["true","1"]))]
            for col, metric in (("alarm","Detection"),
                                ("localization_correct","Localization"),
                                ("correct_diagnosis","Correct diagnosis")):
                groups.append((f"{label} {state}",g,col,metric))
    for cohort,g,col,metric in groups:
        if not len(g): raise RuntimeError(f"Empty primary cohort: {cohort}")
        k=int(g[col].sum()); n=len(g); lo,hi=ci_exact(k,n)
        rows.append(dict(setting=setting.setting,factor=setting.factor,
                         level=setting.level,cohort=cohort,metric=metric,
                         successes=k,n=n,rate_pct=100*k/n,
                         ci_low_pct=100*lo,ci_high_pct=100*hi))
    return pd.DataFrame(rows)


# FIRST verify the saved registered baseline exactly before any new extraction.
reference_states = states_from_segments(BASE, .90).set_index("record_id")
frozen = ORIGINAL.set_index("record_id")
for new_col, old_col in (("T","unit_detection_max_z_q90"),
                         ("BPFI_z_med","unit_BPFI_z_median"),
                         ("BPFO_z_med","unit_BPFO_z_median")):
    if not np.allclose(reference_states.loc[frozen.index,new_col],
                       pd.to_numeric(frozen[old_col]), rtol=0, atol=1e-9):
        raise RuntimeError(f"Registered baseline mismatch: {old_col}")
print("Registered 60-state baseline reproduced from original segment cache")

all_metrics=[]; status=[]
for setting in SETTINGS.itertuples(index=False):
    ok,reason=audit_setting(setting)
    if not ok:
        status.append(dict(setting=setting.setting,status="UNRESOLVED",detail=reason))
        print(f"{setting.setting}: UNRESOLVED: {reason}")
        continue
    f=rows_for_setting(setting)
    s=states_from_segments(f,float(setting.aggregation_q))
    d=decisions(s)
    d.to_csv(OUT / f"decisions_{setting.setting}.csv",index=False)
    all_metrics.append(metrics(d,setting))
    status.append(dict(setting=setting.setting,status="EVALUATED",detail=""))

M=pd.concat(all_metrics,ignore_index=True)
STATUS=pd.DataFrame(status)
M.to_csv(OUT / "metrics_long.csv",index=False)
STATUS.to_csv(OUT / "setting_status.csv",index=False)
wide=M.pivot(index=["setting","factor","level"],
             columns=["cohort","metric"],values="rate_pct")
wide.columns=["_".join(c).replace(" ","_") for c in wide.columns]
wide.reset_index().to_csv(OUT / "reviewer_table.csv",index=False)

ref=M[M.setting=="reference"][["cohort","metric","rate_pct"]].rename(
    columns={"rate_pct":"reference_pct"})
DELTA=M.merge(ref,on=["cohort","metric"])
DELTA["delta_pp"]=DELTA.rate_pct-DELTA.reference_pct
DELTA.to_csv(OUT / "paired_percentage_point_changes.csv",index=False)

fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout="constrained")
for ax,metric in zip(axes,("Detection","Correct diagnosis")):
    sub=M[(M.metric==metric)&(M.cohort!="Healthy")]
    x=np.arange(len(SETTINGS)); width=.18
    for i,(name,g) in enumerate(sub.groupby("cohort",sort=True)):
        vals=g.set_index("setting").reindex(SETTINGS.setting).rate_pct
        ax.bar(x+(i-1.5)*width,vals,width,label=name)
    ax.set(title=metric,ylabel="Bearing-level rate (%)",ylim=(0,105),
           xticks=x,xticklabels=SETTINGS.setting)
    ax.tick_params(axis="x",rotation=80,labelsize=7)
axes[0].legend(fontsize=8)
fig.savefig(OUT/"sensitivity_diagnosis.png",dpi=300,bbox_inches="tight")
plt.close(fig)

fig,ax=plt.subplots(figsize=(11,4.5),layout="constrained")
h=M[(M.cohort=="Healthy")&(M.metric=="False alarm")]
vals=h.set_index("setting").reindex(SETTINGS.setting).rate_pct
ax.bar(np.arange(len(vals)),vals,color="#406b9a")
ax.axhline(5,color="black",ls="--",lw=1,label="Nominal 5%")
ax.set(ylabel="Healthy bearing false alarm (%)",xticks=np.arange(len(vals)),
       xticklabels=SETTINGS.setting)
ax.tick_params(axis="x",rotation=80,labelsize=7)
ax.legend()
fig.savefig(OUT/"sensitivity_false_alarms.png",dpi=300,bbox_inches="tight")
plt.close(fig)
print("\nSensitivity results:\n",M.to_string(index=False))
print("\nSaved tables and figures in",OUT.resolve())
