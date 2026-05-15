"""
config.py
---------
Central configuration: paths, bearing geometries, fault-label mappings.

EDIT THE PATHS BELOW IF YOUR DATA LIVES SOMEWHERE ELSE.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"F:\Umar-Wisal-Work\Datasets")

PATHS = {
    "CWRU":      DATA_ROOT / "CWRU",
    "HUST":      DATA_ROOT / "HUST bearing dataset",
    "MFPT":      DATA_ROOT / "MFPT Fault Data Sets",
    "Paderborn": DATA_ROOT / "Paderborn",
}

CACHE_DIR = DATA_ROOT / "physgen_cache"   # pre-windowed numpy cache
RESULTS_DIR = DATA_ROOT / "physgen_results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# Unified fault classes (the labels the model predicts)
# We collapse to 4 canonical classes — every dataset can be mapped to these.
# ─────────────────────────────────────────────────────────────────────────
FAULT_CLASSES = ["healthy", "inner", "outer", "ball"]
FAULT2IDX = {c: i for i, c in enumerate(FAULT_CLASSES)}
N_CLASSES = len(FAULT_CLASSES)


# ─────────────────────────────────────────────────────────────────────────
# Bearing geometry parameters
# (D_pitch [mm], D_ball [mm], n_balls, contact_angle [rad])
# Used to compute BPFO, BPFI, BSF, FTF analytically.
# ─────────────────────────────────────────────────────────────────────────
import math

BEARING_GEOMETRY = {
    # CWRU drive-end uses SKF 6205-2RS JEM
    "SKF_6205":   dict(D_pitch=39.04, D_ball=7.94,  n_balls=9,  alpha=0.0),
    # MFPT: NICE bearing (numbers from MFPT documentation)
    "NICE":       dict(D_pitch=31.62, D_ball=5.97,  n_balls=8,  alpha=0.0),
    # Paderborn: 6203
    "6203":       dict(D_pitch=28.50, D_ball=6.75,  n_balls=8,  alpha=0.0),
    # HUST: 6204, 6205, 6206, 6207, 6208
    "6204":       dict(D_pitch=33.50, D_ball=7.94,  n_balls=8,  alpha=0.0),
    "6205":       dict(D_pitch=39.04, D_ball=7.94,  n_balls=9,  alpha=0.0),
    "6206":       dict(D_pitch=46.40, D_ball=9.53,  n_balls=9,  alpha=0.0),
    "6207":       dict(D_pitch=53.50, D_ball=11.11, n_balls=9,  alpha=0.0),
    "6208":       dict(D_pitch=60.00, D_ball=11.91, n_balls=9,  alpha=0.0),
}


def bearing_orders(geom):
    """
    Compute fault characteristic frequencies as ORDERS (i.e. multiples of
    shaft rotation frequency f_r). Output is f_r-independent.

    Returns dict with keys BPFO, BPFI, BSF, FTF (orders per shaft revolution).
    """
    Db, Dp, n, a = geom["D_ball"], geom["D_pitch"], geom["n_balls"], geom["alpha"]
    cosa = math.cos(a)
    ratio = Db / Dp
    bpfo = (n / 2.0) * (1.0 - ratio * cosa)
    bpfi = (n / 2.0) * (1.0 + ratio * cosa)
    bsf  = (Dp / (2.0 * Db)) * (1.0 - (ratio * cosa) ** 2)
    ftf  = 0.5 * (1.0 - ratio * cosa)
    return dict(BPFO=bpfo, BPFI=bpfi, BSF=bsf, FTF=ftf)


# Pre-compute geometry feature vector used as model conditioning input
def geometry_vector(geom):
    """Return a fixed-length feature vector [BPFO, BPFI, BSF, FTF, n_balls/20]."""
    orders = bearing_orders(geom)
    return [orders["BPFO"], orders["BPFI"], orders["BSF"], orders["FTF"],
            geom["n_balls"] / 20.0]


GEOM_VEC_DIM = 5


# ─────────────────────────────────────────────────────────────────────────
# Paderborn split (official) — artificial vs real damage
# Source: Lessmeier et al. 2016, "Condition Monitoring of Bearing Damage..."
# ─────────────────────────────────────────────────────────────────────────
PADERBORN_HEALTHY    = ["K001", "K002", "K003", "K004", "K005", "K006"]
PADERBORN_ARTIFICIAL = ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                       "KI01", "KI03", "KI04", "KI05", "KI07", "KI08"]
PADERBORN_REAL       = ["KA04", "KA15", "KA16", "KA22", "KA30",
                       "KB23", "KB24", "KB27",
                       "KI14", "KI16", "KI17", "KI18", "KI21"]

# Bearing-code → fault class mapping for Paderborn
def paderborn_fault(code):
    if code in PADERBORN_HEALTHY:
        return "healthy"
    if code.startswith("KA") or code.startswith("KB"):
        # KB = combined inner+outer; we treat them as outer for the 4-class task
        return "outer"
    if code.startswith("KI"):
        return "inner"
    raise ValueError(f"Unknown Paderborn code: {code}")


# ─────────────────────────────────────────────────────────────────────────
# HUST filename decoding
# Filename pattern: <FaultCode><BearingSize><Load>.mat
#   FaultCode ∈ {N, I, O, B, IB, IO, OB} (note: 1 or 2 letters)
#   BearingSize ∈ {4, 5, 6, 7, 8} (representing 6204..6208)
#   Load ∈ {00, 02, 04} representing 0W, 200W, 400W
# Examples: N502 = Normal, 6205, 200W ;  IB704 = Inner+Ball, 6207, 400W
# ─────────────────────────────────────────────────────────────────────────
HUST_FAULT_LETTERS = {
    "N":  "healthy",
    "I":  "inner",
    "O":  "outer",
    "B":  "ball",
    "IB": "inner",   # compound — map to dominant; we keep 4-class
    "IO": "inner",
    "OB": "outer",
}
HUST_BEARING_MAP = {"4": "6204", "5": "6205", "6": "6206", "7": "6207", "8": "6208"}


def parse_hust_filename(stem):
    """Parse e.g. 'IB502' → (fault='inner', bearing='6205', load_w=200)."""
    # Try 2-letter prefix first
    if stem[:2] in HUST_FAULT_LETTERS:
        fault_code = stem[:2]
        rest = stem[2:]
    else:
        fault_code = stem[:1]
        rest = stem[1:]
    if len(rest) != 3:
        raise ValueError(f"Cannot parse HUST filename stem: {stem!r}")
    bearing_digit = rest[0]
    load_code = rest[1:]
    return dict(
        fault   = HUST_FAULT_LETTERS[fault_code],
        bearing = HUST_BEARING_MAP[bearing_digit],
        load_w  = int(load_code) * 100,   # 00→0, 02→200, 04→400
        raw_fault_code = fault_code,
    )


# ─────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────
WINDOW_SAMPLES = 4096    # window length in samples (raw time domain)
WINDOW_OVERLAP = 0.5     # 50% overlap when extracting windows from a signal
ORDER_BINS     = 1024    # length of the order-domain representation
MAX_ORDER      = 20.0    # we cover up to 20 × shaft rotation order


# ─────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────
BATCH_SIZE     = 64
EPOCHS         = 60
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
LAMBDA_PHYS    = 0.3      # weight of physics-consistency loss
LAMBDA_ADV     = 0.2      # weight of adversarial domain loss
DEVICE         = "cuda"   # "cuda" or "cpu"
SEED           = 42
NUM_WORKERS    = 2


# ─────────────────────────────────────────────────────────────────────────
# Conformal
# ─────────────────────────────────────────────────────────────────────────
CONFORMAL_ALPHA = 0.10    # target coverage 1-alpha = 90%
