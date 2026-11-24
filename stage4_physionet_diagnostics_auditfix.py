"""
stage4_physionet_diagnostics_auditfix.py   (EXPLORATORY E1-E5; nothing here is pre-registered)

Audit-fixed version of Argon's stage4_physionet_diagnostics.py. Fixes:
  F1  Argon's loader exec'd every top-level assignment of the v2 script, including `rec0 = records_data[0]`,
      which crashes with IndexError before any experiment runs. Now only a whitelist of v2 FUNCTIONS and four
      literal constants is loaded, and the v2 file must exist and match the SHA-256 of the script that produced
      the reported run (no silent fallback to different built-in code).
  F2  Data pipeline now matches v2 exactly (resample_poly padtype="line"; Argon's copy omitted it).
  F3  The silent gamma search in SC1 is removed. gamma is fixed at the v2 value (0.125) and SC1 fails loudly.
  F4  Self-check targets are parsed from stage3_physionet_stdout.txt (the reported run) instead of typed in,
      and an extra self-check confirms the instrumented DCD is bit-identical to the v2 DCD.
  F5  E2 compared ridge integer DCD (gamma=0.125) with UN-regularised float RLS, which confounds two changes.
      E2 now has three columns: integer DCD, float64 RLS with the SAME persistent ridge (same cost function),
      and float64 RLS without ridge.
  F6  LORO in E3/E5 used the median over training folds; v2 uses the mean (v2 loro_pick). Now uses v2 loro_pick.
  F7  Cache key now includes this script's SHA-256 so results from an older version are never reused.
  F9  E2 'no ridge' column uses gamma=1e-6 solved directly: the inversion-lemma RLS (gamma=0) blew up numerically
      (-90 dB) on the auditor's mock records even at P=9 and P=11, which would measure numerical instability, not ridge.
  F8  E4 also prints NONE and, per operating point, clean RRMSE minus NONE with a bootstrap CI and the P2 margin
      verdict, so every number used in the paper text is printed by a script.

Usage (from the folder that holds the v2 script and its stdout):
  python3 stage4_physionet_diagnostics_auditfix.py motion_artifact_data --exp ALL
  python3 stage4_physionet_diagnostics_auditfix.py motion_artifact_data --exp E2 --quick   (eeg_1..4, not for reporting)
Results are cached in stage4_cache_{full,quick}.json, so an interrupted run resumes where it stopped.
"""
import ast
import copy
import datetime
import difflib
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import scipy
from scipy.signal import resample_poly, lfilter, butter, sosfiltfilt, freqz  # noqa: F401  (used by v2 functions)
from scipy.stats import wilcoxon

try:
    import wfdb
except ImportError as exc:
    raise RuntimeError("The 'wfdb' package is required: pip3 install wfdb") from exc

V2_SHA_EXPECTED = "c32a8afcbb71673f8052cacfaae6a44901b98c1f7db9b95abd964e29c913d95a"
GAMMA = 0.125
GAMMA_NEAR0 = 1e-6  # E2 "no ridge" comparator, solved directly (the inversion-lemma RLS diverges on this data)
L_AX = 16

# ------------------------------------------------------------------------------ CLI
if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 stage4_physionet_diagnostics_auditfix.py <wfdb_dir> [--exp E1|E2|E3|E4|E5|ALL] [--quick] "
                     "[--v2-script PATH] [--v2-stdout PATH] [--no-cache]")
wfdb_src = sys.argv[1]
if not os.path.isdir(wfdb_src):
    raise SystemExit(f"WFDB directory not found: {wfdb_src}")
args = sys.argv[2:]


def arg_value(flag, default):
    if flag in args:
        i = args.index(flag)
        if i + 1 >= len(args):
            raise SystemExit(f"Missing value after {flag}")
        return args[i + 1]
    return default


quick_mode = "--quick" in args
use_cache = "--no-cache" not in args
exp_target = arg_value("--exp", "ALL").upper()
if exp_target not in ("E1", "E2", "E3", "E4", "E5", "ALL"):
    raise SystemExit(f"Invalid --exp {exp_target}")
v2_script_path = arg_value("--v2-script", "stage3_physionet_benchmark_auditfix_v2.py")
v2_stdout_path = arg_value("--v2-stdout", "stage3_physionet_stdout.txt")
skip_sha = "--allow-other-v2" in args  # used only for the auditor's mock-data test; never for reported runs


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


start_wall = time.time()
SELF_SHA = sha256_file(os.path.abspath(__file__))
print("=== STAGE 4 PHYSIONET DIAGNOSTICS (AUDIT-FIXED; ALL RESULTS EXPLORATORY) ===")
if quick_mode:
    print("QUICK MODE (eeg_1..eeg_4 ONLY) - NOT FOR REPORTING")
print(f"Start Time (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
print(f"Python: {platform.python_version()} | NumPy: {np.__version__} | SciPy: {scipy.__version__}")
print(f"Invocation: --exp {exp_target} | this script SHA-256: {SELF_SHA}")

# ------------------------------------------------------------------------------ F1: load v2 functions by whitelist
for p in (v2_script_path, v2_stdout_path):
    if not os.path.isfile(p):
        raise SystemExit(f"Required file not found in this folder: {p}")
v2_sha = sha256_file(v2_script_path)
print(f"v2 script: {v2_script_path} SHA-256 {v2_sha}")
print(f"v2 stdout: {v2_stdout_path} SHA-256 {sha256_file(v2_stdout_path)}")
if v2_sha != V2_SHA_EXPECTED and not skip_sha:
    raise SystemExit(f"v2 script SHA-256 does not match the script that produced the reported run ({V2_SHA_EXPECTED}). Stop.")

V2_FUNCS = ("hp_pole_shift", "apply_dc_blocker_hz", "parse_f4_intervals", "leaky_integrate_axis", "build_multichan_taps",
            "run_multichan_dcd_ridge_int", "run_multichan_nlms", "sweeney_metrics", "safe_metrics", "loro_pick",
            "boot_median_ci")
V2_CONSTS = ("HEADROOM", "CAL_START", "REQUIRED_CHANNELS", "frozen_gamma_p5")
with open(v2_script_path, "r", encoding="utf-8") as f:
    v2_src = f.read()
v2_tree = ast.parse(v2_src)
keep_nodes, v2_dcd_node, consts = [], None, {}
for node in v2_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in V2_FUNCS:
        keep_nodes.append(node)
        if node.name == "run_multichan_dcd_ridge_int":
            v2_dcd_node = node
    elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in V2_CONSTS:
        consts[node.targets[0].id] = ast.literal_eval(node.value)
missing = set(V2_FUNCS) - {n.name for n in keep_nodes}
if missing or set(consts) != set(V2_CONSTS):
    raise SystemExit(f"Could not find in v2: functions {sorted(missing)}, constants {sorted(set(V2_CONSTS) - set(consts))}")
globals().update(consts)
exec(compile(ast.Module(body=keep_nodes, type_ignores=[]), v2_script_path, "exec"), globals())
if float(frozen_gamma_p5) != GAMMA:  # noqa: F821  (loaded from v2)
    raise SystemExit(f"v2 frozen_gamma_p5={frozen_gamma_p5} != {GAMMA}")  # noqa: F821
print(f"Loaded v2 functions {', '.join(V2_FUNCS)}; HEADROOM={HEADROOM}, CAL_START={CAL_START}, gamma={GAMMA}")  # noqa: F821

# ------------------------------------------------------------------------------ instrumented DCD (adds update counter only)
mod_node = copy.deepcopy(v2_dcd_node)
mod_node.name = "run_dcd_counted"
new_body = []
for stmt in mod_node.body:
    if isinstance(stmt, ast.For) and "u_cnt" in ast.unparse(stmt):
        new_body.append(ast.parse("_u_cnt_arr = np.zeros(len(d_sig), dtype=np.int64)").body[0])
        stmt.body.append(ast.parse("_u_cnt_arr[k] = u_cnt").body[0])
        new_body.append(stmt)
    elif isinstance(stmt, ast.Return):
        stmt.value.elts.append(ast.Name(id="_u_cnt_arr", ctx=ast.Load()))
        new_body.append(stmt)
    else:
        new_body.append(stmt)
mod_node.body = new_body
ast.fix_missing_locations(mod_node)
exec(compile(ast.Module(body=[mod_node], type_ignores=[]), "<run_dcd_counted>", "exec"), globals())
print("\n=== DIFF v2 run_multichan_dcd_ridge_int -> run_dcd_counted ===")
for line in difflib.unified_diff(ast.unparse(v2_dcd_node).splitlines(), ast.unparse(mod_node).splitlines(),
                                 "v2", "stage4", lineterm=""):
    print("  " + line)


# ------------------------------------------------------------------------------ F5: float64 RLS (same scaling as v2 DCD)
def run_float_rls(d_sig, ref_mat, L_ax=16, P_shift=11, gamma=GAMMA, cal_len=2048, return_internals=False):
    """EXPLORATORY E2. Exponentially weighted RLS in float64, lambda = 1 - 2^-P, the same causal calibration scaling
    as the v2 integer DCD (99.9th pct of |x[CAL_START:cal_len]| x 1.05 x HEADROOM), the same a-priori output.
    gamma > 0: solves (R_k + delta I) w_k = beta_k every sample with R_0 = 0 and delta = gamma*var_u*2^P,
               i.e. exactly the normal equations the ridge integer DCD iterates towards.
    gamma = 0: standard RLS (matrix inversion lemma) with initial inverse I/var_u."""
    lam = 1.0 - 2.0 ** (-P_shift)
    ref = np.asarray(ref_mat, dtype=np.float64)
    ref = ref[:, None] if ref.ndim == 1 else ref
    sc_d = max(float(np.percentile(np.abs(d_sig[CAL_START:cal_len]), 99.9)) * 1.05 * HEADROOM, 1e-9)  # noqa: F821
    d_in = np.asarray(d_sig, dtype=np.float64) / sc_d
    cols = []
    for m in range(ref.shape[1]):
        sc_m = max(float(np.percentile(np.abs(ref[CAL_START:cal_len, m]), 99.9)) * 1.05 * HEADROOM, 1e-9)  # noqa: F821
        cols.append(ref[:, m] / sc_m)
    r_in = np.column_stack(cols)
    U_all, _ = build_multichan_taps(r_in, L_ax)  # noqa: F821
    n_s, L_tot = U_all.shape
    var_u = max(float(np.var(r_in[CAL_START:cal_len])), 1e-12)  # noqa: F821
    w = np.zeros(L_tot)
    out = np.zeros(n_s)
    if gamma > 0.0:
        delta = float(gamma) * var_u * 2.0 ** P_shift
        R = np.zeros((L_tot, L_tot))
        beta = np.zeros(L_tot)
        dI = delta * np.eye(L_tot)
        for k in range(n_s):
            u = U_all[k]
            out[k] = (d_in[k] - w @ u) * sc_d
            R *= lam
            R += np.outer(u, u)
            beta *= lam
            beta += d_in[k] * u
            w = np.linalg.solve(R + dI, beta)
    else:
        Pinv = np.eye(L_tot) / var_u
        for k in range(n_s):
            u = U_all[k]
            e0 = d_in[k] - w @ u
            out[k] = e0 * sc_d
            Pu = Pinv @ u
            g = Pu / (lam + u @ Pu)
            w = w + g * e0
            Pinv = (Pinv - np.outer(g, Pu)) / lam
    if return_internals:
        return out, w, U_all, d_in, var_u
    return out


# ------------------------------------------------------------------------------ SC3/SC4: float RLS vs direct solve
print("\n=== SELF-CHECKS ===", flush=True)
rng = np.random.default_rng(2026)
syn_ref = rng.standard_normal((4096 + CAL_START, 3))  # noqa: F821
syn_d = rng.standard_normal(4096 + CAL_START)  # noqa: F821
for P_t in (11, 13):
    for g_t in (0.0, GAMMA_NEAR0, GAMMA):
        _, w_rec, U_s, d_s, var_s = run_float_rls(syn_d, syn_ref, 16, P_t, g_t, cal_len=CAL_START + 2048, return_internals=True)  # noqa: F821
        n_s = len(d_s)
        lam_t = 1.0 - 2.0 ** (-P_t)
        wexp = lam_t ** np.arange(n_s - 1, -1, -1)
        R_dir = (U_s.T * wexp) @ U_s
        R_dir += (g_t * var_s * 2.0 ** P_t) * np.eye(48) if g_t > 0 else (lam_t ** n_s) * var_s * np.eye(48)
        w_dir = np.linalg.solve(R_dir, (U_s.T * wexp) @ d_s)
        err = float(np.linalg.norm(w_rec - w_dir) / np.linalg.norm(w_dir))
        ok = err < 1e-6
        print(f"  SC3/SC4 float RLS vs direct solve, P={P_t}, gamma={g_t:g}: rel weight error {err:.2e} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            sys.exit(1)

# ------------------------------------------------------------------------------ parse v2 stdout for SC targets (F4)
v2_rows, v2_mu_a = {}, None
with open(v2_stdout_path, "r", encoding="utf-8", errors="replace") as f:
    v2_lines = f.readlines()
in_table = False
for ln_no, line in enumerate(v2_lines, start=1):
    s = line.strip()
    if s.startswith("LORO mu, NLMS on RefA_ACC1_3ax:"):
        v2_mu_a = [float(x) for x in s.split(":", 1)[1].split(",")]
    if s.startswith("=== PER-RECORD RESULTS"):
        in_table = True
        continue
    if in_table and s.startswith("eeg_"):
        parts = [p.strip() for p in s.split("|")]
        rr = parts[11].split()[0].split("/")
        v2_rows[parts[0]] = {"line": ln_no, "P": int(parts[1]), "DCD_RefA_P7": float(parts[8]),
                             "DCD_RefA_P11": float(parts[10]), "NLMS_RefA": float(parts[5]), "NONE_rrmse": float(rr[0])}
    elif in_table and s.startswith("==="):
        in_table = False
if v2_mu_a is None or not v2_rows:
    raise SystemExit("Could not parse the per-record table / LORO mu line from the v2 stdout.")

# ------------------------------------------------------------------------------ data loading, identical to v2 (F2)
record_ids = [f"eeg_{i}" for i in (range(1, 5) if quick_mode else range(1, 24))]
print(f"\n=== LOADING {len(record_ids)} RECORDS FROM {wfdb_src} ===", flush=True)
records = []
for rname in record_ids:
    base = os.path.join(wfdb_src, rname)
    rec, ann = wfdb.rdrecord(base), wfdb.rdann(base, "trigger")
    if float(rec.fs) != 2048.0:
        raise SystemExit(f"{rname}: fs={rec.fs}")
    sig = {nm: i for i, nm in enumerate(rec.sig_name)}
    for ch in REQUIRED_CHANNELS:  # noqa: F821
        if ch not in sig:
            raise SystemExit(f"{rname}: missing channel {ch}")
    raw = rec.p_signal
    _, ivals, nF, _ = parse_f4_intervals(ann, raw.shape[0], 2048.0)  # noqa: F821
    if nF != 4 or abs(sum(iv[2] for iv in ivals) - 240.0) > 1.0:
        raise SystemExit(f"{rname}: F4 interval check failed")
    eeg1 = resample_poly(raw[:, sig["EEG1"]], 1, 4, padtype="line")
    eeg2 = resample_poly(raw[:, sig["EEG2"]], 1, 4, padtype="line")
    n512 = len(eeg1)
    mask = np.zeros(n512, dtype=bool)
    first_f = n512
    for (f_s, r_s, _) in ivals:
        a, b = max(0, min(int(round(f_s / 4.0)), n512)), max(0, min(int(round(r_s / 4.0)), n512))
        mask[a:b] = True
        first_f = min(first_f, a)
    acc1, acc2 = [], []
    for ax in ("x", "y", "z"):
        acc1.append(apply_dc_blocker_hz(resample_poly(raw[:, sig[f"ACC_{ax}1"]], 1, 4, padtype="line"), 512.0, 0.64)[0])  # noqa: F821
        acc2.append(apply_dc_blocker_hz(resample_poly(raw[:, sig[f"ACC_{ax}2"]], 1, 4, padtype="line"), 512.0, 0.64)[0])  # noqa: F821
    ref_a = np.column_stack(acc1)
    ref_b = np.column_stack([acc1[j] - acc2[j] for j in range(3)])
    vp = [leaky_integrate_axis(ref_b[:, j], P_int=6) for j in range(3)]  # noqa: F821
    ref_c = np.column_stack([ref_b, np.column_stack([v for v, _ in vp]), np.column_stack([p for _, p in vp])])
    records.append({"name": rname, "c_hp": apply_dc_blocker_hz(eeg1, 512.0, 0.64)[0],  # noqa: F821
                    "n_hp": apply_dc_blocker_hz(eeg2, 512.0, 0.64)[0],  # noqa: F821
                    "mask_art": mask, "mask_clean": ~mask, "cal": max(first_f, 512),
                    "refs": {"RefA": ref_a, "RefC": ref_c}})
    print(f"  {rname} loaded", flush=True)
n_rec = len(records)


def metrics(rd, est):
    c, n, ma, mc = rd["c_hp"], rd["n_hp"], rd["mask_art"], rd["mask_clean"]
    blk = safe_metrics(c, n, est, ma, mc)  # noqa: F821
    return {"dsnr_blocks": float(blk[0]), "corr_imp": float(blk[1]), "clean_rrmse": float(blk[3]),
            "dsnr_clean": float(safe_metrics(c, n, est, mc, mc)[0])}  # noqa: F821


# SC0: data pipeline reproduces v2 NONE clean RRMSE for every loaded record
sc0_ok = True
for rd in records:
    rd["none"] = metrics(rd, rd["n_hp"])
    tgt = v2_rows[rd["name"]]["NONE_rrmse"]
    if abs(rd["none"]["clean_rrmse"] - tgt) > 0.0006:
        sc0_ok = False
    print(f"  SC0 {rd['name']}: NONE clean RRMSE {rd['none']['clean_rrmse']:.4f} vs v2 {tgt:.3f} "
          f"(v2 stdout line {v2_rows[rd['name']]['line']})", flush=True)
print(f"  SC0 data pipeline -> {'PASS' if sc0_ok else 'FAIL'}", flush=True)
if not sc0_ok:
    sys.exit(1)

# SC5: instrumented DCD output bit-identical to the v2 DCD (first 8192 samples of eeg_1)
r0 = records[0]
seg = slice(0, 8192)
o_v2, _ = run_multichan_dcd_ridge_int(r0["n_hp"][seg], r0["refs"]["RefA"][seg], 16, 11, gamma=GAMMA, cal_len=4096)  # noqa: F821
o_c, _, _ = run_dcd_counted(r0["n_hp"][seg], r0["refs"]["RefA"][seg], 16, 11, gamma=GAMMA, cal_len=4096)  # noqa: F821
sc5 = bool(np.array_equal(o_v2, o_c))
print(f"  SC5 instrumented DCD bit-identical to v2 DCD -> {'PASS' if sc5 else 'FAIL'}", flush=True)
if not sc5:
    sys.exit(1)

# ------------------------------------------------------------------------------ cache (F7)
cache_file = "stage4_cache_quick.json" if quick_mode else "stage4_cache_full.json"
cache_tag = f"{SELF_SHA[:16]}|{v2_sha[:16]}"
cache = {}
if use_cache and os.path.isfile(cache_file):
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if blob.get("tag") == cache_tag:
            cache = blob.get("entries", {})
            print(f"  Loaded {len(cache)} cached runs from {cache_file}", flush=True)
        else:
            print(f"  Ignoring {cache_file}: produced by a different script version", flush=True)
    except Exception:
        cache = {}


def cached(key, fn):
    if use_cache and key in cache:
        return cache[key]
    res = fn()
    cache[key] = res
    if use_cache:
        tmp = cache_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"tag": cache_tag, "entries": cache}, f)
        os.replace(tmp, cache_file)
    return res


def ev_dcd(rd, ref, P, Nu=4):
    def run():
        out, _, cnt = run_dcd_counted(rd["n_hp"], rd["refs"][ref], L_AX, P, gamma=GAMMA, Nu=Nu, cal_len=rd["cal"])  # noqa: F821
        m = metrics(rd, out)
        m["mean_upd"], m["p99_upd"] = float(np.mean(cnt)), float(np.percentile(cnt, 99))
        return m
    return cached(f"DCD|{rd['name']}|{ref}|P{P}|Nu{Nu}", run)


def ev_nlms(rd, ref, mu):
    return cached(f"NLMS|{rd['name']}|{ref}|mu{mu:g}",
                  lambda: metrics(rd, run_multichan_nlms(rd["n_hp"], rd["refs"][ref], L_AX, mu=mu, cal_len=rd["cal"])))  # noqa: F821


def ev_float(rd, ref, P, g):
    return cached(f"FLT|{rd['name']}|{ref}|P{P}|g{g:g}",
                  lambda: metrics(rd, run_float_rls(rd["n_hp"], rd["refs"][ref], L_AX, P, g, cal_len=rd["cal"])))


# SC1/SC2: reproduce the reported v2 per-record values (targets parsed from the v2 stdout, printed to 2 dp)
sc_ok = True
for i in range(3):
    rd = records[i]
    row = v2_rows[rd["name"]]
    checks = [("SC1 DCD_RefA P=11", ev_dcd(rd, "RefA", 11)["dsnr_blocks"], row["DCD_RefA_P11"]),
              ("SC2 NLMS_RefA mu=%g" % v2_mu_a[int(rd["name"].split("_")[1]) - 1],
               ev_nlms(rd, "RefA", v2_mu_a[int(rd["name"].split("_")[1]) - 1])["dsnr_blocks"], row["NLMS_RefA"])]
    if i == 0:
        checks.append(("SC1b DCD_RefA P=7", ev_dcd(rd, "RefA", 7)["dsnr_blocks"], row["DCD_RefA_P7"]))
    for lab, got, tgt in checks:
        ok = abs(got - tgt) <= 0.0051
        sc_ok &= ok
        print(f"  {lab} {rd['name']}: {got:+.4f} vs v2 {tgt:+.2f} (line {row['line']}) -> {'PASS' if ok else 'FAIL'}", flush=True)
if not sc_ok:
    print("  SELF-CHECK FAILED: this script does not reproduce the reported v2 run. Stop.", flush=True)
    sys.exit(1)


def med(v):
    return float(np.median(np.asarray(v, dtype=float)))


def paired(label, a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, lo, hi = boot_median_ci(d)  # noqa: F821
    p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else float("nan")
    print(f"  {label}: median {m:+.3f} [{lo:+.3f}, {hi:+.3f}], first>second on {int(np.sum(d > 0))}/{len(d)}, "
          f"Wilcoxon p={p:.3g} (unadjusted)", flush=True)


def loro(grid, table):
    """table[g] = list of per-record dicts. Uses the v2 loro_pick (mean of training-fold scores)."""
    scores = {g: [r["dsnr_blocks"] + r["dsnr_clean"] for r in table[g]] for g in grid}
    picks = loro_pick(grid, scores, n_rec)  # noqa: F821
    held = [table[picks[i]][i] for i in range(n_rec)]
    return picks, held


def picks_str(picks):
    u, c = np.unique(picks, return_counts=True)
    return ", ".join(f"{x:g} ({k}/{n_rec})" for x, k in zip(u, c))


# ------------------------------------------------------------------------------ E1
if exp_target in ("E1", "ALL"):
    print("\n=== EXPLORATORY E1: UPDATE LIMIT (DCD_RefA, Nu in {4,8,16} x P in {7,9,11}) ===", flush=True)
    for P in (7, 9, 11):
        base = None
        for Nu in (4, 8, 16):
            rows = [ev_dcd(rd, "RefA", P, Nu) for rd in records]
            print(f"\n  --- P={P}, Nu={Nu} ---\n  Record | mean upd | p99 upd | dSNR blocks | dSNR clean | clean RRMSE", flush=True)
            for rd, r in zip(records, rows):
                print(f"  {rd['name']:6s} | {r['mean_upd']:8.3f} | {r['p99_upd']:7.1f} | {r['dsnr_blocks']:+11.2f} | "
                      f"{r['dsnr_clean']:+10.2f} | {r['clean_rrmse']:.4f}", flush=True)
            print(f"  MEDIAN P={P} Nu={Nu}: mean upd {med([r['mean_upd'] for r in rows]):.3f} | "
                  f"blocks {med([r['dsnr_blocks'] for r in rows]):+.2f} dB | clean {med([r['dsnr_clean'] for r in rows]):+.2f} dB | "
                  f"clean RRMSE {med([r['clean_rrmse'] for r in rows]):.4f}", flush=True)
            if base is None:
                base = rows
            else:
                paired(f"P={P}: dSNR blocks Nu={Nu} - Nu=4", [r["dsnr_blocks"] for r in rows], [r["dsnr_blocks"] for r in base])
                paired(f"P={P}: dSNR clean  Nu={Nu} - Nu=4", [r["dsnr_clean"] for r in rows], [r["dsnr_clean"] for r in base])

# ------------------------------------------------------------------------------ E2
if exp_target in ("E2", "ALL"):
    print("\n=== EXPLORATORY E2: INTEGER DCD vs FLOAT64 RLS (RefA, Nu=4) ===", flush=True)
    print("  INT = integer ridge DCD (v2); FLT_R = float64 RLS, same ridge gamma=0.125 (same normal equations);"
          " FLT_0 = float64 RLS, ridge 1e-6 (effectively none), solved directly every sample", flush=True)
    for P in (5, 7, 9, 11):
        I_ = [ev_dcd(rd, "RefA", P) for rd in records]
        R_ = [ev_float(rd, "RefA", P, GAMMA) for rd in records]
        Z_ = [ev_float(rd, "RefA", P, GAMMA_NEAR0) for rd in records]
        print(f"\n  --- P={P} ---\n  Record | INT blocks/clean | FLT_R blocks/clean | FLT_0 blocks/clean | RRMSE INT/FLT_R/FLT_0", flush=True)
        for rd, a, b, c in zip(records, I_, R_, Z_):
            print(f"  {rd['name']:6s} | {a['dsnr_blocks']:+6.2f}/{a['dsnr_clean']:+6.2f} | {b['dsnr_blocks']:+6.2f}/{b['dsnr_clean']:+6.2f} | "
                  f"{c['dsnr_blocks']:+6.2f}/{c['dsnr_clean']:+6.2f} | {a['clean_rrmse']:.4f}/{b['clean_rrmse']:.4f}/{c['clean_rrmse']:.4f}", flush=True)
        for lab, X in (("INT", I_), ("FLT_R", R_), ("FLT_0", Z_)):
            print(f"  MEDIAN P={P} {lab:5s}: blocks {med([x['dsnr_blocks'] for x in X]):+.2f} dB | clean {med([x['dsnr_clean'] for x in X]):+.2f} dB"
                  f" | clean RRMSE {med([x['clean_rrmse'] for x in X]):.4f}", flush=True)
        paired(f"P={P}: dSNR blocks INT - FLT_R (cost of integer/DCD)", [x["dsnr_blocks"] for x in I_], [x["dsnr_blocks"] for x in R_])
        paired(f"P={P}: dSNR blocks FLT_R - FLT_0 (effect of ridge)", [x["dsnr_blocks"] for x in R_], [x["dsnr_blocks"] for x in Z_])
        paired(f"P={P}: dSNR clean  FLT_R - FLT_0 (effect of ridge)", [x["dsnr_clean"] for x in R_], [x["dsnr_clean"] for x in Z_])

# ------------------------------------------------------------------------------ E3
if exp_target in ("E3", "ALL"):
    print("\n=== EXPLORATORY E3: DCD_RefA P GRID EXTENDED TO {5,7,9,11,13} (LORO, v2 rule) ===", flush=True)
    grid = (5, 7, 9, 11, 13)
    tab = {P: [ev_dcd(rd, "RefA", P) for rd in records] for P in grid}
    for P in grid:
        print(f"  P={P:2d}: blocks {med([r['dsnr_blocks'] for r in tab[P]]):+.2f} dB | clean {med([r['dsnr_clean'] for r in tab[P]]):+.2f} dB"
              f" | mean LORO score {np.mean([r['dsnr_blocks'] + r['dsnr_clean'] for r in tab[P]]):+.3f} | "
              f"clean RRMSE {med([r['clean_rrmse'] for r in tab[P]]):.4f}", flush=True)
    picks, held = loro(grid, tab)
    print(f"  LORO choice over {{5,7,9,11,13}}: {picks_str(picks)}", flush=True)
    print(f"  Held-out medians: blocks {med([h['dsnr_blocks'] for h in held]):+.2f} dB | clean {med([h['dsnr_clean'] for h in held]):+.2f} dB"
          f" | clean RRMSE {med([h['clean_rrmse'] for h in held]):.4f}", flush=True)

# ------------------------------------------------------------------------------ E4 / E5
MU_GRID = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
P_GRID = (5, 7, 9, 11, 13)
if exp_target in ("E4", "ALL"):
    print("\n=== EXPLORATORY E4: TRADE-OFF CURVES (medians over records) ===", flush=True)
    none_rr = [rd["none"]["clean_rrmse"] for rd in records]
    csv_rows = []
    for ref in ("RefA", "RefC"):
        pts = []
        for mu in MU_GRID:
            pts.append((f"NLMS_{ref}", f"mu={mu:g}", [ev_nlms(rd, ref, mu) for rd in records]))
        for P in P_GRID:
            pts.append((f"DCD_{ref}", f"P={P}", [ev_dcd(rd, ref, P) for rd in records]))
        meds = [(m, p, med([r["dsnr_blocks"] for r in rs]), med([r["dsnr_clean"] for r in rs]), rs) for m, p, rs in pts]
        print(f"\n  --- {ref} ---  (NONE: blocks +0.00, clean +0.00, clean RRMSE {med(none_rr):.4f})")
        print("  Method     | Param    | blocks dB | clean dB | dRRMSE vs NONE [95% CI]  | P2 margin +0.02 | Pareto within | Pareto joint")
        for m, p, b, c, rs in meds:
            def dominated(pool):
                return any((q[2] >= b and q[3] >= c) and (q[2] > b or q[3] > c) for q in pool)
            dm, lo, hi = boot_median_ci(np.array([r["clean_rrmse"] for r in rs]) - np.array(none_rr))  # noqa: F821
            print(f"  {m:10s} | {p:8s} | {b:+9.2f} | {c:+8.2f} | {dm:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                  f"{'pass' if hi < 0.02 else 'fail':15s} | {'no' if dominated([q for q in meds if q[0] == m]) else 'YES':13s} | "
                  f"{'no' if dominated(meds) else 'YES'}", flush=True)
            for rd, r in zip(records, rs):
                csv_rows.append(f"{rd['name']},{ref},{m},{p},{r['dsnr_blocks']:.6f},{r['dsnr_clean']:.6f},{r['clean_rrmse']:.6f},"
                                f"{rd['none']['clean_rrmse']:.6f},{r['corr_imp']:.6f}")
    print("\n=== CSV E4 BEGIN ===\nrecord,reference,method,param,dsnr_blocks,dsnr_clean,clean_rrmse,none_clean_rrmse,corr_imp")
    print("\n".join(csv_rows))
    print("=== CSV E4 END ===", flush=True)

if exp_target in ("E5", "ALL"):
    print("\n=== EXPLORATORY E5: P CHOSEN BY LORO ON RefC ITSELF (vs NLMS_RefC with LORO mu) ===", flush=True)
    nl = {mu: [ev_nlms(rd, "RefC", mu) for rd in records] for mu in MU_GRID}
    mu_picks, nl_held = loro(MU_GRID, nl)
    for lab, grid in (("v2 grid {5,7,9,11}", (5, 7, 9, 11)), ("extended {5,7,9,11,13}", P_GRID)):
        tab = {P: [ev_dcd(rd, "RefC", P) for rd in records] for P in grid}
        picks, held = loro(grid, tab)
        print(f"  [{lab}] P picks: {picks_str(picks)} | NLMS_RefC mu picks: {picks_str(mu_picks)}", flush=True)
        print(f"  [{lab}] DCD_RefC held-out: blocks {med([h['dsnr_blocks'] for h in held]):+.2f} dB, clean "
              f"{med([h['dsnr_clean'] for h in held]):+.2f} dB | NLMS_RefC: blocks {med([h['dsnr_blocks'] for h in nl_held]):+.2f} dB, "
              f"clean {med([h['dsnr_clean'] for h in nl_held]):+.2f} dB", flush=True)
        paired(f"[{lab}] dSNR blocks DCD_RefC - NLMS_RefC", [h["dsnr_blocks"] for h in held], [h["dsnr_blocks"] for h in nl_held])
        paired(f"[{lab}] dSNR clean  DCD_RefC - NLMS_RefC", [h["dsnr_clean"] for h in held], [h["dsnr_clean"] for h in nl_held])

print(f"\nEnd Time (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()} | elapsed {time.time() - start_wall:.0f} s", flush=True)
