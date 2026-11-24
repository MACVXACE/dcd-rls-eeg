"""
stage2_round6_ridge_benchmark.py
Pre-registered Round 6 benchmark evaluating plain Int DCD-RLS (M1), ridge Int DCD-RLS (M2),
float64 ridge EWLS (M3), float64 Givens QR-RLS (M4), tuned NLMS (M5), and standard float64 RLS (M6)
across 18 conditions: SNR {-7, -1, +2} dB x sensor noise {1%, 10%, 31.6%} x P {4, 5}, L = 16.

Usage:
  python3 stage2_round6_ridge_benchmark.py <stage2_round4_benchmark.py> <EEGdenoiseNet_dir> [--quick]
"""
import sys
import os
import time
import hashlib
import datetime
import platform
import numpy as np
import scipy
from scipy.signal import lfilter, butter, resample_poly
from scipy.stats import wilcoxon

# ==============================================================================
# 0. ARGUMENT PARSING, ENVIRONMENT HEADER & SHA-256 VERIFICATION
# ==============================================================================
if len(sys.argv) < 3:
    raise RuntimeError(
        "Usage: python3 stage2_round6_ridge_benchmark.py <stage2_round4_benchmark.py> <EEGdenoiseNet_dir> [--quick]"
    )

r4_script_path = os.path.abspath(sys.argv[1])
eeg_dir_path = os.path.abspath(sys.argv[2])
quick_mode = ("--quick" in sys.argv[3:])

if not os.path.isfile(r4_script_path):
    raise FileNotFoundError(f"Round 4 script not found: {r4_script_path}")

eeg_npy_path = os.path.join(eeg_dir_path, "EEG_all_epochs.npy")
emg_npy_path = os.path.join(eeg_dir_path, "EMG_all_epochs.npy")
if not os.path.isfile(eeg_npy_path):
    raise FileNotFoundError(f"Missing EEG file: {eeg_npy_path}")
if not os.path.isfile(emg_npy_path):
    raise FileNotFoundError(f"Missing EMG file: {emg_npy_path}")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


start_wall = time.time()
start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

print("=== ROUND 6 BENCHMARK HEADER ===")
if quick_mode:
    print("QUICK MODE - NOT FOR REPORTING")
print(f"Start Time (UTC): {start_iso}")
print(f"Python Version:   {platform.python_version()}")
print(f"NumPy Version:    {np.__version__}")
print(f"SciPy Version:    {scipy.__version__}")
print(f"SHA-256 stage2_round4_benchmark.py: {sha256_file(r4_script_path)}")
print(f"SHA-256 EEG_all_epochs.npy:         {sha256_file(eeg_npy_path)}")
print(f"SHA-256 EMG_all_epochs.npy:         {sha256_file(emg_npy_path)}")
print(
    "Fixed Seeds: val_pool=7, held_eeg=99, held_emg=101, val_emg=55, "
    "val_sensor_noise=313, held_sensor_noise=777, bootstrap=42"
)

if quick_mode:
    est_sec = 35.0
else:
    # Estimated from per-epoch measured timings: M1/M2 ~0.06s, M3 ~0.02s, M4 ~0.21s, M5 ~0.007s
    est_sec = 980.0
print(f"Estimated Runtime: ~{est_sec:.0f} seconds ({est_sec / 60.0:.1f} minutes)", flush=True)

# ==============================================================================
# 1. LOAD ROUND 4 FUNCTIONS AND SPLITS VERBATIM VIA EXEC (R7)
# ==============================================================================
with open(r4_script_path, "r", encoding="utf-8") as f_r4:
    r4_source = f_r4.read()

if "# BLOCK 1: TRACKING" not in r4_source:
    raise RuntimeError("Marker '# BLOCK 1: TRACKING' not found in stage2_round4_benchmark.py")

r4_prefix = r4_source.split("# BLOCK 1: TRACKING")[0].replace("print_energy_table_round4()\n", "")
saved_argv = list(sys.argv)
sys.argv = [saved_argv[0], eeg_dir_path]
exec(compile(r4_prefix, r4_script_path, "exec"))
sys.argv = saved_argv

# Reconstruct exact index sets to verify T4 disjointness explicitly
val_pool_idx_check = np.random.default_rng(7).choice(len(R), 200, replace=False)
val_eeg_idx = val_pool_idx_check[:50]
used_eeg_set = set(val_pool_idx_check.tolist())
held_eeg_idx = np.array([i for i in np.random.default_rng(99).permutation(len(R)) if i not in used_eeg_set][:100])

held_emg_idx = np.random.default_rng(101).choice(len(E), 100, replace=False)
held_emg_set = set(held_emg_idx.tolist())
val_emg_idx = np.array([i for i in np.random.default_rng(55).permutation(len(E)) if i not in held_emg_set][:50])

X_val = R[val_eeg_idx]
E_val = E[val_emg_idx]

# ==============================================================================
# 2. DEFINITIONS OF M2 (run_dcd_ridge_int) AND M3 (ridge_ewls)
# ==============================================================================
def ridge_ewls(d_sig, ref_sig, L=16, P_shift=5, gamma=0.0, p0=1.0):
    """
    M3: Float64 exponentially weighted least squares with diagonal ridge load
    delta = gamma * var(ref_sig) * 2^P_shift.
    At gamma = 0.0, solves the exact same normal equations as M4 (run_qr_rls_float64).
    """
    lam = 1.0 - 2.0**(-P_shift)
    U_all = _build_tap_vectors(ref_sig, L)
    L_tot = U_all.shape[1]
    var_r = float(np.var(ref_sig))
    R_mat = np.eye(L_tot, dtype=np.float64) * (p0 * var_r + 1e-6)
    r_du = np.zeros(L_tot, dtype=np.float64)
    w = np.zeros(L_tot, dtype=np.float64)
    out = np.zeros(len(d_sig), dtype=np.float64)
    delta = float(gamma) * var_r * (2.0**P_shift)
    I_mat = np.eye(L_tot, dtype=np.float64)

    for k in range(len(d_sig)):
        u = U_all[k]
        out[k] = d_sig[k] - np.dot(w, u)
        R_mat = lam * R_mat + np.outer(u, u)
        r_du = lam * r_du + d_sig[k] * u
        w = np.linalg.solve(R_mat + delta * I_mat, r_du)
    return out


def run_dcd_ridge_int(
    d_sig,
    ref_sig,
    L=16,
    P_shift=5,
    gamma=0.0,
    Nu=4,
    Mb=16,
    H_int=16,
    delta_reg=1.0,
    scale_d=None,
    scale_r=None,
    pow2_load=False,
    return_hw_stats=False,
):
    """
    M2: Integer Q15/Q30 Leading-Element DCD-RLS with diagonal ridge load
    delta = gamma * var(u) * 2^P_shift in Q30.
    When gamma == 0.0, scale_d is None, scale_r is None, and pow2_load is False,
    executes the exact same operations as run_dcd_rls_q30_int (M1).
    """
    if scale_d is None:
        sc_d = max(float(np.max(np.abs(d_sig))), 1e-9) * 1.05
    else:
        sc_d = float(scale_d)
    if scale_r is None:
        sc_r = max(float(np.max(np.abs(ref_sig))), 1e-9) * 1.05
    else:
        sc_r = float(scale_r)

    Q15 = 1 << 15
    Q30 = 1 << 30

    raw_d_scaled = np.round((d_sig / sc_d) * Q15)
    raw_r_scaled = np.round((ref_sig / sc_r) * Q15)
    clipped_samples = int(
        np.sum((raw_d_scaled < -Q15) | (raw_d_scaled >= Q15) | (raw_r_scaled < -Q15) | (raw_r_scaled >= Q15))
    )

    d_q15 = np.clip(raw_d_scaled, -Q15, Q15 - 1).astype(np.int64)
    r_q15 = np.clip(raw_r_scaled, -Q15, Q15 - 1)
    U_all = _build_tap_vectors(r_q15, L).astype(np.int64)
    L_tot = U_all.shape[1]
    single_axis = (np.asarray(ref_sig).ndim == 1)

    var_u_q15 = float(np.var(r_q15 / float(Q15)))
    w_q15 = np.zeros(L_tot, dtype=np.int64)
    reg_q30 = max(int(round(delta_reg * var_u_q15 * Q30)), 1)
    R_q30 = np.eye(L_tot, dtype=np.int64) * reg_q30
    r_q30 = np.zeros(L_tot, dtype=np.int64)
    out = np.zeros(len(d_sig), dtype=np.float64)

    if gamma > 0.0:
        delta_q30 = max(int(round(float(gamma) * var_u_q15 * (1 << P_shift) * Q30)), 1)
        if pow2_load:
            p2 = int(round(np.log2(float(delta_q30))))
            delta_q30 = 1 << max(p2, 0)
    else:
        delta_q30 = 0

    total_updates = 0
    max_updates_sample = 0
    total_bit_iters = 0
    max_abs_R = int(np.max(np.abs(R_q30)))
    max_abs_r = 0
    max_abs_w = 0

    for k in range(len(d_sig)):
        u_q15 = U_all[k]
        y_hat_q15 = int(np.dot(w_q15, u_q15) >> 15)
        e_q15 = int(d_q15[k] - y_hat_q15)
        out[k] = (e_q15 / float(Q15)) * sc_d

        if single_axis:
            R_q30[1:, 1:] = R_q30[:-1, :-1]
            col0 = R_q30[:, 0] - (R_q30[:, 0] >> P_shift) + u_q15[0] * u_q15
            R_q30[:, 0] = col0
            R_q30[0, :] = col0
        else:
            R_q30 = R_q30 - (R_q30 >> P_shift) + np.outer(u_q15, u_q15)

        if delta_q30 == 0:
            r_q30 = r_q30 - (r_q30 >> P_shift) + e_q15 * u_q15
        else:
            leak_vec = ((delta_q30 >> P_shift) * w_q15) >> 15
            r_q30 = r_q30 - (r_q30 >> P_shift) + e_q15 * u_q15 - leak_vec

        m = 1
        u_cnt = 0
        bit_iters = 0
        while m <= Mb and u_cnt < Nu:
            bit_iters += 1
            n = int(np.argmax(np.abs(r_q30)))
            sh_th = m - 3
            diag_val = R_q30[n, n] + delta_q30
            thresh = (diag_val >> sh_th) if sh_th >= 0 else (diag_val << (-sh_th))
            if np.abs(r_q30[n]) > thresh:
                s = 1 if r_q30[n] > 0 else -1
                step_q15 = (H_int * Q15) >> m
                w_q15[n] += s * step_q15
                sh_r = m - 4
                r_q30 -= s * ((R_q30[:, n] >> sh_r) if sh_r >= 0 else (R_q30[:, n] << (-sh_r)))
                if delta_q30 != 0:
                    r_q30[n] -= s * ((delta_q30 >> sh_r) if sh_r >= 0 else (delta_q30 << (-sh_r)))
                u_cnt += 1
            else:
                m += 1

        if return_hw_stats:
            total_updates += u_cnt
            if u_cnt > max_updates_sample:
                max_updates_sample = u_cnt
            total_bit_iters += bit_iters
            cur_R = int(np.max(np.abs(R_q30[:, 0])))
            if cur_R > max_abs_R:
                max_abs_R = cur_R
            cur_r = int(np.max(np.abs(r_q30)))
            if cur_r > max_abs_r:
                max_abs_r = cur_r
            cur_w = int(np.max(np.abs(w_q15)))
            if cur_w > max_abs_w:
                max_abs_w = cur_w

    if return_hw_stats:
        n_samp = len(d_sig)
        hw = {
            "mean_updates": total_updates / float(n_samp),
            "max_updates": max_updates_sample,
            "mean_bit_iters": total_bit_iters / float(n_samp),
            "max_abs_R": max_abs_R,
            "max_abs_r": max_abs_r,
            "max_abs_w": max_abs_w,
            "clipped_samples": clipped_samples,
            "total_samples": n_samp,
        }
        return out, hw
    return out


# ==============================================================================
# 3. MANDATORY SELF-CHECKS T1, T2, T3, T4 (ABORT ON ANY FAIL)
# ==============================================================================
print("\n=== SELF-CHECKS (T1 - T4) ===", flush=True)

# T4: Disjoint validation and held-out index sets for EEG and EMG
t4_eeg_disjoint = set(val_eeg_idx.tolist()).isdisjoint(set(held_eeg_idx.tolist()))
t4_emg_disjoint = set(val_emg_idx.tolist()).isdisjoint(set(held_emg_idx.tolist()))
if t4_eeg_disjoint and t4_emg_disjoint and len(val_eeg_idx) == 50 and len(val_emg_idx) == 50 and len(held_eeg_idx) == 100:
    print("  T4 (Disjoint EEG and EMG validation vs held-out indices): PASS", flush=True)
else:
    print("  T4 (Disjoint EEG and EMG validation vs held-out indices): FAIL", flush=True)
    assert False, "T4 FAILED: Validation and held-out sets are not disjoint."

# Helper to generate epoch pair (y, ref_n, x) identically to Round 4/5
def make_trial(X_arr, E_arr, idx, snr_db, sensor_rms_ratio, rng_obj):
    x = X_arr[idx]
    raw_n = E_arr[idx]
    n1 = lfilter(b_path1, a_path1, raw_n)
    n2 = -0.8 * lfilter(b_path2, a_path2, raw_n)
    n_c = np.concatenate([n1[:512], n2[512:]])
    g = rms(x) / ((10.0**(snr_db / 10.0)) * rms(n_c))
    y = x + g * n_c
    ref_n = raw_n + (rms(raw_n) * sensor_rms_ratio) * rng_obj.standard_normal(len(raw_n))
    return y, ref_n, x

# T1: M2 with gamma = 0 is bit-identical (np.array_equal) to M1 on 3 held-out epochs, P = 4 and 5
t1_ok = True
for P_test in (4, 5):
    rng_t1 = np.random.default_rng(777)
    for ep_i in range(3):
        y_t1, ref_t1, _ = make_trial(X_held, E_held, ep_i, -7.0, 0.01, rng_t1)
        out_m1 = run_dcd_rls_q30_int(y_t1, ref_t1, 16, P_test, Nu=4, Mb=16, H_int=16)
        out_m2 = run_dcd_ridge_int(y_t1, ref_t1, 16, P_test, gamma=0.0, Nu=4, Mb=16, H_int=16)
        if not np.array_equal(out_m1, out_m2):
            t1_ok = False
if t1_ok:
    print("  T1 (M2 gamma=0 bit-identical to M1 on 3 epochs, P=4,5):   PASS", flush=True)
else:
    print("  T1 (M2 gamma=0 bit-identical to M1 on 3 epochs, P=4,5):   FAIL", flush=True)
    sys.exit(1)

# T2: M3 with gamma = 0 vs M4: per-epoch RRMSE agree to 1e-6 relative on 5 epochs, P = 4 and 5
t2_ok = True
max_rel_err_t2 = 0.0
for P_test in (4, 5):
    rng_t2 = np.random.default_rng(777)
    for ep_i in range(5):
        y_t2, ref_t2, x_t2 = make_trial(X_held, E_held, ep_i, -7.0, 0.01, rng_t2)
        out_m3 = ridge_ewls(y_t2, ref_t2, 16, P_test, gamma=0.0)
        out_m4 = run_qr_rls_float64(y_t2, ref_t2, 16, P_test)
        for sl in (slice(256, 512), slice(512, 1024)):
            r3 = rms(out_m3[sl] - x_t2[sl]) / rms(x_t2[sl])
            r4 = rms(out_m4[sl] - x_t2[sl]) / rms(x_t2[sl])
            rel_err = abs(r3 - r4) / max(abs(r4), 1e-12)
            if rel_err > max_rel_err_t2:
                max_rel_err_t2 = rel_err
            if rel_err >= 1e-6:
                t2_ok = False
if t2_ok:
    print(f"  T2 (M3 gamma=0 vs M4 rel RRMSE < 1e-6 on 5 epochs):       PASS (max rel diff = {max_rel_err_t2:.2e})", flush=True)
else:
    print(f"  T2 (M3 gamma=0 vs M4 rel RRMSE < 1e-6 on 5 epochs):       FAIL (max rel diff = {max_rel_err_t2:.2e})", flush=True)
    sys.exit(1)

# T3: Reproduces Round 4/5 held-out values at (-7 dB, 1%) to 3 decimals:
#     M1 P=4 pre 1.198 / post 1.681; M1 P=5 pre 1.093 / post 1.920; M4 P=4 pre 1.341 / post 1.892
def check_t3_pair(fn, P_val):
    rng_t3 = np.random.default_rng(777)
    pre_list, post_list = [], []
    for i in range(100):
        y_i, ref_i, x_i = make_trial(X_held, E_held, i, -7.0, 0.01, rng_t3)
        xh = fn(y_i, ref_i, P_val)
        pre_list.append(rms(xh[256:512] - x_i[256:512]) / rms(x_i[256:512]))
        post_list.append(rms(xh[512:1024] - x_i[512:1024]) / rms(x_i[512:1024]))
    return round(float(np.mean(pre_list)), 3), round(float(np.mean(post_list)), 3)

m1_p4_pre, m1_p4_post = check_t3_pair(lambda y, r, P: run_dcd_rls_q30_int(y, r, 16, P, Nu=4, Mb=16), 4)
m1_p5_pre, m1_p5_post = check_t3_pair(lambda y, r, P: run_dcd_rls_q30_int(y, r, 16, P, Nu=4, Mb=16), 5)
m4_p4_pre, m4_p4_post = check_t3_pair(lambda y, r, P: run_qr_rls_float64(y, r, 16, P), 4)

t3_ok = (
    (m1_p4_pre == 1.198 and m1_p4_post == 1.681)
    and (m1_p5_pre == 1.093 and m1_p5_post == 1.920)
    and (m4_p4_pre == 1.341 and m4_p4_post == 1.892)
)
print(
    f"  T3 Values: M1 P=4 {m1_p4_pre:.3f}/{m1_p4_post:.3f} (target 1.198/1.681) | "
    f"M1 P=5 {m1_p5_pre:.3f}/{m1_p5_post:.3f} (target 1.093/1.920) | "
    f"M4 P=4 {m4_p4_pre:.3f}/{m4_p4_post:.3f} (target 1.341/1.892) -> {'PASS' if t3_ok else 'FAIL'}",
    flush=True,
)
if not t3_ok:
    sys.exit(1)

# ==============================================================================
# 4. VALIDATION TUNING WITH AUTOMATIC x4 BOUNDARY GRID EXTENSION (R4)
# ==============================================================================
if quick_mode:
    snr_grid = (-7.0,)
    noise_grid = (0.10,)
    P_grid = (4, 5)
    n_val_epochs = 10
    n_held_epochs = 10
else:
    snr_grid = (-7.0, -1.0, 2.0)
    noise_grid = (0.01, 0.10, 0.316)
    P_grid = (4, 5)
    n_val_epochs = 50
    n_held_epochs = 100

base_gamma_grid = [0.0] + [2.0**k for k in range(-8, 2)]  # {0, 2^-8, ..., 2^1}
base_mu_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Pre-compute causal validation scale factors (99.9th percentile * 1.05 per cell) for Secondary S3
causal_scales = {}
for snr_db in snr_grid:
    for s_noise in noise_grid:
        rng_cal = np.random.default_rng(313)
        all_d_abs, all_r_abs = [], []
        for i in range(n_val_epochs):
            y_v, ref_v, _ = make_trial(X_val, E_val, i, snr_db, s_noise, rng_cal)
            all_d_abs.append(np.abs(y_v))
            all_r_abs.append(np.abs(ref_v))
        sc_d_val = float(np.percentile(np.concatenate(all_d_abs), 99.9)) * 1.05
        sc_r_val = float(np.percentile(np.concatenate(all_r_abs), 99.9)) * 1.05
        causal_scales[(snr_db, s_noise)] = (sc_d_val, sc_r_val)


def eval_val_cell(method_fn, param_val, P_val, snr_db, s_noise):
    rng_v = np.random.default_rng(313)
    scores = []
    for i in range(n_val_epochs):
        y_v, ref_v, x_v = make_trial(X_val, E_val, i, snr_db, s_noise, rng_v)
        xh = method_fn(y_v, ref_v, P_val, param_val)
        r_pre = rms(xh[256:512] - x_v[256:512]) / rms(x_v[256:512])
        r_post = rms(xh[512:1024] - x_v[512:1024]) / rms(x_v[512:1024])
        scores.append(0.5 * (r_pre + r_post))
    return float(np.mean(scores))


def tune_with_extension(method_label, method_fn, P_val, cells_subset, init_grid, is_gamma=True):
    """
    Evaluates parameter grid on validation cells_subset.
    If selected value sits at either boundary of the positive grid (or min/max for mu),
    extends the grid x4 in that direction and re-tunes up to 3 times (R4).
    """
    grid = sorted(list(set(init_grid)))
    score_cache = {}

    def get_score(val):
        if val not in score_cache:
            cell_means = [eval_val_cell(method_fn, val, P_val, s_db, s_ns) for (s_db, s_ns) in cells_subset]
            score_cache[val] = float(np.mean(cell_means))
        return score_cache[val]

    for ext_iter in range(4):
        for g_val in grid:
            get_score(g_val)
        best_val = min(grid, key=lambda g: score_cache[g])
        pos_vals = [g for g in grid if g > 0.0]
        min_pos = min(pos_vals)
        max_pos = max(pos_vals)

        if ext_iter < 3:
            if best_val == max_pos:
                new_vals = [max_pos * 2.0, max_pos * 4.0]
                print(
                    f"  [R4 GRID EXTENSION #{ext_iter+1}] {method_label} (P={P_val}) hit upper boundary "
                    f"{best_val:g}; extending x4 upward with {new_vals}",
                    flush=True,
                )
                grid = sorted(list(set(grid + new_vals)))
                continue
            elif is_gamma and best_val == min_pos:
                new_vals = [min_pos / 2.0, min_pos / 4.0]
                print(
                    f"  [R4 GRID EXTENSION #{ext_iter+1}] {method_label} (P={P_val}) hit lower positive boundary "
                    f"{best_val:g}; extending x4 downward with {new_vals}",
                    flush=True,
                )
                grid = sorted(list(set(grid + new_vals)))
                continue
            elif (not is_gamma) and best_val == min_pos:
                new_vals = [min_pos / 2.0, min_pos / 4.0]
                print(
                    f"  [R4 GRID EXTENSION #{ext_iter+1}] {method_label} (P={P_val}) hit lower boundary "
                    f"{best_val:g}; extending x4 downward with {new_vals}",
                    flush=True,
                )
                grid = sorted(list(set(grid + new_vals)))
                continue
        break
    return best_val, score_cache[best_val]


print("\n=== VALIDATION HYPERPARAMETER TUNING (50 Validation Epochs, Seed 313) ===", flush=True)
m2_fn_tune = lambda y, r, P, gam: run_dcd_ridge_int(y, r, 16, P, gamma=gam, Nu=4, Mb=16, H_int=16)
m3_fn_tune = lambda y, r, P, gam: ridge_ewls(y, r, 16, P, gamma=gam)
m5_fn_tune = lambda y, r, P, mu: run_nlms(y, r, 16, mu=mu)

fixed_params = {}     # key: (method_id, P)
per_noise_params = {} # key: (method_id, P, s_noise)

for P_val in P_grid:
    all_cells = [(s_db, s_ns) for s_db in snr_grid for s_ns in noise_grid]
    g_m2_fix, sc_m2_fix = tune_with_extension(f"M2_FIXED_P{P_val}", m2_fn_tune, P_val, all_cells, base_gamma_grid, True)
    g_m3_fix, sc_m3_fix = tune_with_extension(f"M3_FIXED_P{P_val}", m3_fn_tune, P_val, all_cells, base_gamma_grid, True)
    m_m5_fix, sc_m5_fix = tune_with_extension(f"M5_FIXED_P{P_val}", m5_fn_tune, P_val, all_cells, base_mu_grid, False)
    fixed_params[("M2", P_val)] = g_m2_fix
    fixed_params[("M3", P_val)] = g_m3_fix
    fixed_params[("M5", P_val)] = m_m5_fix
    print(
        f"  FIXED Selections (P={P_val}): M2 gamma={g_m2_fix:g} (val={sc_m2_fix:.4f}) | "
        f"M3 gamma={g_m3_fix:g} (val={sc_m3_fix:.4f}) | M5 mu={m_m5_fix:g} (val={sc_m5_fix:.4f})",
        flush=True,
    )

    for s_ns in noise_grid:
        ns_cells = [(s_db, s_ns) for s_db in snr_grid]
        g_m2_ns, sc_m2_ns = tune_with_extension(
            f"M2_PERNOISE_{s_ns}_P{P_val}", m2_fn_tune, P_val, ns_cells, base_gamma_grid, True
        )
        g_m3_ns, sc_m3_ns = tune_with_extension(
            f"M3_PERNOISE_{s_ns}_P{P_val}", m3_fn_tune, P_val, ns_cells, base_gamma_grid, True
        )
        m_m5_ns, sc_m5_ns = tune_with_extension(
            f"M5_PERNOISE_{s_ns}_P{P_val}", m5_fn_tune, P_val, ns_cells, base_mu_grid, False
        )
        per_noise_params[("M2", P_val, s_ns)] = g_m2_ns
        per_noise_params[("M3", P_val, s_ns)] = g_m3_ns
        per_noise_params[("M5", P_val, s_ns)] = m_m5_ns
        print(
            f"  PER-NOISE Selections (P={P_val}, noise={s_ns*100:4.1f}%): "
            f"M2 gamma={g_m2_ns:g} (val={sc_m2_ns:.4f}) | "
            f"M3 gamma={g_m3_ns:g} (val={sc_m3_ns:.4f}) | "
            f"M5 mu={m_m5_ns:g} (val={sc_m5_ns:.4f})",
            flush=True,
        )

# ==============================================================================
# 5. SINGLE-PASS HELD-OUT EVALUATION ACROSS ALL 18 CELLS (SEED 777)
# ==============================================================================
print("\n=== RUNNING HELD-OUT EVALUATION (100 Held-Out Epochs, Seed 777) ===", flush=True)

held_records = {}  # key: (snr_db, s_noise, P_val, method_id, regime) -> dict of per-epoch arrays
hw_tracker = {
    "M1": {"updates": [], "max_up": 0, "bit_iters": [], "max_R": 0, "max_r": 0, "max_w": 0},
    "M2": {"updates": [], "max_up": 0, "bit_iters": [], "max_R": 0, "max_r": 0, "max_w": 0},
}


def run_held_cell(method_id, regime, fn_call, snr_db, s_noise, P_val, gamma_str, mu_str, track_hw_key=None):
    rng_h = np.random.default_rng(777)
    pre_arr = np.zeros(n_held_epochs, dtype=np.float64)
    post_arr = np.zeros(n_held_epochs, dtype=np.float64)
    w_arrs = [np.zeros(n_held_epochs, dtype=np.float64) for _ in wins]
    cc_arr = np.zeros(n_held_epochs, dtype=np.float64)

    for i in range(n_held_epochs):
        y_h, ref_h, x_h = make_trial(X_held, E_held, i, snr_db, s_noise, rng_h)
        if track_hw_key is not None:
            xh, hw = fn_call(y_h, ref_h)
            tr = hw_tracker[track_hw_key]
            tr["updates"].append(hw["mean_updates"])
            if hw["max_updates"] > tr["max_up"]:
                tr["max_up"] = hw["max_updates"]
            tr["bit_iters"].append(hw["mean_bit_iters"])
            if hw["max_abs_R"] > tr["max_R"]:
                tr["max_R"] = hw["max_abs_R"]
            if hw["max_abs_r"] > tr["max_r"]:
                tr["max_r"] = hw["max_abs_r"]
            if hw["max_abs_w"] > tr["max_w"]:
                tr["max_w"] = hw["max_abs_w"]
        else:
            xh = fn_call(y_h, ref_h)

        pre_arr[i] = rms(xh[256:512] - x_h[256:512]) / rms(x_h[256:512])
        post_arr[i] = rms(xh[512:1024] - x_h[512:1024]) / rms(x_h[512:1024])
        for j, (s0, s1) in enumerate(wins):
            w_arrs[j][i] = rms(xh[s0:s1] - x_h[s0:s1]) / rms(x_h[s0:s1])
        if np.std(xh[512:1024]) > 1e-12:
            cc_arr[i] = float(np.corrcoef(xh[512:1024], x_h[512:1024])[0, 1])
        else:
            cc_arr[i] = 0.0

    rec = {
        "snr": snr_db,
        "noise": s_noise,
        "P": P_val,
        "method": method_id,
        "regime": regime,
        "gamma": gamma_str,
        "mu": mu_str,
        "pre_epochs": pre_arr,
        "post_epochs": post_arr,
        "pre": float(np.mean(pre_arr)),
        "post": float(np.mean(post_arr)),
        "w1": float(np.mean(w_arrs[0])),
        "w2": float(np.mean(w_arrs[1])),
        "w3": float(np.mean(w_arrs[2])),
        "w4": float(np.mean(w_arrs[3])),
        "cc_post": float(np.mean(cc_arr)),
    }
    held_records[(snr_db, s_noise, P_val, method_id, regime)] = rec
    return rec


csv_results_rows = []
for P_val in P_grid:
    for s_noise in noise_grid:
        for snr_db in snr_grid:
            print(f"  Scoring Cell: P={P_val}, noise={s_noise*100:4.1f}%, SNR={snr_db:+.0f} dB ...", flush=True)
            # M1: plain Int DCD-RLS (gamma=0)
            r_m1 = run_held_cell(
                "M1",
                "NONE",
                lambda y, r: run_dcd_ridge_int(y, r, 16, P_val, gamma=0.0, Nu=4, Mb=16, H_int=16, return_hw_stats=True),
                snr_db,
                s_noise,
                P_val,
                "0",
                "NA",
                track_hw_key="M1",
            )
            csv_results_rows.append(r_m1)

            # M2 FIXED and PER-NOISE
            gam_m2_fix = fixed_params[("M2", P_val)]
            r_m2_fix = run_held_cell(
                "M2",
                "FIXED",
                lambda y, r: run_dcd_ridge_int(
                    y, r, 16, P_val, gamma=gam_m2_fix, Nu=4, Mb=16, H_int=16, return_hw_stats=True
                ),
                snr_db,
                s_noise,
                P_val,
                f"{gam_m2_fix:g}",
                "NA",
                track_hw_key="M2",
            )
            csv_results_rows.append(r_m2_fix)

            gam_m2_ns = per_noise_params[("M2", P_val, s_noise)]
            r_m2_ns = run_held_cell(
                "M2",
                "PER-NOISE",
                lambda y, r: run_dcd_ridge_int(y, r, 16, P_val, gamma=gam_m2_ns, Nu=4, Mb=16, H_int=16),
                snr_db,
                s_noise,
                P_val,
                f"{gam_m2_ns:g}",
                "NA",
            )
            csv_results_rows.append(r_m2_ns)

            # M3 FIXED and PER-NOISE
            gam_m3_fix = fixed_params[("M3", P_val)]
            r_m3_fix = run_held_cell(
                "M3",
                "FIXED",
                lambda y, r: ridge_ewls(y, r, 16, P_val, gamma=gam_m3_fix),
                snr_db,
                s_noise,
                P_val,
                f"{gam_m3_fix:g}",
                "NA",
            )
            csv_results_rows.append(r_m3_fix)

            gam_m3_ns = per_noise_params[("M3", P_val, s_noise)]
            r_m3_ns = run_held_cell(
                "M3",
                "PER-NOISE",
                lambda y, r: ridge_ewls(y, r, 16, P_val, gamma=gam_m3_ns),
                snr_db,
                s_noise,
                P_val,
                f"{gam_m3_ns:g}",
                "NA",
            )
            csv_results_rows.append(r_m3_ns)

            # M4: Givens QR-RLS (unregularised)
            r_m4 = run_held_cell(
                "M4",
                "NONE",
                lambda y, r: run_qr_rls_float64(y, r, 16, P_val),
                snr_db,
                s_noise,
                P_val,
                "0",
                "NA",
            )
            csv_results_rows.append(r_m4)

            # M5 FIXED and PER-NOISE
            mu_m5_fix = fixed_params[("M5", P_val)]
            r_m5_fix = run_held_cell(
                "M5",
                "FIXED",
                lambda y, r: run_nlms(y, r, 16, mu=mu_m5_fix),
                snr_db,
                s_noise,
                P_val,
                "NA",
                f"{mu_m5_fix:g}",
            )
            csv_results_rows.append(r_m5_fix)

            mu_m5_ns = per_noise_params[("M5", P_val, s_noise)]
            r_m5_ns = run_held_cell(
                "M5",
                "PER-NOISE",
                lambda y, r: run_nlms(y, r, 16, mu=mu_m5_ns),
                snr_db,
                s_noise,
                P_val,
                "NA",
                f"{mu_m5_ns:g}",
            )
            csv_results_rows.append(r_m5_ns)

            # M6: Standard Float64 RLS (untuned)
            r_m6 = run_held_cell(
                "M6",
                "NONE",
                lambda y, r: run_rls_float64(y, r, 16, P_val),
                snr_db,
                s_noise,
                P_val,
                "0",
                "NA",
            )
            csv_results_rows.append(r_m6)

# ==============================================================================
# 6. HUMAN-READABLE SUMMARY TABLE & PRE-REGISTERED HYPOTHESES (H1 - H4)
# ==============================================================================
print("\n=== HELD-OUT RESULTS SUMMARY TABLE (Pre / Post RRMSE | W1 W2 W3 W4 | CC_post) ===")
print("  Cell (SNR, Noise, P) | Method (Regime, Param)     | Pre   | Post  | W1    | W2    | W3    | W4    | CC")
for r in csv_results_rows:
    cell_str = f"{r['snr']:+2.0f}dB,{r['noise']*100:4.1f}%,P={r['P']}"
    p_str = f"g={r['gamma']}" if r["gamma"] != "NA" else f"mu={r['mu']}"
    m_str = f"{r['method']}({r['regime']},{p_str})"
    print(
        f"  {cell_str:20s} | {m_str:26s} | {r['pre']:5.3f} | {r['post']:5.3f} | "
        f"{r['w1']:5.3f} | {r['w2']:5.3f} | {r['w3']:5.3f} | {r['w4']:5.3f} | {r['cc_post']:5.3f}"
    )


def paired_boot_stats(diff_vec, n_boot=10000, seed=42):
    arr = np.asarray(diff_vec, dtype=np.float64)
    rng_b = np.random.default_rng(seed)
    idx_b = rng_b.choice(len(arr), size=(n_boot, len(arr)), replace=True)
    means_b = arr[idx_b].mean(axis=1)
    ci_lo, ci_hi = np.percentile(means_b, [2.5, 97.5])
    if np.all(arr == 0.0):
        p_raw = 1.0
    else:
        _, p_raw = wilcoxon(arr)
    return float(np.mean(arr)), float(np.median(arr)), float(ci_lo), float(ci_hi), float(p_raw)


def holm_adjust(p_list):
    m = len(p_list)
    order = np.argsort(p_list)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p_list[idx]
        if val > running_max:
            running_max = val
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()


hyp_defs = [
    ("H1", "M1", "NONE", "superiority"),
    ("H2", "M3", "FIXED", "non_inferiority"),
    ("H3", "M5", "FIXED", "superiority"),
    ("H4", "M4", "NONE", "superiority"),
]

csv_tests_rows = []
print("\n=== PRE-REGISTERED HYPOTHESIS TESTS (Primary Metric = Post-Switch RRMSE, FIXED Regime) ===")
for h_id, comp_m, comp_reg, h_type in hyp_defs:
    raw_rows = []
    raw_ps = []
    for P_val in P_grid:
        for s_noise in noise_grid:
            for snr_db in snr_grid:
                rec_m2 = held_records[(snr_db, s_noise, P_val, "M2", "FIXED")]
                rec_c = held_records[(snr_db, s_noise, P_val, comp_m, comp_reg)]
                diff = rec_m2["post_epochs"] - rec_c["post_epochs"]
                mean_d, med_d, ci_lo, ci_hi, p_raw = paired_boot_stats(diff, n_boot=10000, seed=42)
                margin = 0.02 * rec_c["post"]
                cell_tag = f"SNR={snr_db:+.0f}dB_Noise={s_noise*100:.1f}%_P={P_val}"
                raw_rows.append((cell_tag, mean_d, med_d, ci_lo, ci_hi, p_raw, margin))
                raw_ps.append(p_raw)

    adj_ps = holm_adjust(raw_ps)
    print(f"\n  --- {h_id}: M2(FIXED) vs {comp_m}({comp_reg}) [{h_type}] ---")
    for (cell_tag, mean_d, med_d, ci_lo, ci_hi, p_raw, margin), p_holm in zip(raw_rows, adj_ps):
        if h_type == "non_inferiority":
            verdict = "PASS" if (ci_hi < margin) else "FAIL"
            extra_info = f"(margin 0.02*mean(M3) = {margin:+.4f})"
        else:
            verdict = "PASS" if (mean_d < 0.0 and p_holm < 0.05) else "FAIL"
            extra_info = ""
        print(
            f"  {h_id} | {cell_tag:28s} | mean={mean_d:+.4f} med={med_d:+.4f} "
            f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}] | p_raw={p_raw:.3e} p_holm={p_holm:.3e} | {verdict} {extra_info}"
        )
        csv_tests_rows.append(
            {
                "hypothesis": h_id,
                "cell": cell_tag,
                "mean_diff": mean_d,
                "median_diff": med_d,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_raw": p_raw,
                "p_holm": p_holm,
                "verdict": verdict,
            }
        )

# ==============================================================================
# 7. HARDWARE MEASUREMENTS (M1, M2)
# ==============================================================================
print("\n=== HARDWARE MEASUREMENTS (M1 & M2 OVER ALL HELD-OUT RUNS) ===")
for m_key in ("M1", "M2"):
    tr = hw_tracker[m_key]
    mean_up = float(np.mean(tr["updates"]))
    max_up = int(tr["max_up"])
    mean_bi = float(np.mean(tr["bit_iters"]))
    bw_R = int(np.ceil(np.log2(max(tr["max_R"], 1) + 1.0))) + 1
    bw_r = int(np.ceil(np.log2(max(tr["max_r"], 1) + 1.0))) + 1
    bw_w = int(np.ceil(np.log2(max(tr["max_w"], 1) + 1.0))) + 1
    print(
        f"  {m_key}: mean_updates/samp = {mean_up:.3f}, max_updates/samp = {max_up}, "
        f"mean_bit_iters/samp = {mean_bi:.3f}"
    )
    print(
        f"      max |R_q30| = {tr['max_R']} ({bw_R} signed bits), "
        f"max |r_q30| = {tr['max_r']} ({bw_r} signed bits), "
        f"max |w_q15| = {tr['max_w']} ({bw_w} signed bits)"
    )

L_eval, Nu_eval, Mb_eval = 16, 4, 16
m1_mults = 3 * L_eval
m2_mults = 4 * L_eval  # 3L (FIR + R_col0 + e0*u) + L (leakage term)
m1_adds_max = 5 * L_eval + Nu_eval * (L_eval + 1) + Nu_eval * (L_eval - 1) + Nu_eval + Mb_eval
m2_adds_max = 6 * L_eval + Nu_eval * (L_eval + 2) + Nu_eval * (L_eval - 1) + Nu_eval + Mb_eval
print(
    f"  Formula at L={L_eval}, Nu={Nu_eval}, Mb={Mb_eval}: "
    f"M1 multiplies/samp = 3*L = {m1_mults}, max adds+cmps/samp = {m1_adds_max}; "
    f"M2 multiplies/samp = 4*L = {m2_mults} (including L leakage multiplies), max adds+cmps/samp = {m2_adds_max}"
)

# ==============================================================================
# 8. SECONDARY ANALYSES (S1, S2, S3)
# ==============================================================================
print("\n=== SECONDARY S1: Nu SWEEP {1, 2, 4, 8} FOR M2 AT FIXED GAMMA, CELLS (-7 dB, 10%) & (-1 dB, 10%) ===")
s1_snrs = [s for s in (-7.0, -1.0) if s in snr_grid]
s1_noise = 0.10 if 0.10 in noise_grid else noise_grid[0]
for P_val in P_grid:
    gam_fix = fixed_params[("M2", P_val)]
    for snr_db in s1_snrs:
        for nu_val in (1, 2, 4, 8):
            rng_s1 = np.random.default_rng(777)
            pre_s1, post_s1 = [], []
            for i in range(n_held_epochs):
                y_i, ref_i, x_i = make_trial(X_held, E_held, i, snr_db, s1_noise, rng_s1)
                xh = run_dcd_ridge_int(y_i, ref_i, 16, P_val, gamma=gam_fix, Nu=nu_val, Mb=16, H_int=16)
                pre_s1.append(rms(xh[256:512] - x_i[256:512]) / rms(x_i[256:512]))
                post_s1.append(rms(xh[512:1024] - x_i[512:1024]) / rms(x_i[512:1024]))
            print(
                f"  SECONDARY S1 | P={P_val}, SNR={snr_db:+.0f}dB, Noise={s1_noise*100:.1f}%, Nu={nu_val} | "
                f"Pre RRMSE = {np.mean(pre_s1):.4f} | Post RRMSE = {np.mean(post_s1):.4f}",
                flush=True,
            )

print("\n=== SECONDARY S2: POWER-OF-TWO ROUNDING OF delta_q30 (CHANGE IN HELD-OUT POST RRMSE) ===")
for P_val in P_grid:
    gam_fix = fixed_params[("M2", P_val)]
    for s_noise in noise_grid:
        for snr_db in snr_grid:
            rng_s2 = np.random.default_rng(777)
            post_p2 = []
            for i in range(n_held_epochs):
                y_i, ref_i, x_i = make_trial(X_held, E_held, i, snr_db, s_noise, rng_s2)
                xh = run_dcd_ridge_int(y_i, ref_i, 16, P_val, gamma=gam_fix, Nu=4, Mb=16, H_int=16, pow2_load=True)
                post_p2.append(rms(xh[512:1024] - x_i[512:1024]) / rms(x_i[512:1024]))
            base_post = held_records[(snr_db, s_noise, P_val, "M2", "FIXED")]["post"]
            mean_p2 = float(np.mean(post_p2))
            print(
                f"  SECONDARY S2 | P={P_val}, SNR={snr_db:+.0f}dB, Noise={s_noise*100:4.1f}% | "
                f"Exact Post={base_post:.4f}, Pow2 Post={mean_p2:.4f}, Delta={mean_p2 - base_post:+.5f}",
                flush=True,
            )

print("\n=== SECONDARY S3: CAUSAL VALIDATION-SET SCALING (99.9th PERCENTILE x 1.05 PER CELL) ===")
for P_val in P_grid:
    gam_fix = fixed_params[("M2", P_val)]
    for s_noise in noise_grid:
        for snr_db in snr_grid:
            sc_d_c, sc_r_c = causal_scales[(snr_db, s_noise)]
            rng_s3 = np.random.default_rng(777)
            post_c = []
            clip_cnt = 0
            tot_cnt = 0
            for i in range(n_held_epochs):
                y_i, ref_i, x_i = make_trial(X_held, E_held, i, snr_db, s_noise, rng_s3)
                xh, hw_c = run_dcd_ridge_int(
                    y_i,
                    ref_i,
                    16,
                    P_val,
                    gamma=gam_fix,
                    Nu=4,
                    Mb=16,
                    H_int=16,
                    scale_d=sc_d_c,
                    scale_r=sc_r_c,
                    return_hw_stats=True,
                )
                post_c.append(rms(xh[512:1024] - x_i[512:1024]) / rms(x_i[512:1024]))
                clip_cnt += hw_c["clipped_samples"]
                tot_cnt += hw_c["total_samples"]
            base_post = held_records[(snr_db, s_noise, P_val, "M2", "FIXED")]["post"]
            mean_c = float(np.mean(post_c))
            pct_clip = 100.0 * clip_cnt / float(max(tot_cnt, 1))
            print(
                f"  SECONDARY S3 | P={P_val}, SNR={snr_db:+.0f}dB, Noise={s_noise*100:4.1f}% | "
                f"Clipped={pct_clip:.3f}% | NonCausal Post={base_post:.4f}, Causal Post={mean_c:.4f}, "
                f"Delta={mean_c - base_post:+.5f}",
                flush=True,
            )

# ==============================================================================
# 9. MACHINE-READABLE CSV OUTPUT BLOCKS
# ==============================================================================
print("\n=== CSV RESULTS BEGIN ===")
print("snr,noise,P,method,regime,gamma,mu,pre,post,w1,w2,w3,w4,cc_post")
for r in csv_results_rows:
    print(
        f"{r['snr']:.1f},{r['noise']:.3f},{r['P']},{r['method']},{r['regime']},"
        f"{r['gamma']},{r['mu']},{r['pre']:.6f},{r['post']:.6f},"
        f"{r['w1']:.6f},{r['w2']:.6f},{r['w3']:.6f},{r['w4']:.6f},{r['cc_post']:.6f}"
    )
print("=== CSV RESULTS END ===")

print("\n=== CSV TESTS BEGIN ===")
print("hypothesis,cell,mean_diff,median_diff,ci_lo,ci_hi,p_raw,p_holm,verdict")
for t in csv_tests_rows:
    print(
        f"{t['hypothesis']},{t['cell']},{t['mean_diff']:.6f},{t['median_diff']:.6f},"
        f"{t['ci_lo']:.6f},{t['ci_hi']:.6f},{t['p_raw']:.6e},{t['p_holm']:.6e},{t['verdict']}"
    )
print("=== CSV TESTS END ===")

end_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
print(f"\nEnd Time (UTC): {end_iso} | Total Elapsed: {time.time() - start_wall:.1f} s", flush=True)
