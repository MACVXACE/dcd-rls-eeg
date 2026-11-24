"""
stage2_round4_benchmark.py
Round-4-compliant benchmark for the 4-page IEEE EMBC paper:
  - Block 0: Horowitz (2014) pJ/sample table (NLMS, Int DCD-RLS, Fixed-Pt RLS, Float RLS, Givens QR-RLS).
  - Block 1: Mid-record path-switch tracking & short-memory stability benchmark across P in {4,5,6,7,8}
    and NLMS mu in {0.05,0.1,0.2,0.5}, including Givens QR-RLS, Fixed-Pt Standard RLS, 1000-sample
    bootstrap 95% CIs, paired Wilcoxon p-values, multi-SNR (-7, -1, +2 dB), and realistic reference
    sensor noise (1% RMS = 40 dB power SNR, 10% RMS = 20 dB power SNR, 31.6% RMS = 10 dB power SNR).
  - Block 2: Non-tautological Stage A drift test (exponential polarization step-relaxation + Brownian
    random walk, no linear ramp) with identical first-sample initialization.
  - Block 3: Real PhysioNet 'motion-artifact' loader using the `wfdb` package (2048 Hz aligned records
    decimated 8x to 256 Hz), automatic clean vs. disturbed channel identification, active-artifact vs.
    rest segmentation, 3-axis accelerometer tapped-delay-line reference, and Stage A + Stage B ablation.

Usage:
  python3 stage2_round4_benchmark.py /path/to/EEGdenoiseNet [/path/to/wfdb_dir_or_'physionet_remote']
"""
import sys, os, glob, numpy as np
from scipy.signal import lfilter, butter, filtfilt, resample_poly
from scipy.stats import wilcoxon

rms = lambda v: float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64)**2)))

def boot_ci(vals, n_boot=1000, seed=42):
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)

# ==============================================================================
# BLOCK 0: HOROWITZ (2014) ENERGY PER SAMPLE (INCLUDES FIXED-PT RLS & QR-RLS)
# ==============================================================================
def print_energy_table_round4():
    E_add32 = 0.10
    E_mult32 = 3.10
    E_mult16_scaled = 3.10 * (16.0 / 32.0)**2  # 0.775 pJ
    E_fp32_add = 0.90
    E_fp32_mult = 3.70
    E_fp32_div_sqrt = 10.0

    print("=== BLOCK 0: ENERGY PER SAMPLE (pJ/sample) — HOROWITZ (2014) 45nm 0.9V ===")
    for L in (16, 32):
        Nu, Mb = 4, 16
        mem_nlms  = 3 * L
        mem_dcd   = (6 + 3 * Nu) * L
        mem_rls   = 3 * (L**2) + 6 * L
        mem_qrrls = (L**2) + 7 * L

        for mult_label, E_m_int in [("16b-Scaled Mult (0.775 pJ)", E_mult16_scaled),
                                    ("Horowitz 32b Mult (3.10 pJ)", E_mult32)]:
            arith_nlms   = (2*L + 3)*E_m_int + (2*L + 1)*E_add32 + 5.0
            arith_dcd    = (3*L)*E_m_int + (5*L + Nu*(L+1) + Nu*(L-1) + Nu + Mb)*E_add32
            arith_fx_rls = (3*L**2 + 4*L)*E_m_int + (3*L**2 + 3*L)*E_add32 + 5.0
            arith_fp_rls = (3*L**2 + 4*L)*E_fp32_mult + (3*L**2 + 3*L)*E_fp32_add + 10.0
            arith_qr_rls = (3*L**2 + 8*L)*E_fp32_mult + int(1.5*L**2 + 3.5*L)*E_fp32_add + (4*L)*E_fp32_div_sqrt

            print(f"  L={L:2d}, Nu={Nu}, Mb={Mb} [{mult_label}]:")
            for e_mem in (0.0, 5.0):
                t_nlms   = arith_nlms   + mem_nlms  * e_mem
                t_dcd    = arith_dcd    + mem_dcd   * e_mem
                t_fx_rls = arith_fx_rls + mem_rls   * e_mem
                t_fp_rls = arith_fp_rls + mem_rls   * e_mem
                t_qr_rls = arith_qr_rls + mem_qrrls * e_mem
                print(f"    e_mem={e_mem:3.1f} pJ | NLMS={t_nlms:6.1f} | DCD={t_dcd:6.1f} | "
                      f"FxRLS={t_fx_rls:7.1f} | FPRLS={t_fp_rls:7.1f} | QR-RLS={t_qr_rls:7.1f} | "
                      f"DCD/NLMS={t_dcd/t_nlms:3.1f}x | QR/DCD={t_qr_rls/t_dcd:4.1f}x")

print_energy_table_round4()

if len(sys.argv) < 2:
    sys.exit(0)

d_eeg = sys.argv[1]
d_phys = sys.argv[2] if len(sys.argv) > 2 else None
fs = 512.0
N = 1024

# ==============================================================================
# STAGE A & STAGE B SOLVER DEFINITIONS (SINGLE-CHANNEL & MULTI-AXIS REFERENCE)
# ==============================================================================
def dc_blocker_1st(y, P_pole=9):
    a = 1.0 - 2.0**(-P_pole)
    out = np.zeros_like(y, dtype=np.float64)
    x_prev, y_prev = float(y[0]), 0.0
    for k in range(len(y)):
        out[k] = y[k] - x_prev + a * y_prev
        x_prev = float(y[k]); y_prev = float(out[k])
    return out

def alpha_beta_tracker(y, alpha=0.008, mode="benedict_bordner"):
    beta = (alpha**2) / (2.0 - alpha) if mode == "benedict_bordner" else 2.0*(2.0 - alpha) - 4.0*np.sqrt(1.0 - alpha)
    d_hat, v_hat = float(y[0]), 0.0
    out = np.zeros_like(y, dtype=np.float64)
    for k in range(len(y)):
        d_pred = d_hat + v_hat
        err = float(y[k]) - d_pred
        d_hat = d_pred + alpha * err
        v_hat = v_hat + beta * err
        out[k] = y[k] - d_hat
    return out

def _build_tap_vectors(ref_sig, L_per_axis):
    """Supports both 1D reference (N,) and multi-axis reference (N, M_axes) -> (N, L_total) tap matrix."""
    ref = np.asarray(ref_sig, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    N_s, M_ax = ref.shape
    L_tot = L_per_axis * M_ax
    U_all = np.zeros((N_s, L_tot), dtype=np.float64)
    u_buf = np.zeros((M_ax, L_per_axis), dtype=np.float64)
    for k in range(N_s):
        u_buf[:, 1:] = u_buf[:, :-1]
        u_buf[:, 0] = ref[k, :]
        U_all[k, :] = u_buf.ravel()
    return U_all

def run_nlms(d_sig, ref_sig, L=16, mu=0.2, eps=1e-4):
    U_all = _build_tap_vectors(ref_sig, L)
    L_tot = U_all.shape[1]
    w = np.zeros(L_tot, dtype=np.float64)
    out = np.zeros(len(d_sig), dtype=np.float64)
    for k in range(len(d_sig)):
        u = U_all[k]
        e = d_sig[k] - np.dot(w, u)
        out[k] = e
        w += (mu / (np.dot(u, u) + eps)) * e * u
    return out

def run_rls_float64(d_sig, ref_sig, L=16, P_shift=5, p0=1.0):
    lam = 1.0 - 2.0**(-P_shift)
    U_all = _build_tap_vectors(ref_sig, L)
    L_tot = U_all.shape[1]
    w = np.zeros(L_tot, dtype=np.float64)
    P = np.eye(L_tot, dtype=np.float64) / (p0 * np.var(ref_sig) + 1e-6)
    out = np.zeros(len(d_sig), dtype=np.float64)
    for k in range(len(d_sig)):
        u = U_all[k]
        e = d_sig[k] - np.dot(w, u)
        out[k] = e
        Pu = P @ u
        denom = lam + np.dot(u, Pu)
        if abs(denom) < 1e-15:
            denom = 1e-15
        g = Pu / denom
        w += g * e
        P = (P - np.outer(g, Pu)) / lam
    return out

def run_rls_fixed_q30(d_sig, ref_sig, L=16, P_shift=5, p0=1.0):
    """Quantization-aware Standard Matrix-Inversion-Lemma RLS (Q15 inputs, Q20 covariance P_mat)."""
    scale_d = max(np.max(np.abs(d_sig)), 1e-9) * 1.05
    scale_r = max(np.max(np.abs(ref_sig)), 1e-9) * 1.05
    Q15 = 32768.0
    d_q = np.clip(np.round((d_sig / scale_d) * Q15), -32768, 32767) / Q15
    r_q = np.clip(np.round((ref_sig / scale_r) * Q15), -32768, 32767) / Q15
    U_all = _build_tap_vectors(r_q, L)
    L_tot = U_all.shape[1]
    lam = 1.0 - 2.0**(-P_shift)
    w = np.zeros(L_tot, dtype=np.float64)
    P = np.eye(L_tot, dtype=np.float64) / (p0 * np.var(r_q) + 1e-4)
    out = np.zeros(len(d_sig), dtype=np.float64)
    Q20 = float(1 << 20)
    for k in range(len(d_sig)):
        u = U_all[k]
        e = d_q[k] - np.dot(w, u)
        out[k] = e * scale_d
        Pu = np.round((P @ u) * Q20) / Q20
        denom = lam + np.dot(u, Pu)
        if abs(denom) < 1e-9:
            denom = 1e-9
        g = np.round((Pu / denom) * Q20) / Q20
        w = np.clip(np.round((w + g * e) * Q15) / Q15, -16.0, 16.0)
        P = np.clip(np.round(((P - np.outer(g, Pu)) / lam) * Q20) / Q20, -1e6, 1e6)
    return out

def run_qr_rls_float64(d_sig, ref_sig, L=16, P_shift=5, p0=1.0):
    """
    Numerically stabilized Givens-Rotation Square-Root QR-RLS:
      Propagates the upper-triangular Cholesky factor U_fac of R(k) via orthogonal Givens rotations.
    """
    lam = 1.0 - 2.0**(-P_shift)
    sqrt_lam = np.sqrt(lam)
    U_all = _build_tap_vectors(ref_sig, L)
    L_tot = U_all.shape[1]
    reg_init = np.sqrt(p0 * np.var(ref_sig) + 1e-6)
    U_fac = np.eye(L_tot, dtype=np.float64) * reg_init
    z_vec = np.zeros(L_tot, dtype=np.float64)
    w = np.zeros(L_tot, dtype=np.float64)
    out = np.zeros(len(d_sig), dtype=np.float64)

    for k in range(len(d_sig)):
        u = U_all[k]
        out[k] = d_sig[k] - np.dot(w, u)
        Aug = np.zeros((L_tot + 1, L_tot + 1), dtype=np.float64)
        Aug[:L_tot, :L_tot] = sqrt_lam * U_fac
        Aug[:L_tot, L_tot]  = sqrt_lam * z_vec
        Aug[L_tot, :L_tot]  = u
        Aug[L_tot, L_tot]   = d_sig[k]

        for i in range(L_tot):
            a_ii = Aug[i, i]
            b_li = Aug[L_tot, i]
            if abs(b_li) > 1e-15:
                r_hyp = np.hypot(a_ii, b_li)
                c = a_ii / r_hyp
                s = b_li / r_hyp
                row_i = c * Aug[i, i:] + s * Aug[L_tot, i:]
                row_L = -s * Aug[i, i:] + c * Aug[L_tot, i:]
                Aug[i, i:] = row_i
                Aug[L_tot, i:] = row_L

        U_fac = Aug[:L_tot, :L_tot]
        z_vec = Aug[:L_tot, L_tot]
        U_safe = U_fac.copy()
        for i in range(L_tot):
            if abs(U_safe[i, i]) < 1e-8:
                U_safe[i, i] = 1e-8 if U_safe[i, i] >= 0 else -1e-8
        w = np.linalg.solve(U_safe, z_vec)
    return out

def run_dcd_rls_q30_int(d_sig, ref_sig, L=16, P_shift=5, Nu=4, Mb=16, H_int=16, delta_reg=1.0):
    """
    Pure signed-integer Q15/Q30 Leading-Element DCD-RLS (supports 1D or multi-axis ref_sig).
    """
    scale_d = max(np.max(np.abs(d_sig)), 1e-9) * 1.05
    scale_r = max(np.max(np.abs(ref_sig)), 1e-9) * 1.05
    Q15 = 1 << 15
    Q30 = 1 << 30

    d_q15 = np.clip(np.round((d_sig / scale_d) * Q15), -Q15, Q15 - 1).astype(np.int64)
    r_q15 = np.clip(np.round((ref_sig / scale_r) * Q15), -Q15, Q15 - 1)
    U_all = _build_tap_vectors(r_q15, L).astype(np.int64)
    L_tot = U_all.shape[1]
    single_axis = (np.asarray(ref_sig).ndim == 1)

    w_q15 = np.zeros(L_tot, dtype=np.int64)
    reg_q30 = max(int(round(delta_reg * np.var(r_q15 / float(Q15)) * Q30)), 1)
    R_q30 = np.eye(L_tot, dtype=np.int64) * reg_q30
    r_q30 = np.zeros(L_tot, dtype=np.int64)
    out = np.zeros(len(d_sig), dtype=np.float64)

    for k in range(len(d_sig)):
        u_q15 = U_all[k]
        y_hat_q15 = int(np.dot(w_q15, u_q15) >> 15)
        e_q15 = int(d_q15[k] - y_hat_q15)
        out[k] = (e_q15 / float(Q15)) * scale_d

        if single_axis:
            R_q30[1:, 1:] = R_q30[:-1, :-1]
            col0 = R_q30[:, 0] - (R_q30[:, 0] >> P_shift) + u_q15[0] * u_q15
            R_q30[:, 0] = col0; R_q30[0, :] = col0
        else:
            R_q30 = R_q30 - (R_q30 >> P_shift) + np.outer(u_q15, u_q15)

        r_q30 = r_q30 - (r_q30 >> P_shift) + e_q15 * u_q15

        m = 1; u_cnt = 0
        while m <= Mb and u_cnt < Nu:
            n = int(np.argmax(np.abs(r_q30)))
            sh_th = m - 3
            thresh = (R_q30[n, n] >> sh_th) if sh_th >= 0 else (R_q30[n, n] << (-sh_th))
            if np.abs(r_q30[n]) > thresh:
                s = 1 if r_q30[n] > 0 else -1
                step_q15 = (H_int * Q15) >> m
                w_q15[n] += s * step_q15
                sh_r = m - 4
                r_q30 -= s * ((R_q30[:, n] >> sh_r) if sh_r >= 0 else (R_q30[:, n] << (-sh_r)))
                u_cnt += 1
            else:
                m += 1
    return out

# ==============================================================================
# LOAD EEGDENOISENET HELD-OUT SPLIT (100 EPOCHS, DISJOINT FROM SEED=7)
# ==============================================================================
eeg_raw = np.load(os.path.join(d_eeg, "EEG_all_epochs.npy"))
emg_raw = np.load(os.path.join(d_eeg, "EMG_all_epochs.npy"))
R = resample_poly(eeg_raw, 2, 1, axis=1) if eeg_raw.shape[1] == 512 else eeg_raw.copy()
E = resample_poly(emg_raw, 2, 1, axis=1) if emg_raw.shape[1] == 512 else emg_raw.copy()
R -= R.mean(1, keepdims=True); R /= R.std(1, keepdims=True)
E -= E.mean(1, keepdims=True); E /= E.std(1, keepdims=True)

val_pool_idx = np.random.default_rng(7).choice(len(R), 200, replace=False)
used_set = set(val_pool_idx.tolist())
held_idx = np.array([i for i in np.random.default_rng(99).permutation(len(R)) if i not in used_set][:100])
X_held = R[held_idx]
E_held = E[np.random.default_rng(101).choice(len(E), 100, replace=False)]

b_path1, a_path1 = butter(3, [5.0 / (fs / 2.0), 100.0 / (fs / 2.0)], btype='band')
b_path2, a_path2 = butter(3, [12.0 / (fs / 2.0), 75.0 / (fs / 2.0)], btype='band')
wins = [(512, 576), (576, 640), (640, 768), (768, 1024)]

# ==============================================================================
# BLOCK 1: TRACKING & STABILITY SWEEP (QR-RLS, 95% CIs, WILCOXON, SENSOR NOISE)
# ==============================================================================
def eval_b2_detailed(fn, snr_db=-7.0, sensor_rms_ratio=0.01):
    rng_sw = np.random.default_rng(777)
    pre_rr = []
    w_rr = [[] for _ in wins]
    post_all = []
    for i in range(100):
        x = X_held[i]; raw_n = E_held[i]
        n1 = lfilter(b_path1, a_path1, raw_n)
        n2 = -0.8 * lfilter(b_path2, a_path2, raw_n)
        n_c = np.concatenate([n1[:512], n2[512:]])
        g = rms(x) / ((10.0**(snr_db / 10.0)) * rms(n_c))
        y = x + g * n_c
        ref_n = raw_n + (rms(raw_n) * sensor_rms_ratio) * rng_sw.standard_normal(len(raw_n))
        xh = fn(y, ref_n)
        pre_rr.append(rms(xh[256:512] - x[256:512]) / rms(x[256:512]))
        post_all.append(rms(xh[512:1024] - x[512:1024]) / rms(x[512:1024]))
        for j, (s0, s1) in enumerate(wins):
            w_rr[j].append(rms(xh[s0:s1] - x[s0:s1]) / rms(x[s0:s1]))
    return np.array(pre_rr), [np.array(v) for v in w_rr], np.array(post_all)

print("\n=== BLOCK 1A: TRACKING & STABILITY AT -7 dB (1% Sensor Noise, With 95% Bootstrap CIs & QR-RLS) ===")
print("  Method                               | Pre (0.5-1s)        | 0-125 ms            | 125-250 ms          | 250-500 ms          | 500-1000 ms")
b2_candidates = [
    ("Float64 Standard RLS P=4 (Mem 16)",  lambda y, r: run_rls_float64(y, r, 16, 4)),
    ("Fixed-Pt Q20 Std RLS P=4 (Mem 16)",  lambda y, r: run_rls_fixed_q30(y, r, 16, 4)),
    ("Float64 Givens QR-RLS P=4 (Mem 16)", lambda y, r: run_qr_rls_float64(y, r, 16, 4)),
    ("Float64 Givens QR-RLS P=5 (Mem 32)", lambda y, r: run_qr_rls_float64(y, r, 16, 5)),
    ("Float64 Standard RLS P=5 (Mem 32)",  lambda y, r: run_rls_float64(y, r, 16, 5)),
    ("Float64 Standard RLS P=6 (Mem 64)",  lambda y, r: run_rls_float64(y, r, 16, 6)),
    ("NLMS mu=0.1",                        lambda y, r: run_nlms(y, r, 16, 0.1)),
    ("NLMS mu=0.2",                        lambda y, r: run_nlms(y, r, 16, 0.2)),
    ("NLMS mu=0.5",                        lambda y, r: run_nlms(y, r, 16, 0.5)),
    ("Int DCD-RLS Nu=1 P=4 (Mem 16)",      lambda y, r: run_dcd_rls_q30_int(y, r, 16, 4, Nu=1, Mb=16)),
    ("Int DCD-RLS Nu=4 P=4 (Mem 16)",      lambda y, r: run_dcd_rls_q30_int(y, r, 16, 4, Nu=4, Mb=16)),
    ("Int DCD-RLS Nu=4 P=5 (Mem 32)",      lambda y, r: run_dcd_rls_q30_int(y, r, 16, 5, Nu=4, Mb=16)),
    ("Int DCD-RLS Nu=4 P=6 (Mem 64)",      lambda y, r: run_dcd_rls_q30_int(y, r, 16, 6, Nu=4, Mb=16)),
]
results_b1a = {}
for name, fn in b2_candidates:
    pre_arr, w_arrs, post_arr = eval_b2_detailed(fn, snr_db=-7.0, sensor_rms_ratio=0.01)
    results_b1a[name] = (pre_arr, w_arrs, post_arr)
    m_pre, l_pre, h_pre = boot_ci(pre_arr)
    w_strs = []
    for wa in w_arrs:
        m_w, l_w, h_w = boot_ci(wa)
        w_strs.append(f"{m_w:5.3f}[{l_w:5.3f},{h_w:5.3f}]")
    print(f"  {name:36s} | {m_pre:5.3f}[{l_pre:5.3f},{h_pre:5.3f}] | " + " | ".join(w_strs), flush=True)

# Paired Wilcoxon tests on post-switch recovery
for dcd_key, ref_key in [
    ("Int DCD-RLS Nu=4 P=5 (Mem 32)", "NLMS mu=0.2"),
    ("Int DCD-RLS Nu=4 P=4 (Mem 16)", "NLMS mu=0.5"),
    ("Int DCD-RLS Nu=4 P=4 (Mem 16)", "Float64 Givens QR-RLS P=4 (Mem 16)"),
    ("Int DCD-RLS Nu=4 P=5 (Mem 32)", "Float64 Givens QR-RLS P=5 (Mem 32)"),
]:
    _, _, post_dcd = results_b1a[dcd_key]
    _, _, post_ref = results_b1a[ref_key]
    stat, p_val = wilcoxon(post_dcd, post_ref)
    diff_m, diff_lo, diff_hi = boot_ci(post_dcd - post_ref)
    print(f"  Paired Wilcoxon ({dcd_key} vs {ref_key}): Delta Post-Switch RRMSE = {diff_m:+.4f} [{diff_lo:+.4f}, {diff_hi:+.4f}], p = {p_val:.3e}")

print("\n=== BLOCK 1B: MULTI-SNR (-7, -1, +2 dB) & REALISTIC SENSOR NOISE (10% RMS = 20 dB Power SNR, 31.6% RMS = 10 dB Power SNR) ===")
for s_ratio, s_label in [(0.10, "10% Sensor Noise (20 dB Power SNR)"), (0.316, "31.6% Sensor Noise (10 dB Power SNR)")]:
    for snr_db in (-7.0, -1.0, 2.0):
        print(f"  --- {s_label} | Primary Input SNR = {snr_db:+.0f} dB (Pre | 0-125 | 125-250 | 250-500 | 500-1000 ms) ---")
        for name, fn in [
            ("NLMS mu=0.2",                   lambda y, r: run_nlms(y, r, 16, 0.2)),
            ("NLMS mu=0.5",                   lambda y, r: run_nlms(y, r, 16, 0.5)),
            ("Float64 Standard RLS P=5",      lambda y, r: run_rls_float64(y, r, 16, 5)),
            ("Float64 Givens QR-RLS P=5",     lambda y, r: run_qr_rls_float64(y, r, 16, 5)),
            ("Int DCD-RLS Nu=4 P=4 (Mem 16)", lambda y, r: run_dcd_rls_q30_int(y, r, 16, 4, Nu=4, Mb=16)),
            ("Int DCD-RLS Nu=4 P=5 (Mem 32)", lambda y, r: run_dcd_rls_q30_int(y, r, 16, 5, Nu=4, Mb=16)),
        ]:
            pre_a, w_as, _ = eval_b2_detailed(fn, snr_db=snr_db, sensor_rms_ratio=s_ratio)
            print(f"    {name:32s} | {np.mean(pre_a):.3f} | " + " | ".join(f"{np.mean(w):.3f}" for w in w_as), flush=True)

# ==============================================================================
# BLOCK 2: NON-TAUTOLOGICAL STAGE A DRIFT TEST (EXPONENTIAL STEP + BROWNIAN WALK)
# ==============================================================================
print("\n=== BLOCK 2: NON-RAMP ELECTRODE POLARIZATION RECOVERY + RANDOM-WALK DRIFT (25 x 32s Streams, No Linear Ramp) ===")
N_long = 16 * N
t_long = np.arange(N_long) / fs
sl_long = slice(int(5.0 * fs), None)
b_wander, a_wander = butter(2, 0.35 / (fs / 2.0), btype='low')

for name, fn in [
    ("Unfiltered (With Polarization + Random Walk)", lambda y: y),
    ("1st-Order DC Blocker (P=9, init y[0])",        lambda y: dc_blocker_1st(y, 9)),
    ("1st-Order DC Blocker (P=8, init y[0])",        lambda y: dc_blocker_1st(y, 8)),
    ("2-State Benedict-Bordner (alpha=0.008)",       lambda y: alpha_beta_tracker(y, 0.008, "benedict_bordner")),
    ("2-State Kalata 1984 (alpha=0.008)",            lambda y: alpha_beta_tracker(y, 0.008, "kalata")),
]:
    rr, cc = [], []
    rng_st = np.random.default_rng(404)
    for _ in range(25):
        ep_ids = rng_st.choice(len(X_held), 16, replace=False)
        x_long = np.concatenate([X_held[i] for i in ep_ids])
        # Non-ramp drift: Brownian random walk + 3 electrode motion step-exponential relaxation transients
        rw = lfilter(b_wander, a_wander, np.cumsum(rng_st.standard_normal(N_long)) / np.sqrt(fs))
        rw = 1.5 * (rw / (np.std(rw) + 1e-9))
        exp_pulses = np.zeros(N_long)
        for t0_sec in (8.0, 16.0, 24.0):
            idx0 = int(t0_sec * fs)
            tau_sec = rng_st.uniform(0.8, 2.5)
            amp_step = rng_st.uniform(2.0, 4.0) * rng_st.choice([-1.0, 1.0])
            exp_pulses[idx0:] += amp_step * np.exp(-(t_long[idx0:] - t0_sec) / tau_sec)
        drift = rw + exp_pulses
        xh = fn(x_long + drift)
        rr.append(rms(xh[sl_long] - x_long[sl_long]) / rms(x_long[sl_long]))
        cc.append(np.corrcoef(xh[sl_long], x_long[sl_long])[0, 1])
    m_r, lo_r, hi_r = boot_ci(rr)
    print(f"  {name:44s} | RRMSE = {m_r:.3f} [{lo_r:.3f}, {hi_r:.3f}] | CC = {np.mean(cc):.3f}")

# ==============================================================================
# BLOCK 3: REAL PHYSIONET WFDB LOADER, CHANNEL DETECTION & ARTIFACT SEGMENTATION
# ==============================================================================
if d_phys is not None:
    try:
        import wfdb
        print(f"\n=== BLOCK 3: REAL PHYSIONET MOTION-ARTIFACT WFDB BENCHMARK (Decimated 8x from 2048 Hz -> 256 Hz) ===")
        rec_names = [f"eeg{i}" for i in range(1, 24)]
        active_rrmse = {k: [] for k in ("unfilt", "nlms", "rls_p5", "qr_p5", "dcd_p5_dc", "dcd_p5_ab")}
        rest_rrmse   = {k: [] for k in ("unfilt", "nlms", "rls_p5", "qr_p5", "dcd_p5_dc", "dcd_p5_ab")}
        lags_ms = []
        n_loaded = 0

        for rname in rec_names:
            try:
                if os.path.isdir(d_phys):
                    rec_path = os.path.join(d_phys, rname)
                    if not os.path.exists(rec_path + ".hea"):
                        continue
                    rec = wfdb.rdrecord(rec_path)
                elif d_phys == "physionet_remote":
                    rec = wfdb.rdrecord(rname, pn_dir="motion-artifact")
                else:
                    continue
            except Exception:
                continue

            sig = rec.p_signal
            if sig is None or sig.shape[1] < 5:
                continue
            n_loaded += 1
            # Decimate all 2048 Hz aligned WFDB channels by 8x to fs_p = 256 Hz
            fs_p = 256.0
            ch0 = resample_poly(sig[:, 0], 1, 8)
            ch1 = resample_poly(sig[:, 1], 1, 8)
            # Extract 3-axis accelerometer channels (columns 2..4 or 3..5 depending on trigger presence in WFDB)
            acc_start = 3 if sig.shape[1] >= 6 else 2
            acc_3d = np.column_stack([
                dc_blocker_1st(resample_poly(sig[:, acc_start + j], 1, 8), 8)
                for j in range(min(3, sig.shape[1] - acc_start))
            ])
            acc_env = np.linalg.norm(acc_3d, axis=1)
            # Sliding 0.5-s (128-sample) RMS of accelerometer dynamic magnitude to segment active motion vs rest
            win_len = 128
            acc_rms = np.sqrt(np.convolve(acc_env**2, np.ones(win_len)/win_len, mode='same'))
            med_acc = np.median(acc_rms)
            active_mask = acc_rms > (2.5 * med_acc)
            rest_mask   = (acc_rms <= (1.2 * med_acc))
            active_mask[:256] = False; rest_mask[:256] = False
            if np.sum(active_mask) < 128 or np.sum(rest_mask) < 128:
                continue

            # Identify which EEG channel was manipulated (higher variance ratio during active motion vs rest)
            ch0_c = dc_blocker_1st(ch0, 9); ch1_c = dc_blocker_1st(ch1, 9)
            ratio0 = np.var(ch0_c[active_mask]) / (np.var(ch0_c[rest_mask]) + 1e-9)
            ratio1 = np.var(ch1_c[active_mask]) / (np.var(ch1_c[rest_mask]) + 1e-9)
            if ratio1 >= ratio0:
                eeg_clean_raw, eeg_noisy_raw, noisy_id = ch0, ch1, "EEG2"
            else:
                eeg_clean_raw, eeg_noisy_raw, noisy_id = ch1, ch0, "EEG1"

            # Stage A pre-filtering variants: 1st-order DC blocker vs 2-state Alpha-Beta
            clean_dc = dc_blocker_1st(eeg_clean_raw, 9)
            noisy_dc = dc_blocker_1st(eeg_noisy_raw, 9)
            clean_ab = alpha_beta_tracker(eeg_clean_raw, 0.008, "benedict_bordner")
            noisy_ab = alpha_beta_tracker(eeg_noisy_raw, 0.008, "benedict_bordner")

            # Empirical cross-correlation lag between dominant accelerometer axis and artifact (noisy_dc - clean_dc)
            art = noisy_dc - clean_dc
            xc = np.correlate(art[active_mask] - np.mean(art[active_mask]),
                              acc_env[active_mask] - np.mean(acc_env[active_mask]), mode='full')
            mid = np.sum(active_mask) - 1
            peak_lag = int(np.argmax(np.abs(xc[mid:min(mid + 64, len(xc))])))
            lags_ms.append(peak_lag * (1000.0 / fs_p))

            # Run Stage B filters (L = 12 taps per accelerometer axis = 36 total taps across 3 axes)
            L_ax = 12
            out_nlms   = run_nlms(noisy_dc, acc_3d, L=L_ax, mu=0.1)
            out_rls    = run_rls_float64(noisy_dc, acc_3d, L=L_ax, P_shift=6)
            out_qr     = run_qr_rls_float64(noisy_dc, acc_3d, L=L_ax, P_shift=6)
            out_dcd_dc = run_dcd_rls_q30_int(noisy_dc, acc_3d, L=L_ax, P_shift=6, Nu=4, Mb=16)
            out_dcd_ab = run_dcd_rls_q30_int(noisy_ab, acc_3d, L=L_ax, P_shift=6, Nu=4, Mb=16)

            for key, sig_out, ref_clean in [
                ("unfilt",    noisy_dc,   clean_dc),
                ("nlms",      out_nlms,   clean_dc),
                ("rls_p5",    out_rls,    clean_dc),
                ("qr_p5",     out_qr,     clean_dc),
                ("dcd_p5_dc", out_dcd_dc, clean_dc),
                ("dcd_p5_ab", out_dcd_ab, clean_ab),
            ]:
                active_rrmse[key].append(rms(sig_out[active_mask] - ref_clean[active_mask]) / rms(ref_clean[active_mask]))
                rest_rrmse[key].append(rms(sig_out[rest_mask] - ref_clean[rest_mask]) / rms(ref_clean[rest_mask]))

            print(f"  Trial {rname:5s}: Disturbed={noisy_id} (VarRatio={max(ratio0,ratio1):5.1f}x) | "
                  f"Active RRMSE: Unfilt={active_rrmse['unfilt'][-1]:.3f}, NLMS={active_rrmse['nlms'][-1]:.3f}, "
                  f"QR-RLS={active_rrmse['qr_p5'][-1]:.3f}, DCD(DC)={active_rrmse['dcd_p5_dc'][-1]:.3f}, DCD(AB)={active_rrmse['dcd_p5_ab'][-1]:.3f}")

        if n_loaded > 0:
            print(f"\n  --- PHYSIONET SUMMARY ({n_loaded} Trials, 3-Axis Acc Reference, L={L_ax} taps/axis @ 256 Hz) ---")
            print(f"  Empirical Peak Coupling Lag: {np.mean(lags_ms):.1f} +/- {np.std(lags_ms):.1f} ms")
            for k_label, k_id in [
                ("Unfiltered (DC-Blocked)",        "unfilt"),
                ("3-Axis NLMS (mu=0.1)",           "nlms"),
                ("3-Axis Float64 RLS (P=6)",       "rls_p5"),
                ("3-Axis Float64 QR-RLS (P=6)",    "qr_p5"),
                ("3-Axis Int DCD-RLS + DC Block",  "dcd_p5_dc"),
                ("3-Axis Int DCD-RLS + 2-State AB","dcd_p5_ab"),
            ]:
                m_a, lo_a, hi_a = boot_ci(active_rrmse[k_id])
                m_r, lo_r, hi_r = boot_ci(rest_rrmse[k_id])
                print(f"  {k_label:34s} | Active Motion RRMSE: {m_a:.3f} [{lo_a:.3f},{hi_a:.3f}] | Rest Distortion RRMSE: {m_r:.3f} [{lo_r:.3f},{hi_r:.3f}]")
    except ImportError:
        print("\n[NOTE] Python package `wfdb` is not installed. Run `pip install wfdb` to execute Block 3 on PhysioNet records.")
