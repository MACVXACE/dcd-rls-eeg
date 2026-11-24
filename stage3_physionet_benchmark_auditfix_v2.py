"""
stage3_physionet_benchmark.py
Implements Part B and Part C on the 23 PhysioNet 'motion-artifact' EEG WFDB records,
strictly following INSPECT FINDINGS F1-F8.

Usage:
  python3 stage3_physionet_benchmark.py <wfdb_dir_or_physionet_remote> [--gamma-p5 0.125] [--quick]
"""
import sys
import os
import time
import datetime
import platform
import numpy as np
import scipy
from scipy.signal import resample_poly, lfilter, butter, sosfiltfilt, freqz
from scipy.stats import wilcoxon

try:
    import wfdb
except ImportError as exc:
    raise RuntimeError("The 'wfdb' package is required. Install via: pip install wfdb") from exc

if len(sys.argv) < 2:
    raise RuntimeError(
        "Usage: python3 stage3_physionet_benchmark.py <wfdb_dir_or_physionet_remote> [--gamma-p5 0.125] [--quick]"
    )

wfdb_src = sys.argv[1]
is_remote = (wfdb_src == "physionet_remote")
if (not is_remote) and (not os.path.isdir(wfdb_src)):
    raise FileNotFoundError(f"WFDB directory not found: {wfdb_src}")

quick_mode = ("--quick" in sys.argv[2:])
frozen_gamma_p5 = 0.125  # AUDIT FIX: FIXED M2 gamma for P=5 printed by stage2_round6_ridge_benchmark.py
HEADROOM = 16.0          # AUDIT FIX C: 4 bits of headroom above the clean-segment 99.9th percentile (artifacts are larger)
CAL_START = 1024         # AUDIT FIX B: skip the first 2 s at 512 Hz when calibrating scales
if "--gamma-p5" in sys.argv:
    g_idx = sys.argv.index("--gamma-p5")
    if g_idx + 1 >= len(sys.argv):
        raise ValueError("Missing value after --gamma-p5")
    frozen_gamma_p5 = float(sys.argv[g_idx + 1])

REQUIRED_CHANNELS = [
    "EEG1", "EEG2", "EEG_tr",
    "ACC_x1", "ACC_y1", "ACC_z1",
    "ACC_x2", "ACC_y2", "ACC_z2", "ACC_tr"
]

print("=== STAGE 3 PHYSIONET BENCHMARK HEADER ===")
if quick_mode:
    print("QUICK MODE - NOT FOR REPORTING")
print(f"Start Time (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
print(f"Python: {platform.python_version()} | NumPy: {np.__version__} | SciPy: {scipy.__version__}")
print(f"Frozen Part-A Parameters: fs=512 Hz, P=5 (62.5 ms), L_ax=16, Nu=4, Mb=16, H_int=16, gamma={frozen_gamma_p5:g}")

# ==============================================================================
# 1. STAGE A HIGH-PASS FILTER (SPECIFIED BY CORNER IN HZ) & RESPONSE TABLE
# ==============================================================================
def hp_pole_shift(fs_hz, fc_hz):
    return int(round(np.log2(fs_hz / (2.0 * np.pi * fc_hz))))


def apply_dc_blocker_hz(sig, fs_hz, fc_hz):
    P_hp = hp_pole_shift(fs_hz, fc_hz)
    a = 1.0 - 2.0**(-P_hp)
    y = np.asarray(sig, dtype=np.float64)
    out = np.zeros_like(y)
    x_prev = float(y[0])
    y_prev = 0.0
    for k in range(len(y)):
        out[k] = y[k] - x_prev + a * y_prev
        x_prev = float(y[k])
        y_prev = float(out[k])
    return out, P_hp


def print_stage_a_response(fs_hz=512.0):
    print("\n=== STAGE A HIGH-PASS FREQUENCY RESPONSE AT fs = 512 Hz ===")
    for fc in (0.16, 0.32, 0.64):
        P_hp = hp_pole_shift(fs_hz, fc)
        a = 1.0 - 2.0**(-P_hp)
        w_eval = 2.0 * np.pi * np.array([0.5, 1.0, 2.0]) / fs_hz
        _, h_val = freqz([1.0, -1.0], [1.0, -a], worN=w_eval)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(h_val), 1e-15))
        phs_deg = np.angle(h_val, deg=True)
        print(
            f"  Target fc={fc:4.2f} Hz -> P_hp={P_hp} (a=1-2^-{P_hp}) | "
            f"0.5 Hz: {mag_db[0]:+6.2f} dB, {phs_deg[0]:+6.1f} deg | "
            f"1.0 Hz: {mag_db[1]:+6.2f} dB, {phs_deg[1]:+6.1f} deg | "
            f"2.0 Hz: {mag_db[2]:+6.2f} dB, {phs_deg[2]:+6.1f} deg",
            flush=True,
        )


print_stage_a_response(512.0)

# ==============================================================================
# 2. LOAD ALL 23 RECORDS, VERIFY F1-F4 INTERVALS & AUDIT eeg_20 (F5)
# ==============================================================================
def load_wfdb_record(rec_name):
    if is_remote:
        rec = wfdb.rdrecord(rec_name, pn_dir="motion-artifact")
        ann = wfdb.rdann(rec_name, "trigger", pn_dir="motion-artifact")
    else:
        base = os.path.join(wfdb_src, rec_name)
        rec = wfdb.rdrecord(base)
        ann = wfdb.rdann(base, "trigger")
    return rec, ann


def parse_f4_intervals(ann, n_samples, fs_in):
    """
    Implements F4 rule exactly:
    'an R before any F is ignored; an F with no later R runs to the end of the record.'
    Returns mask (bool array at fs_in), list of (f_sample, r_sample, dur_sec), count_F, count_R.
    """
    syms = list(ann.symbol)
    samps = list(ann.sample)
    count_F = sum(1 for s in syms if s == "F")
    count_R = sum(1 for s in syms if s == "R")
    mask = np.zeros(n_samples, dtype=bool)
    intervals = []
    start_f = None
    for s_idx, sym in zip(samps, syms):
        s_int = int(s_idx)
        if sym == "F":
            start_f = s_int
        elif sym == "R" and start_f is not None:
            end_r = min(s_int, n_samples)
            mask[start_f:end_r] = True
            intervals.append((start_f, end_r, (end_r - start_f) / float(fs_in)))
            start_f = None
    if start_f is not None:
        mask[start_f:n_samples] = True
        intervals.append((start_f, n_samples, (n_samples - start_f) / float(fs_in)))
    return mask, intervals, count_F, count_R


def leaky_integrate_axis(acc_col, P_int=6):
    a_int = 1.0 - 2.0**(-P_int)
    vel = lfilter([1.0], [1.0, -a_int], acc_col)
    vel -= np.mean(vel)
    pos = lfilter([1.0], [1.0, -a_int], vel)
    pos -= np.mean(pos)
    return vel, pos


print("\n=== LOADING 23 WFDB RECORDS & VERIFYING F1-F5 SELF-CHECKS ===", flush=True)
records_data = []
record_ids = [f"eeg_{i}" for i in (range(1, 5) if quick_mode else range(1, 24))]

for rname in record_ids:
    try:
        rec, ann = load_wfdb_record(rname)
    except Exception as exc:
        raise RuntimeError(f"Failed to load required record '{rname}': {exc}") from exc

    if float(rec.fs) != 2048.0:
        raise RuntimeError(f"Record {rname} has fs={rec.fs}, expected 2048.0 Hz")

    sig_map = {nm: idx for idx, nm in enumerate(rec.sig_name)}
    for req_nm in REQUIRED_CHANNELS:
        if req_nm not in sig_map:
            raise RuntimeError(f"Record {rname} missing required channel '{req_nm}'. Found: {rec.sig_name}")

    raw_mat = rec.p_signal
    n_2048 = raw_mat.shape[0]
    mask_2048, ivals_2048, nF, nR = parse_f4_intervals(ann, n_2048, 2048.0)
    tot_art_sec = sum(iv[2] for iv in ivals_2048)

    ival_str = ", ".join(f"[{f_s}-> {r_s}: {d_s:.2f}s]" for (f_s, r_s, d_s) in ivals_2048)
    print(
        f"  {rname:6s} | dur={n_2048/2048.0:6.1f}s | F={nF}, R={nR} | "
        f"Total Artifact={tot_art_sec:6.2f}s | Intervals: {ival_str}",
        flush=True,
    )

    if nF != 4 or abs(tot_art_sec - 240.0) > 1.0:
        raise RuntimeError(
            f"F4 Check Failed on {rname}: F_count={nF} (expected 4), total_artifact={tot_art_sec:.2f}s (expected 240.0 +/- 1.0s)"
        )

    # F5 Diagnostic Audit on eeg_20 (and general channel health check)
    if rname == "eeg_20":
        ax1 = raw_mat[:, sig_map["ACC_x1"]]
        ay1 = raw_mat[:, sig_map["ACC_y1"]]
        az1 = raw_mat[:, sig_map["ACC_z1"]]
        zero_diff_frac = float(np.mean(np.diff(ax1) == 0.0))
        min_ax1, max_ax1 = float(np.min(ax1)), float(np.max(ax1))
        at_rail_frac = float(np.mean((ax1 <= min_ax1 + 1e-9) | (ax1 >= max_ax1 - 1e-9)))
        sos_hp05 = butter(2, 0.5 / 1024.0, btype="high", output="sos")
        hp_ax1 = sosfiltfilt(sos_hp05, ax1 - np.mean(ax1))
        ratio_ax1 = float(np.sqrt(np.mean(hp_ax1[mask_2048]**2)) / max(np.sqrt(np.mean(hp_ax1[~mask_2048]**2)), 1e-12))
        print(
            f"  [F5 AUDIT eeg_20] ACC_x1 range=[{min_ax1:.4f}, {max_ax1:.4f}], "
            f"zero_diff_frac={zero_diff_frac*100:.2f}%, rail_frac={at_rail_frac*100:.3f}%, "
            f"art/clean_RMS_ratio={ratio_ax1:.2f}x (ACC_y1/z1 active; eeg_20 retained in 23-record set and also reported in 22-record sensitivity set).",
            flush=True,
        )

    # Decimate 2048 Hz -> 512 Hz via Kaiser-windowed FIR polyphase filter (factor 4)
    fs_512 = 512.0
    eeg1_512 = resample_poly(raw_mat[:, sig_map["EEG1"]], 1, 4, padtype="line")
    eeg2_512 = resample_poly(raw_mat[:, sig_map["EEG2"]], 1, 4, padtype="line")
    n_512 = len(eeg1_512)

    # Map F->R intervals to 512 Hz sample indices
    mask_512 = np.zeros(n_512, dtype=bool)
    first_f_512 = n_512
    for (f_s, r_s, _) in ivals_2048:
        f_512 = max(0, min(int(round(f_s / 4.0)), n_512))
        r_512 = max(0, min(int(round(r_s / 4.0)), n_512))
        mask_512[f_512:r_512] = True
        if f_512 < first_f_512:
            first_f_512 = f_512

    # Decimate accelerometers and high-pass at 0.5 Hz (P_hp=7 at 512 Hz) to remove static gravity offset
    acc1_cols, acc2_cols = [], []
    for ax_nm in ("x", "y", "z"):
        c1_raw = resample_poly(raw_mat[:, sig_map[f"ACC_{ax_nm}1"]], 1, 4, padtype="line")
        c2_raw = resample_poly(raw_mat[:, sig_map[f"ACC_{ax_nm}2"]], 1, 4, padtype="line")
        c1_hp, _ = apply_dc_blocker_hz(c1_raw, fs_512, 0.64)
        c2_hp, _ = apply_dc_blocker_hz(c2_raw, fs_512, 0.64)
        acc1_cols.append(c1_hp)
        acc2_cols.append(c2_hp)

    ref_a = np.column_stack(acc1_cols)  # (a) Disturbed electrode ACC_*1 (3 axes)
    ref_b = np.column_stack([acc1_cols[j] - acc2_cols[j] for j in range(3)])  # (b) Differential ACC_*1 - ACC_*2
    vel_cols, pos_cols = [], []
    for j in range(3):
        v_j, p_j = leaky_integrate_axis(ref_b[:, j], P_int=6)
        vel_cols.append(v_j)
        pos_cols.append(p_j)
    ref_c = np.column_stack([ref_b, np.column_stack(vel_cols), np.column_stack(pos_cols)])  # (c) 9 axes

    # Active-motion sub-window mask (F6): 0.5-s (256-sample) sliding RMS of ||ACC_*1||_2 > 1.5x clean median
    acc1_mag = np.linalg.norm(ref_a, axis=1)
    win_half_sec = 256
    sliding_rms = np.sqrt(np.convolve(acc1_mag**2, np.ones(win_half_sec) / float(win_half_sec), mode="same"))
    clean_med_rms = float(np.median(sliding_rms[~mask_512]))
    active_sub_mask = mask_512 & (sliding_rms > (1.5 * clean_med_rms))

    records_data.append(
        {
            "name": rname,
            "eeg1_512": eeg1_512,
            "eeg2_512": eeg2_512,
            "mask_art": mask_512,
            "mask_clean": ~mask_512,
            "mask_active_sub": active_sub_mask,
            "first_f_512": max(first_f_512, 512),
            "refs": {"RefA_ACC1_3ax": ref_a, "RefB_Diff_3ax": ref_b, "RefC_DiffVelPos_9ax": ref_c},
        }
    )

# ==============================================================================
# 3. MULTI-CHANNEL BLOCK-SHIFT DCD-RLS SOLVER & BIT-IDENTITY SELF-CHECK
# ==============================================================================
def build_multichan_taps(ref_mat, L_ax):
    ref = np.asarray(ref_mat, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    N_s, M_ch = ref.shape
    L_tot = M_ch * L_ax
    U_all = np.zeros((N_s, L_tot), dtype=np.float64)
    buf = np.zeros((M_ch, L_ax), dtype=np.float64)
    for k in range(N_s):
        buf[:, 1:] = buf[:, :-1]
        buf[:, 0] = ref[k, :]
        U_all[k, :] = buf.ravel()
    return U_all, M_ch


def run_multichan_dcd_ridge_int(
    d_sig,
    ref_mat,
    L_ax=16,
    P_shift=5,
    gamma=0.25,
    Nu=4,
    Mb=16,
    H_int=16,
    cal_len=2048,
    use_block_shift=True,
):
    """
    Multi-channel integer Q15/Q30 ridge DCD-RLS with causal calibration scaling (from [:cal_len])
    and M x M block-Toeplitz time-shift update of R_q30 (costing M * L_tot multiplies/sample).
    """
    Q15 = 1 << 15
    Q30 = 1 << 30
    if cal_len - CAL_START < 512:
        raise RuntimeError(f"calibration window too short: [{CAL_START}, {cal_len})")
    sc_d = max(float(np.percentile(np.abs(d_sig[CAL_START:cal_len]), 99.9)) * 1.05 * HEADROOM, 1e-9)
    raw_d = np.round((d_sig / sc_d) * Q15)
    clip_d = (raw_d < -Q15) | (raw_d >= Q15)
    d_q15 = np.clip(raw_d, -Q15, Q15 - 1).astype(np.int64)

    ref = np.asarray(ref_mat, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    N_s, M_ch = ref.shape
    r_q15_cols = []
    clip_r_any = np.zeros(N_s, dtype=bool)
    for m_i in range(M_ch):
        sc_rm = max(float(np.percentile(np.abs(ref[CAL_START:cal_len, m_i]), 99.9)) * 1.05 * HEADROOM, 1e-9)
        raw_rm = np.round((ref[:, m_i] / sc_rm) * Q15)
        clip_r_any |= (raw_rm < -Q15) | (raw_rm >= Q15)
        r_q15_cols.append(np.clip(raw_rm, -Q15, Q15 - 1).astype(np.int64))
    r_q15 = np.column_stack(r_q15_cols)

    U_all, _ = build_multichan_taps(r_q15, L_ax)
    U_all = U_all.astype(np.int64)
    L_tot = M_ch * L_ax

    var_u = float(np.var(r_q15[CAL_START:cal_len] / float(Q15)))
    reg_q30 = max(int(round(var_u * Q30)), 1)
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0 (R starts at zero; the persistent load delta keeps it solvable)")
    R_q30 = np.zeros((L_tot, L_tot), dtype=np.int64)  # AUDIT FIX A
    r_q30 = np.zeros(L_tot, dtype=np.int64)
    w_q15 = np.zeros(L_tot, dtype=np.int64)
    out = np.zeros(N_s, dtype=np.float64)

    delta_q30 = max(int(round(float(gamma) * var_u * (1 << P_shift) * Q30)), 1) if gamma > 0.0 else 0

    # Precompute block-shift indices for M_ch blocks of length L_ax
    lead_idx = np.array([m_i * L_ax for m_i in range(M_ch)], dtype=np.int64)
    src_mask = np.ones(L_tot, dtype=bool)
    dst_mask = np.ones(L_tot, dtype=bool)
    for m_i in range(M_ch):
        src_mask[(m_i + 1) * L_ax - 1] = False
        dst_mask[m_i * L_ax] = False

    for k in range(N_s):
        u_q15 = U_all[k]
        y_hat_q15 = int(np.dot(w_q15, u_q15) >> 15)
        e_q15 = int(d_q15[k] - y_hat_q15)
        out[k] = (e_q15 / float(Q15)) * sc_d

        if use_block_shift:
            # Save old leading rows/cols before shifting sub-blocks
            R_lead_old = R_q30[:, lead_idx].copy()
            R_q30[np.ix_(dst_mask, dst_mask)] = R_q30[np.ix_(src_mask, src_mask)]
            new_cols = R_lead_old - (R_lead_old >> P_shift) + np.outer(u_q15, u_q15[lead_idx])
            R_q30[:, lead_idx] = new_cols
            R_q30[lead_idx, :] = new_cols.T
        else:
            R_q30 = R_q30 - (R_q30 >> P_shift) + np.outer(u_q15, u_q15)

        if delta_q30 == 0:
            r_q30 = r_q30 - (r_q30 >> P_shift) + e_q15 * u_q15
        else:
            leak_vec = ((delta_q30 >> P_shift) * w_q15) >> 15
            r_q30 = r_q30 - (r_q30 >> P_shift) + e_q15 * u_q15 - leak_vec

        m = 1
        u_cnt = 0
        while m <= Mb and u_cnt < Nu:
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

    return out, clip_d | clip_r_any


def run_multichan_nlms(d_sig, ref_mat, L_ax=16, mu=0.1, cal_len=2048):
    ref = np.asarray(ref_mat, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref[:, None]
    N_s, M_ch = ref.shape
    ref_norm = np.zeros_like(ref)
    for m_i in range(M_ch):
        sc_m = max(float(np.std(ref[CAL_START:cal_len, m_i])), 1e-9)
        ref_norm[:, m_i] = ref[:, m_i] / sc_m
    U_all, _ = build_multichan_taps(ref_norm, L_ax)
    L_tot = U_all.shape[1]
    w = np.zeros(L_tot, dtype=np.float64)
    out = np.zeros(N_s, dtype=np.float64)
    for k in range(N_s):
        u = U_all[k]
        e = d_sig[k] - np.dot(w, u)
        out[k] = e
        w += (mu / (np.dot(u, u) + 1e-4)) * e * u
    return out


# Self-check: Verify per-block shift update of R_q30 is bit-identical to full outer-product update
rec0 = records_data[0]
eeg2_0_hp, _ = apply_dc_blocker_hz(rec0["eeg2_512"][:2048], 512.0, 0.64)
out_blk, _ = run_multichan_dcd_ridge_int(
    eeg2_0_hp, rec0["refs"]["RefA_ACC1_3ax"][:2048], L_ax=16, P_shift=5, gamma=frozen_gamma_p5, use_block_shift=True
)
out_full, _ = run_multichan_dcd_ridge_int(
    eeg2_0_hp, rec0["refs"]["RefA_ACC1_3ax"][:2048], L_ax=16, P_shift=5, gamma=frozen_gamma_p5, use_block_shift=False
)
if np.array_equal(out_blk, out_full):
    print("  [SELF-CHECK] Multi-channel block-shift R_q30 update bit-identical to full outer-product: PASS", flush=True)
else:
    raise RuntimeError("Multi-channel block-shift R_q30 update did NOT match full outer-product update!")


# ==============================================================================
# 4. SWEENEY ET AL. (2012) METRICS & EVALUATION LOOP
# ==============================================================================
def sweeney_metrics(clean_sig, noisy_sig, est_sig, mask_art, mask_clean):
    x_art = clean_sig[mask_art]
    n_art = noisy_sig[mask_art]
    xh_art = est_sig[mask_art]

    snr_in = 10.0 * np.log10(np.var(x_art) / max(np.var(n_art - x_art), 1e-15))
    snr_out = 10.0 * np.log10(np.var(x_art) / max(np.var(xh_art - x_art), 1e-15))
    delta_snr = snr_out - snr_in

    r_clean = float(np.corrcoef(clean_sig[mask_clean], noisy_sig[mask_clean])[0, 1])
    r_before = float(np.corrcoef(x_art, n_art)[0, 1])
    r_after = float(np.corrcoef(x_art, xh_art)[0, 1])
    denom = max(abs(r_clean - r_before), 1e-9)
    corr_imp_pct = ((r_after - r_before) / denom) * 100.0

    # Clean-interval preservation metric (always-on filter distortion on clean epochs)
    x_cl = clean_sig[mask_clean]
    xh_cl = est_sig[mask_clean]
    clean_rrmse = float(np.sqrt(np.mean((xh_cl - x_cl) ** 2)) / max(np.sqrt(np.mean(x_cl ** 2)), 1e-15))
    return float(delta_snr), float(corr_imp_pct), float(r_after), clean_rrmse


# ==============================================================================
# AUDIT FIX D-H: same-reference comparisons, unprocessed baseline, whole-record and gated scoring, one tuned
# parameter per method chosen leave-one-record-out with the same score (NLMS: mu; DCD: memory P), and
# pre-registered primary tests. Written by the auditor before any PhysioNet result existed.
# LORO score for both methods = dSNR(artifact blocks) + dSNR(clean epochs)  (rewards removal, penalises damage)
# ==============================================================================
def safe_metrics(clean_sig, noisy_sig, est_sig, mask_eval, mask_clean):
    if int(np.sum(mask_eval)) < 512 or int(np.sum(mask_clean)) < 512:
        return (float("nan"),) * 4
    return sweeney_metrics(clean_sig, noisy_sig, est_sig, mask_eval, mask_clean)


def loro_score(c_hp, n_hp, e, rec_d):
    return (safe_metrics(c_hp, n_hp, e, rec_d["mask_art"], rec_d["mask_clean"])[0]
            + safe_metrics(c_hp, n_hp, e, rec_d["mask_clean"], rec_d["mask_clean"])[0])


def loro_pick(grid, scores, n_rec):
    return [max(grid, key=lambda g: np.nanmean([scores[g][j] for j in range(n_rec) if j != i])) for i in range(n_rec)]


def boot_median_ci(v, n_boot=10000, seed=42):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    rng_b = np.random.default_rng(seed)
    meds = np.median(v[rng_b.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


t_eval0 = time.time()
n_rec = len(records_data)
hp_cache = []
for rec_d in records_data:
    c_hp, _ = apply_dc_blocker_hz(rec_d["eeg1_512"], 512.0, 0.64)
    n_hp, _ = apply_dc_blocker_hz(rec_d["eeg2_512"], 512.0, 0.64)
    hp_cache.append((c_hp, n_hp))

mu_candidates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
NLMS_REFS = ("RefA_ACC1_3ax", "RefC_DiffVelPos_9ax")
P_GRID_DCD = (5, 7, 9, 11)  # AUDIT: extended by the pre-declared grid-edge rule (R4) after the quick run picked P=9 (grid top) on 4 of 4 records
nlms_sc = {rk: {m: [] for m in mu_candidates} for rk in NLMS_REFS}
dcd_a_est = {Pg: [] for Pg in P_GRID_DCD}
dcd_a_sc = {Pg: [] for Pg in P_GRID_DCD}
clip_store = []
print("\n=== TUNING GRIDS (scored per record; selection is leave-one-record-out) ===", flush=True)
for idx_rec, rec_d in enumerate(records_data):
    c_hp, n_hp = hp_cache[idx_rec]; cal_n = rec_d["first_f_512"]
    for rk in NLMS_REFS:
        for m in mu_candidates:
            e = run_multichan_nlms(n_hp, rec_d["refs"][rk], L_ax=16, mu=m, cal_len=cal_n)
            nlms_sc[rk][m].append(loro_score(c_hp, n_hp, e, rec_d))
    for Pg in P_GRID_DCD:
        e, clip = run_multichan_dcd_ridge_int(n_hp, rec_d["refs"]["RefA_ACC1_3ax"], 16, Pg, gamma=frozen_gamma_p5, cal_len=cal_n)
        dcd_a_est[Pg].append(e); dcd_a_sc[Pg].append(loro_score(c_hp, n_hp, e, rec_d))
        if Pg == P_GRID_DCD[0]:
            clip_store.append((100.0 * float(np.mean(clip[rec_d["mask_clean"]])), 100.0 * float(np.mean(clip[rec_d["mask_art"]]))))
    print(f"  {rec_d['name']}: grids done, input clipping clean/artifact = {clip_store[-1][0]:.3f}/{clip_store[-1][1]:.3f} % "
          f"({time.time() - t_eval0:.0f} s)", flush=True)
loro_mu = {rk: loro_pick(mu_candidates, nlms_sc[rk], n_rec) for rk in NLMS_REFS}
loro_P = loro_pick(P_GRID_DCD, dcd_a_sc, n_rec)
for rk in NLMS_REFS:
    print(f"  LORO mu, NLMS on {rk}: " + ", ".join(f"{m:g}" for m in loro_mu[rk]), flush=True)
print("  LORO P, DCD (selected on RefA, used for RefA/RefB/RefC): " + ", ".join(str(p) for p in loro_P), flush=True)

METHODS = ("NONE", "DCD_RefA", "DCD_RefB", "DCD_RefC", "NLMS_RefA", "NLMS_RefC", "DCD_RefA_P5", "DCD_RefA_P7", "DCD_RefA_P9", "DCD_RefA_P11")
KEYS = ("dsnr_block", "ci_block", "dsnr_active", "dsnr_clean", "clean_rrmse", "dsnr_record_always_on", "dsnr_record_gated")
store = {m: {k: [] for k in KEYS} for m in METHODS}
print("\n=== PER-RECORD RESULTS (fc = 0.64 Hz; dSNR in dB over the 60-s blocks; NONE = EEG2 unprocessed) ===", flush=True)
print("  Record | P | " + " | ".join(f"{m}" for m in METHODS[1:]) + " | clean RRMSE NONE/DCD_A/NLMS_A", flush=True)
for idx_rec, rec_d in enumerate(records_data):
    c_hp, n_hp = hp_cache[idx_rec]; cal_n = rec_d["first_f_512"]; Psel = loro_P[idx_rec]
    m_art = rec_d["mask_art"]; m_cl = rec_d["mask_clean"]; m_sub = rec_d["mask_active_sub"]
    all_true = np.ones(len(n_hp), dtype=bool)
    ests = {"NONE": n_hp, "DCD_RefA": dcd_a_est[Psel][idx_rec],
            "NLMS_RefA": run_multichan_nlms(n_hp, rec_d["refs"]["RefA_ACC1_3ax"], 16, mu=loro_mu["RefA_ACC1_3ax"][idx_rec], cal_len=cal_n),
            "NLMS_RefC": run_multichan_nlms(n_hp, rec_d["refs"]["RefC_DiffVelPos_9ax"], 16, mu=loro_mu["RefC_DiffVelPos_9ax"][idx_rec], cal_len=cal_n)}
    ests["DCD_RefB"], _ = run_multichan_dcd_ridge_int(n_hp, rec_d["refs"]["RefB_Diff_3ax"], 16, Psel, gamma=frozen_gamma_p5, cal_len=cal_n)
    ests["DCD_RefC"], _ = run_multichan_dcd_ridge_int(n_hp, rec_d["refs"]["RefC_DiffVelPos_9ax"], 16, Psel, gamma=frozen_gamma_p5, cal_len=cal_n)
    for Pg in P_GRID_DCD:
        ests[f"DCD_RefA_P{Pg}"] = dcd_a_est[Pg][idx_rec]
    for m in METHODS:
        e = ests[m]
        d_b, ci_b, _, cl_rr = safe_metrics(c_hp, n_hp, e, m_art, m_cl)
        store[m]["dsnr_block"].append(d_b); store[m]["ci_block"].append(ci_b); store[m]["clean_rrmse"].append(cl_rr)
        store[m]["dsnr_active"].append(safe_metrics(c_hp, n_hp, e, m_sub, m_cl)[0])
        store[m]["dsnr_clean"].append(safe_metrics(c_hp, n_hp, e, m_cl, m_cl)[0])
        store[m]["dsnr_record_always_on"].append(safe_metrics(c_hp, n_hp, e, all_true, m_cl)[0])
        store[m]["dsnr_record_gated"].append(safe_metrics(c_hp, n_hp, np.where(m_art, e, n_hp), all_true, m_cl)[0])
    print(f"  {rec_d['name']:6s} | {Psel} | " + " | ".join(f"{store[m]['dsnr_block'][-1]:+6.2f}" for m in METHODS[1:])
          + f" | {store['NONE']['clean_rrmse'][-1]:.3f}/{store['DCD_RefA']['clean_rrmse'][-1]:.3f}/{store['NLMS_RefA']['clean_rrmse'][-1]:.3f}"
          + f" ({time.time() - t_eval0:.0f} s)", flush=True)
for Pg in P_GRID_DCD:
    dcd_a_est[Pg] = None  # free memory

for subset_label, exclude_eeg20 in [("ALL RECORDS", False), ("EXCLUDING eeg_20 (F5 sensitivity)", True)]:
    keep = [i for i, r in enumerate(records_data) if not (exclude_eeg20 and r["name"] == "eeg_20")]
    if len(keep) < 2:
        continue
    print(f"\n=== AGGREGATE (N = {len(keep)}): {subset_label} - medians over records ===", flush=True)
    print("  Method       | dSNR blocks | CorrImp % | dSNR active | dSNR clean | dSNR record always-on | dSNR record gated | clean RRMSE (minus NONE)")
    cl_none = np.array([store["NONE"]["clean_rrmse"][i] for i in keep])
    for m in METHODS:
        g = lambda k: np.array([store[m][k][i] for i in keep])
        print(f"  {m:12s} | {np.nanmedian(g('dsnr_block')):+6.2f} dB | {np.nanmedian(g('ci_block')):7.1f} | "
              f"{np.nanmedian(g('dsnr_active')):+6.2f} dB | {np.nanmedian(g('dsnr_clean')):+6.2f} dB | "
              f"{np.nanmedian(g('dsnr_record_always_on')):+6.2f} dB | {np.nanmedian(g('dsnr_record_gated')):+6.2f} dB | "
              f"{np.nanmedian(g('clean_rrmse')):.3f} ({np.nanmedian(g('clean_rrmse') - cl_none):+.4f})", flush=True)

    print(f"\n  --- PRE-REGISTERED TESTS (N = {len(keep)}, paired over records, two-sided Wilcoxon, bootstrap 95% CI of the median) ---")
    def paired(a_key, b_key, metric):
        d = np.array([store[a_key][metric][i] - store[b_key][metric][i] for i in keep])
        d = d[np.isfinite(d)]
        med, lo, hi = boot_median_ci(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        return med, lo, hi, p, len(d)
    med, lo, hi, p, n_ok = paired("DCD_RefA", "NLMS_RefA", "dsnr_block")
    print(f"  PRIMARY P1  dSNR(blocks) DCD_RefA - NLMS_RefA: median {med:+.3f} dB [{lo:+.3f}, {hi:+.3f}], p = {p:.3e}, n = {n_ok}")
    med, lo, hi, p, n_ok = paired("DCD_RefA", "NONE", "clean_rrmse")
    verdict = "PASS" if hi < 0.02 else "FAIL"
    print(f"  PRIMARY P2  clean-epoch RRMSE DCD_RefA - NONE (non-inferiority, margin +0.02): median {med:+.4f} [{lo:+.4f}, {hi:+.4f}], p = {p:.3e} -> {verdict}")
    for a_key, b_key in (("DCD_RefC", "NLMS_RefC"), ("DCD_RefA_P5", "NLMS_RefA"), ("DCD_RefA", "DCD_RefB"), ("DCD_RefA", "DCD_RefC")):
        med, lo, hi, p, n_ok = paired(a_key, b_key, "dsnr_block")
        print(f"  SECONDARY   dSNR(blocks) {a_key} - {b_key}: median {med:+.3f} dB [{lo:+.3f}, {hi:+.3f}], p = {p:.3e} (unadjusted)")

print("\n=== STAGE A CUTOFF SWEEP (0.16, 0.32, 0.64 Hz) FOR DCD_RefA AT THE LORO P (SECONDARY) ===", flush=True)
for fc_val in (0.16, 0.32, 0.64):
    ds, cis, cls = [], [], []
    for idx_rec, rec_d in enumerate(records_data):
        c_hp, _ = apply_dc_blocker_hz(rec_d["eeg1_512"], 512.0, fc_val)
        n_hp, _ = apply_dc_blocker_hz(rec_d["eeg2_512"], 512.0, fc_val)
        e, _ = run_multichan_dcd_ridge_int(n_hp, rec_d["refs"]["RefA_ACC1_3ax"], 16, loro_P[idx_rec], gamma=frozen_gamma_p5, cal_len=rec_d["first_f_512"])
        d_b, ci_b, _, cl_rr = safe_metrics(c_hp, n_hp, e, rec_d["mask_art"], rec_d["mask_clean"])
        ds.append(d_b); cis.append(ci_b); cls.append(cl_rr)
    print(f"  fc={fc_val:4.2f} Hz (P_hp={hp_pole_shift(512.0, fc_val)}) | median dSNR(blocks) = {np.nanmedian(ds):+5.2f} dB | "
          f"median CorrImp = {np.nanmedian(cis):5.1f}% | median clean RRMSE = {np.nanmedian(cls):.3f}", flush=True)

print(f"\nEnd Time (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()} | evaluation elapsed {time.time() - t_eval0:.0f} s", flush=True)
