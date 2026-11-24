"""
stage5_tradeoff_figure.py  (EXPLORATORY; reads stage4_stdout.txt, no filtering is re-run)

1. Parses the E4 CSV block (=== CSV E4 BEGIN === ... === CSV E4 END ===) from stage4_stdout.txt.
2. Recomputes per-operating-point medians and the clean-RRMSE-minus-NONE bootstrap CI exactly as stage4/v2
   (boot_median_ci: 10,000 resamples, seed 42) and checks them against the E4 table printed in stage4_stdout.txt.
3. Prints the post-hoc paired comparisons used in the paper text (Wilcoxon signed-rank, two-sided, unadjusted),
   so every number in the text is printed by a script.
4. Draws a two-panel figure (RefA, RefC): median dSNR over clean epochs (x) vs median dSNR over artifact blocks (y).
   Filled markers = operating point passes the P2 margin (upper 95% CI of median dRRMSE < +0.02); hollow = fails.

Usage: python3 stage5_tradeoff_figure.py stage4_stdout.txt
Output: physionet_tradeoff.png, physionet_tradeoff.pdf, and printed tables.
"""
import io
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 stage5_tradeoff_figure.py stage4_stdout.txt")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = f.readlines()

try:
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "=== CSV E4 BEGIN ===")
    i1 = next(i for i, l in enumerate(lines) if l.strip() == "=== CSV E4 END ===")
except StopIteration:
    raise SystemExit("CSV E4 block not found in the stage4 output.")
df = pd.read_csv(io.StringIO("".join(lines[i0 + 1:i1])))
print(f"CSV rows: lines {i0 + 2}-{i1} of {sys.argv[1]} ({len(df)} rows)")
expected_cols = ["record", "reference", "method", "param", "dsnr_blocks", "dsnr_clean", "clean_rrmse", "none_clean_rrmse", "corr_imp"]
if list(df.columns) != expected_cols:
    raise SystemExit(f"Unexpected CSV columns: {list(df.columns)}")
df["d_rrmse"] = df["clean_rrmse"] - df["none_clean_rrmse"]
df["pval"] = df["param"].str.split("=").str[1].astype(float)
df["rec_no"] = df["record"].str.split("_").str[1].astype(int)  # stage4 order eeg_1..eeg_23 (bootstrap depends on order)


def boot_median_ci(v, n_boot=10000, seed=42):  # identical to the v2 function
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    rng_b = np.random.default_rng(seed)
    meds = np.median(v[rng_b.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.median(v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# E4 table as printed by stage4, for the self-check
pat = re.compile(r"^\s+(NLMS_Ref[AC]|DCD_Ref[AC])\s+\|\s+(\S+)\s+\|\s+([+-]\d+\.\d+)\s+\|\s+([+-]\d+\.\d+)\s+\|\s+([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]")
printed = {}
for ln, l in enumerate(lines[:i0], start=1):
    m = pat.match(l)
    if m:
        printed[(m.group(1), m.group(2))] = (ln, [float(m.group(k)) for k in range(3, 8)])

rows = []
ok = True
for (meth, par), g in df.groupby(["method", "param"]):
    g = g.sort_values("rec_no")
    b, c = float(np.median(g["dsnr_blocks"])), float(np.median(g["dsnr_clean"]))
    dm, lo, hi = boot_median_ci(g["d_rrmse"].values)
    rows.append(dict(method=meth, ref=meth.split("_")[1], param=par, pval=float(par.split("=")[1]),
                     blocks=b, clean=c, drr=dm, lo=lo, hi=hi, p2=hi < 0.02))
    if (meth, par) not in printed:
        ok = False
        print(f"  missing printed E4 row for {meth} {par}")
        continue
    ln, ref = printed[(meth, par)]
    mine = [b, c, dm, lo, hi]
    tol = [0.006, 0.006, 0.00006, 0.00006, 0.00006]
    if any(abs(x - y) > t for x, y, t in zip(mine, ref, tol)):
        ok = False
        print(f"  MISMATCH {meth} {par} (stage4 line {ln}): mine {mine} vs printed {ref}")
summ = pd.DataFrame(rows)
print(f"SELF-CHECK recomputed medians/CIs match the printed E4 table ({len(printed)} rows): {'PASS' if ok else 'FAIL'}")
if not ok or len(printed) != len(summ):
    sys.exit(1)

print("\n=== POST-HOC PAIRED COMPARISONS (EXPLORATORY; two-sided Wilcoxon, unadjusted; bootstrap CI as v2) ===")


def sel(meth, par):
    return df[(df.method == meth) & (df.param == par)].sort_values("rec_no").set_index("record")


for a, b in ((("NLMS_RefA", "mu=0.005"), ("DCD_RefA", "P=11")),
             (("DCD_RefC", "P=11"), ("NLMS_RefC", "mu=0.005")),
             (("DCD_RefC", "P=13"), ("NLMS_RefC", "mu=0.005"))):
    A, B = sel(*a), sel(*b)
    if list(A.index) != list(B.index):
        raise SystemExit("record mismatch")
    for k, lab in (("dsnr_blocks", "dSNR blocks"), ("dsnr_clean", "dSNR clean"), ("d_rrmse", "clean RRMSE")):
        d = (A[k] - B[k]).values
        m, lo, hi = boot_median_ci(d)
        print(f"  {a[0]} {a[1]} - {b[0]} {b[1]} | {lab:11s}: median {m:+.4f} [{lo:+.4f}, {hi:+.4f}], "
              f"first>second {int(np.sum(d > 0))}/{len(d)}, p={wilcoxon(d).pvalue:.3g}")

print("\n=== OPERATING POINTS (medians over records) ===")
for _, r in summ.sort_values(["ref", "method", "pval"]).iterrows():
    print(f"  {r.method:9s} {r.param:8s} blocks {r.blocks:+6.2f} | clean {r.clean:+6.2f} | dRRMSE {r.drr:+.4f} "
          f"[{r.lo:+.4f}, {r.hi:+.4f}] | P2 {'pass' if r.p2 else 'fail'}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
style = {"NLMS": dict(color="#1f77b4", marker="o"), "DCD": dict(color="#d62728", marker="s")}
for ax, ref, title in zip(axes, ("RefA", "RefC"), ("(a) RefA: ACC of disturbed electrode, 3 axes",
                                                    "(b) RefC: differential ACC + velocity + position, 9 axes")):
    for fam in ("NLMS", "DCD"):
        s = summ[summ.method == f"{fam}_{ref}"].sort_values("pval", ascending=(fam == "NLMS"))
        st = style[fam]
        ax.plot(s.clean, s.blocks, "-", color=st["color"], lw=1.2,
                label=("NLMS (step size μ)" if fam == "NLMS" else "Integer DCD-RLS (memory P)"))
        for _, r in s.iterrows():
            ax.plot(r.clean, r.blocks, st["marker"], ms=6, mec=st["color"], mfc=st["color"] if r.p2 else "white", mew=1.4)
            lab = (f"μ={r.param.split('=')[1]}" if fam == "NLMS" else r.param)
            ax.annotate(lab, (r.clean, r.blocks), textcoords="offset points", xytext=(4, 3), fontsize=7, color=st["color"])
    ax.plot(0, 0, "k+", ms=9, mew=1.5, label="Unprocessed (NONE)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Median ΔSNR over clean epochs (dB)  ← more damage")
    ax.set_ylabel("Median ΔSNR over artifact blocks (dB)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("physionet_tradeoff.png", dpi=300)
fig.savefig("physionet_tradeoff.pdf")
print("\nSaved physionet_tradeoff.png and physionet_tradeoff.pdf")
