# Fixed-Point Ridge DCD-RLS Adaptive Filtering for Wearable EEG

Code, output logs and LaTeX sources for the preprint *Fixed-Point Ridge DCD-RLS Adaptive Filtering for Wearable EEG: Hardware Fidelity and Real-Data Comparison Against NLMS* (Varun Bajpai).

Every number in the paper is printed by one of the scripts below. The supplement (`supplement.tex`) maps each number to the line of the log file that prints it.

## Files

| File | Role |
|---|---|
| `stage2_round4_benchmark.py` | Stage 2 data loading, resampling and path filters (loaded by the round-6 script) |
| `stage2_round6_ridge_benchmark.py` | Stage 2 semi-synthetic benchmark (hypotheses H1–H4) |
| `stage2_round6_ridge_stdout.txt` | Stage 2 output log |
| `stage3_physionet_benchmark_auditfix_v2.py` | Stage 3 PhysioNet benchmark (pre-specified tests P1, P2) |
| `stage3_physionet_stdout.txt` | Stage 3 output log |
| `stage4_physionet_diagnostics_auditfix.py` | Stage 4 exploratory diagnostics E1–E5 |
| `stage4_stdout.txt` | Stage 4 output log (includes the per-record CSV for E4) |
| `stage5_tradeoff_figure.py` | Recomputes the E4 medians and CIs, prints the post-hoc paired tests, draws Fig. 1 |
| `stage5_stdout.txt` | Stage 5 output log |
| `physionet_tradeoff.pdf` | Fig. 1 |
| `main.tex` | Paper (pdfLaTeX, IEEEtran) |
| `supplement.tex` | Supplementary material (compile with XeLaTeX) |
| `supplement.pdf` | Supplementary material, compiled from `supplement.tex` with XeLaTeX |
| `LICENSE` | MIT licence |
| `push_to_github.sh` | One-time script that publishes this folder to GitHub (macOS zsh) |

## Requirements

The logged runs used Python 3.13.17, NumPy 2.5.4 and SciPy 1.18.1 (see the log headers).

```
pip3 install numpy scipy wfdb pandas matplotlib
```

## Data (not included; download separately)

- **EEGdenoiseNet** (Zhang et al., J. Neural Eng. 18:056057, 2021). Place `EEG_all_epochs.npy` and `EMG_all_epochs.npy` in a folder named `eegdenoisenet/`. The SHA-256 of both files is printed at the top of `stage2_round6_ridge_stdout.txt`.
- **PhysioNet Motion Artifact Contaminated fNIRS and EEG Data** (doi:10.13026/C2988P). Download the 23 EEG records with:

```
python3 -c "
import wfdb, os
os.makedirs('motion_artifact_data', exist_ok=True)
for i in range(1, 24):
    wfdb.dl_database('motion-artifact', 'motion_artifact_data', records=[f'eeg_{i}'], annotators=['trigger'])
print('ALL DONE')
"
```

The folder should then contain 69 files (`.hea`, `.dat` and `.trigger` for each of the 23 records).

## Reproduction

Run every command from the repository folder. The elapsed times are the ones recorded in each log.

```
# Stage 2: semi-synthetic benchmark (2,348 s)
python3 stage2_round6_ridge_benchmark.py stage2_round4_benchmark.py eegdenoisenet > stage2_round6_ridge_stdout.txt 2>&1

# Stage 3: PhysioNet benchmark, pre-specified P1/P2 (6,865 s)
python3 stage3_physionet_benchmark_auditfix_v2.py motion_artifact_data --gamma-p5 0.125 > stage3_physionet_stdout.txt 2>&1

# Stage 4: exploratory diagnostics E1-E5 (18,019 s; resumes from stage4_cache_full.json if interrupted)
python3 stage4_physionet_diagnostics_auditfix.py motion_artifact_data --exp ALL > stage4_stdout.txt 2>&1

# Stage 5: post-hoc paired tests and Fig. 1 (seconds)
python3 stage5_tradeoff_figure.py stage4_stdout.txt > stage5_stdout.txt
```

Stage 4 needs `stage3_physionet_benchmark_auditfix_v2.py` and `stage3_physionet_stdout.txt` in the same folder. It refuses to run if the v2 script's SHA-256 differs from the one that produced the reported run. Before any experiment, it reproduces values from the Stage 3 log (self-checks SC0–SC5).

`stage5_tradeoff_figure.py` was edited after the logged run only to remove a caption line drawn inside the figure. Its printed output is unchanged.

## Pre-specification and disclosures

"Pre-specified" means that hypotheses, metrics and pass rules were fixed in code before the evaluation runs reported here. There was no public registration. Two disclosures:

- **Stage 2:** the held-out epochs had been used in earlier development rounds.
- **Stage 3:** the DCD memory grid was extended from {5, 7, 9} to {5, 7, 9, 11} under a pre-declared grid-edge rule, after a four-record pilot run.

All Stage 4 and Stage 5 analyses are exploratory, with unadjusted p-values.

## Checksums (SHA-256, first 16 hex characters)

| File | SHA-256 |
|---|---|
| `stage2_round4_benchmark.py` | `08bbee58b5a6633a…` |
| `stage2_round6_ridge_benchmark.py` | `55e215c03d69fb51…` |
| `stage3_physionet_benchmark_auditfix_v2.py` | `c32a8afcbb71673f…` |
| `stage4_physionet_diagnostics_auditfix.py` | `db845a2cb8c7f9ad…` |
| `stage5_tradeoff_figure.py` | `dae635eea4b0f073…` |
| `stage2_round6_ridge_stdout.txt` | `271d2f19938d3a8b…` |
| `stage3_physionet_stdout.txt` | `40499c6e24f076ce…` |
| `stage4_stdout.txt` | `efe9d4ba18c3991a…` |
| `stage5_stdout.txt` | `b3d2a380bd56e0a1…` |
| `main.tex` | `b54cc4f795afe211…` |
| `supplement.tex` | `cca9f67647011ac2…` |
| `supplement.pdf` | `721ccdd1545293f4…` |
| `physionet_tradeoff.pdf` | `73e41e79cb331a0f…` |
| `LICENSE` | `02cc46b3c9ab8454…` |
| `push_to_github.sh` | `256f3fc2939dda1c…` |

Three of these hashes are also recorded inside the logs:

- `stage2_round4_benchmark.py` in `stage2_round6_ridge_stdout.txt`;
- `stage3_physionet_benchmark_auditfix_v2.py` in `stage4_stdout.txt`;
- `stage4_physionet_diagnostics_auditfix.py` in `stage4_stdout.txt`.

The Stage 2 and Stage 3 logs do not record their own script's hash.

## Citation

If you use this code, please cite the preprint (arXiv identifier to be added; code at https://github.com/MACVXACE/dcd-rls-eeg) and both datasets.
