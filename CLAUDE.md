# CLAUDE.md — Mamba Stability Certificate Transfer to Closed-Loop Control

This file tells Claude how to work on this project. Read it fully before doing anything.

## 1. What this project is

**Research question:** Does the built-in latent stability certificate of selective state-space models (Mamba) — proven by Halloran et al. (2025) — transfer to stability of the *physical* predictions a model predictive controller actually uses? If not, what condition is missing, and can it be built into the architecture?

**Owner:** Md. Adnan Abdullah Sadi, Department of Electrical and Computer Engineering, North South University, Dhaka.
**Status:** Restarted from scratch on 9 October 2026, after an eight-phase first attempt (summarised in Section 12).

**The full research brief** — plain-language explanation, mathematical formulation, related work, hypotheses and success criteria — is `docs/research_proposal_certificate_transfer.md`. Read it at the start of every new session. This file covers *how to work*; the brief covers *what and why*. The brief's statistical success criteria (8+ seeds, significance tests) apply only at the confirmation stage described in Section 9 below.

**Key papers (read before proposing designs):**
- Halloran, Gulati, Roysdon (2025), arXiv:2406.00209 — the latent certificate
- Cevaal, de Jong, Lazar (2026), arXiv:2604.13857 — Mamba inside model predictive control
- Chung, Choi, Kim (2026), arXiv:2605.07755 — affine recurrences cannot correct drift; state-dependent maps can
- Cao et al. (2026), ICML, PMLR v306 — single-layer bounded-input bounded-output analysis of Mamba
- Zubić, Scaramuzza (2025), arXiv:2505.11602 — core certificate did not improve end-task error
- Biswas (2026), arXiv:2605.24868 — contractive one-step Jacobian, yet rollout error still grows

## 2. Working rules (non-negotiable)

1. **Claude never runs experiment code.** No training, no evaluation runs, no project tests, no installing the project environment. Claude writes and edits files; Sadi clones the repository and runs them from the terminal. **The one exception is mathematical analysis** (Section 3): Claude may run small, standalone calculations — symbolic algebra, derivatives, integrals, eigenvalues, small matrix computations, series bounds — to get a mathematical signal. These never involve training a model or running the project's experiment code.
2. **Mathematical signal first.** Before proposing a design or an experiment, Claude analyses it mathematically and reports whether the signal is positive, negative or inconclusive (Section 3).
3. **Single seed first.** Every experiment is first run on one seed (seed 0). Only if that result passes the experiment's pre-declared "worth confirming" rule does it go to multiple seeds (Section 9).
4. **Honesty over a positive result.** A well-supported negative answer is acceptable. Never tune until something "looks good". Hyperparameters are fixed by a declared pre-sweep, then the hypothesis is tested once.
5. **Declare the decision rule before running.** Each experiment config states its hypothesis, its mathematical prediction, and the exact threshold that would make it "worth confirming", before any result exists.
6. **Single-seed results show direction only.** Never call a single-seed result significant, proven or confirmed.
7. **Separate "the code works" from "the idea works".** Clean training and no divergence say nothing about whether a method helps.
8. **Flag your own mistakes plainly** and correct them.

## 3. Mathematical signal first

This project is mostly mathematics. Many outcomes can be predicted on paper before any training, and the first attempt shows it: the limit-cycle trap and the dead correction gate were both visible from a few lines of calculus, and each cost a full run to discover. So **every design, fix and next step starts with a mathematical check**, and the result shapes the recommendation.

### 3.1 What to check (pick whichever apply)

| Check | What it predicts | Example from this project |
|---|---|---|
| **Linearisation and Jacobian eigenvalues** of the augmented state (physical prediction + hidden state) at fixed points and along the limit cycle | Whether chained-prediction errors grow or shrink | A 2×2 block with the latent part at 0.5 still has an eigenvalue of 1.78 → certificate holds, errors still grow |
| **Singular values of the true system's Jacobian** along the orbit | Whether a proposed bound is even achievable | Van der Pol's largest singular value exceeds 1 on parts of the limit cycle → global contraction is impossible |
| **Error-recursion bound** $\|e_T\| \le \sum_k \prod_j \|J_j\| \,\|\varepsilon_k\|$ and its geometric-series form | How fast rollout error can grow for a given per-step amplification | $\gamma = 1.02$ per step over 50 steps → up to 2.7× amplification |
| **Derivative of the loss with respect to a new parameter at initialisation** | Whether gradient can reach a new component at all | A correction network $g(h)$ with $g(0)=0$ at $h_0 = 0$ gives zero contribution and zero gradient at the first step |
| **Order-of-magnitude comparison** of competing terms at initialisation | Whether a new term is large enough to matter | Penalty term $\sim 10^{-5}$ against prediction loss $\sim 10^{-3}$ → the penalty barely acts |
| **Fixed-point and invariance arguments** (zero in → zero out, symmetries, conserved quantities) | Degenerate behaviour that training cannot escape | Zero-initialised final layer → output constant in its input → no upstream gradient |
| **Lyapunov-function or contraction-metric candidates** | Whether a stability claim can be proven for a design | Transverse contraction is compatible with a limit cycle; global contraction is not |
| **Counterexamples** | Whether a proposed theorem can be true | Construct the smallest system where the claim fails |
| **Parameter count and run-time formulas** | Fairness of comparisons; whether an experiment fits the machine | Arms × seeds × minutes per run |

### 3.2 How to report it

Every proposal, design or next-step answer includes a short **Mathematical signal** block:

```
Mathematical signal: POSITIVE | NEGATIVE | INCONCLUSIVE
What was checked:    (one line, e.g. "eigenvalues of the augmented Jacobian at the origin")
Result:              (the number or expression)
Assumptions:         (e.g. "linearisation valid near the trajectory; inputs held fixed")
Implication:         (how this changes the recommendation)
```

### 3.3 How the signal changes the next step

- **Negative signal:** do not spend compute on the idea as stated. Redesign first, or run it only if the experiment's purpose is to test the mathematics itself.
- **Positive signal:** proceed to a single-seed run. The mathematics is a prediction, not evidence.
- **Inconclusive signal:** run the cheapest experiment that would settle it, and say what would settle it.

### 3.4 Rules

- The mathematical signal **never replaces an experiment** and never justifies dropping a planned control arm.
- Always state assumptions. Linearisation is local; bounds may be loose; a positive bound proves possibility, not that training will find it.
- Each experiment config records its prediction in a `math_signal` field, and the summary states whether the result **agreed or disagreed** with it. Disagreements are valuable: they show where the mathematical model of the problem is wrong.
- Calculations Claude runs for this purpose are standalone scripts (Python with SymPy or NumPy). Any derivation worth keeping is also saved to `analysis/` so Sadi can rerun it, and written up in `docs/math_notes.md`.

## 4. Writing and documentation style

- **No unexplained abbreviations or short forms** in documents, summaries, figure labels or printed output. Use full names ("model predictive control", "Long Short-Term Memory network", "root-mean-square error"). An abbreviation may follow only after the full name has been given once.
- **Define every technical term before using it**, or avoid it.
- **No LaTeX tables** anywhere in the project output. Results go to CSV, JSON and plain-language Markdown summaries.
- Chat answers: short and direct. When Sadi says "shortly tell me", give a few sentences.
- When interpreting results, always compare side by side against the baseline, with numbers.

## 5. Environment and installation

**Primary machine:** MacBook Air M1 (8 GB memory), Miniconda, VS Code, terminal. **Also supported:** Kaggle or any machine with an NVIDIA graphics card.

The repository ships both files, and they must stay in sync:

`environment.yml` (creates the whole environment in one command):
```yaml
name: mamba-cert
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - -r requirements.txt
```

`requirements.txt` (every Python library, installable at once with pip):
```
torch>=2.2
numpy
scipy
sympy
pandas
matplotlib
pyyaml
tqdm
pytest
```

When a new library is needed, add it to `requirements.txt` only (the environment file reads it), and tell Sadi to run the update command in Section 7.

**Device selection:** `--device auto` (default) picks NVIDIA graphics card (`cuda`) → Apple graphics (`mps`) → `cpu`. Notes for the Mac:
- Use `float32` only; Apple graphics does not support `float64`.
- Some operations are not deterministic on Apple graphics, and a few are not implemented; set `PYTORCH_ENABLE_MPS_FALLBACK=1` (the scripts do this) so missing operations fall back to the processor.
- The models loop over time step by step, so a small model may run as fast or faster on `cpu` than on `mps`. The first experiment should time both with `tools/benchmark_device.py` and record the faster one in `configs/base.yaml`.
- 8 GB memory: keep batch sizes modest (default 256) and never hold all per-run histories in memory at once.

## 6. Project structure

Mirrors the layout of Sadi's event camera repository (ABP): settings live in `configs/`, reusable code in `src/`, helper programs in `tools/`, shell wrappers in `scripts/`, entry-point programs at the top level. **One rule added:** a new experiment variant is a new configuration file — never a new `train_v2.py`. The top level stays small.

```
mamba-certificate-control/
├── README.md                      # what the project is + exact commands to run
├── CLAUDE.md                      # this file
├── environment.yml                # conda environment (reads requirements.txt)
├── requirements.txt               # all Python libraries
├── .gitignore                     # ignores outputs/, checkpoints, __pycache__, .DS_Store
│
├── train.py                       # train all arms of one experiment from one config
├── evaluate.py                    # re-evaluate saved checkpoints
├── summarize.py                   # write the plain-language summary for an experiment
│
├── configs/
│   ├── base.yaml                  # shared defaults: device, batch size, epochs, horizon
│   ├── systems/
│   │   └── vanderpol.yaml         # mu = 1, time step, data generation settings
│   ├── models/
│   │   ├── ssm_minimal.yaml       # minimal selective state-space cell
│   │   ├── mamba_full.yaml        # full Mamba block, Cevaal-matched scale
│   │   └── lstm.yaml              # parameter-matched Long Short-Term Memory baseline
│   └── experiments/
│       └── exp01_certificate_ablation.yaml   # hypothesis, math signal, arms, decision rule
│
├── src/
│   ├── __init__.py
│   ├── systems/                   # true dynamics: vanderpol.py (+ later systems)
│   ├── data/                      # trajectory generation, windowing
│   ├── models/                    # ssm_minimal.py, mamba_block.py, mamba_dynamics.py,
│   │                              # lstm_dynamics.py, registry.py (name -> class)
│   ├── losses/                    # teacher forcing, chained-prediction loss, penalties
│   ├── training/                  # trainer.py: loop, checkpointing, tqdm progress
│   ├── evaluation/                # metrics.py, diagnostics.py (error amplification, Jacobians)
│   ├── control/                   # model predictive control (later phase)
│   └── utils/                     # config.py, seed.py, device.py, io.py, stats.py, summary.py
│
├── analysis/                      # standalone mathematical checks (SymPy / NumPy, no training)
│   ├── a01_augmented_jacobian_counterexample.py
│   └── a02_vanderpol_jacobian_along_orbit.py
│
├── tools/
│   ├── check_model.py             # architecture checks before training (see Section 8)
│   ├── benchmark_device.py        # time one epoch on cpu vs mps vs cuda
│   └── plot_results.py            # figures for an experiment folder
│
├── scripts/                       # shell wrappers that only call Python programs
│   ├── setup_env.sh
│   ├── smoke_test.sh              # tiny run of everything, ~1 minute
│   └── run_experiment.sh          # check -> train -> summarize for one experiment
│
├── tests/                         # pytest: equivalence, gradient flow, true-system sanity
│
├── docs/
│   ├── research_proposal_certificate_transfer.md
│   ├── math_notes.md              # derivations and mathematical signals, dated
│   ├── decisions.md               # dated: what was decided and why
│   └── results_log.md             # dated: each experiment's verdict + math agreement
│
└── outputs/                       # git-ignored; written by runs
    └── exp01_certificate_ablation/
        ├── results.csv            # one row per (arm, seed); used to resume
        ├── summary.md             # plain-language summary (Section 10)
        ├── figures/
        └── runs/<arm>_seed<k>/    # config.yaml, metrics.json, train_log.csv, checkpoint.pt
```

## 7. How Sadi runs things (terminal style)

Every program takes `--config`; every result lands in `outputs/<experiment_name>/`. The `README.md` must list these exact commands and stay up to date.

```bash
# one-time setup
git clone https://github.com/sadi2003q/<repository-name>.git
cd <repository-name>
conda env create -f environment.yml
conda activate mamba-cert

# after pulling changes that add a library
conda env update -f environment.yml --prune

# 0. (optional) rerun a mathematical check
python analysis/a01_augmented_jacobian_counterexample.py

# 1. check the architecture before spending time on training
python tools/check_model.py --config configs/experiments/exp01_certificate_ablation.yaml

# 2. tiny end-to-end run to confirm nothing crashes (minutes, not hours)
python train.py --config configs/experiments/exp01_certificate_ablation.yaml --quick

# 3. the real single-seed run
python train.py --config configs/experiments/exp01_certificate_ablation.yaml --seed 0

# 4. plain-language summary
python summarize.py --experiment outputs/exp01_certificate_ablation

# only if step 4 says "worth confirming": multiple seeds
python train.py --config configs/experiments/exp01_certificate_ablation.yaml --seeds 0 1 2 3 4 5 6 7
python summarize.py --experiment outputs/exp01_certificate_ablation

# or everything for one experiment in one command
bash scripts/run_experiment.sh exp01_certificate_ablation

# tests
pytest -q
```

Common flags for `train.py`: `--seed` / `--seeds`, `--quick`, `--device auto|cpu|mps|cuda`, `--arms <names>` (run only some arms), `--fresh` (ignore previous results; default is to resume).

## 8. Code conventions

- **Configuration-driven.** No hard-coded hyperparameters in code. Experiment configs inherit from `configs/base.yaml` plus a system and a model config, and override only what differs. The resolved configuration is saved into each run folder.
- **Every experiment config contains** `name`, `hypothesis` (one sentence), `math_signal` (prediction, what was checked, and assumptions — Section 3), `arms` (each with a short `why`), and `decision_rule` (the threshold for "worth confirming").
- **Progress with `tqdm`, always.** An outer bar over runs (`run 3/6: free_A seed0`) and an inner bar over epochs showing the current loss as a postfix. Use `tqdm.write` for messages so bars are not broken.
- **Resume by default.** After each completed run, append a row to `results.csv` and save `metrics.json` and `checkpoint.pt`. On restart, skip completed (arm, seed) pairs. Also checkpoint every N epochs within a run, so an interrupted run continues mid-training.
- **Determinism:** `src/utils/seed.py` sets Python, NumPy and PyTorch seeds, deterministic cuDNN settings, `CUBLAS_WORKSPACE_CONFIG`, and `torch.use_deterministic_algorithms(True, warn_only=True)`. Note in the summary that Apple graphics may still be slightly non-deterministic.
- **Architecture checks (`tools/check_model.py` and `tests/`) must pass before training** and fail loudly if:
  - the step-by-step and the parallel forward passes disagree by more than 1e-4;
  - any newly added component receives zero gradient (print gradient norms in scientific notation);
  - any newly added gate or correction is exactly zero at the first time step.
  Print the parameter count of every arm next to Cevaal et al.'s reference (Van der Pol: Mamba 3,418; Long Short-Term Memory 3,052).
- **Divergence handling:** clamp states during chained prediction so a diverging run still produces a finite loss; record `diverged: true/false`; skip and count non-finite gradient steps.
- **Both prediction regimes** are evaluated for every model: chained prediction (one-step model applied repeatedly) and direct multi-step prediction (whole horizon in one forward pass).
- **Before proposing any experiment, estimate its run time** (arms × seeds × minutes per run on Sadi's device) and state it. Loss curves in the first attempt flattened by epoch 30–50; do not default to 80 epochs.
- Figures use `tick_labels=` in `boxplot` (the older `labels=` is deprecated) and are saved to `figures/`.

## 9. Experiment workflow: math, explore, then confirm

**Stage 0 — Mathematical signal.** Section 3. Recorded in the config before running.

**Stage 1 — Explore (one seed).** Run seed 0 for every arm. The summary reports direction only. The experiment's `decision_rule` decides whether to continue. Default rule unless the config says otherwise:
- a proposed method is **worth confirming** if its 50-step chained-prediction error is at least **20% lower** than the baseline's, its one-step error is no more than **10% worse**, and it did not diverge;
- an ablation (removing something) is **worth confirming** if any arm differs from the baseline by more than **20%** in either direction.

**Stage 2 — Confirm (8+ seeds).** Only for experiments that passed Stage 1. Add medians with bootstrap 95% confidence intervals, two-sided Mann–Whitney tests and Cliff's delta. "Significant" means p < 0.05 **and** |Cliff's delta| > 0.33. Only Stage 2 results go into the paper.

**Always:** include a parameter-matched control arm for any proposed method; report diverged runs as a count (e.g. "2 of 8 runs blew up"), never as an error number — a clamped error of ~1,000 measures the clamp, not the model; `--quick` checks that code runs and never answers a hypothesis.

After each experiment, add 2–3 lines to `docs/results_log.md` (including whether the mathematical prediction held) and any design choice to `docs/decisions.md`.

## 10. The result summary (`summary.md`, also printed to the terminal)

Short, intuitive, readable by a non-technical person in under a minute, so Sadi can glance at the numbers and picture what happened. Always this shape:

```
RESULT — exp01_certificate_ablation   (1 seed: direction only)

Question: If we remove the built-in stability guarantee, do predictions get worse?

                              Next-step error   50-step error   Blew up?
  Baseline (guarantee on)          0.004            0.077          no
  Guarantee removed                0.004            0.079          no
  Started unstable                 —                —              yes

What this means:
  • Removing the guarantee changed 50-step error by +3%  → about the same.
  • Errors grow about 19× between 1 step and 50 steps (smaller is better).
  • The model that started unstable blew up, so the guarantee matters only
    when training begins in a bad place.

Math predicted: no difference once trained (training keeps decay negative anyway).
Result agreed with the prediction.

Decision: NOT worth confirming (no arm differed by more than 20%).
```

Rules for the summary:
- Lead with the plain-language question, then one small table, then 2–4 "what this means" bullets, then the mathematical prediction and whether it held, then the decision.
- Every difference is given as a percentage against the baseline, with a word label: within ±10% → "about the same"; 10–20% → "slightly better/worse"; more than 20% → "clearly better/worse (on this seed)".
- Say which direction is good ("smaller is better").
- No abbreviations, no statistics jargon at Stage 1. At Stage 2, add one line per comparison in plain words (e.g. "better in 8 of 8 seeds; the difference is reliable").
- Technical detail (Jacobian norms, error-amplification curves, gradient counts) goes to `metrics.json` and `figures/`, not the summary.

## 11. Known pitfalls (each one happened in the first attempt)

| Pitfall | Fix |
|---|---|
| Forcing global contraction destroys the Van der Pol limit cycle | Bound *error* dynamics relative to the true system, or use transverse contraction; never require the state Jacobian below 1 everywhere |
| A single-step Jacobian bound never activated (model already under-expands) | Measure finite-horizon error amplification through the full recurrent state |
| Zero-initialising a gate's final layer made it constant, so no gradient reached anything upstream | Small non-zero initialisation; verified by `check_model.py` |
| A correction conditioned only on the hidden state is exactly zero at the initial state h = 0 | Condition on the current input as well |
| A gate initialised near zero receives vanishing gradient (cold start) | Start gates near 0.5 for all arms so comparisons are fair |
| `torch.autograd.grad` through `nn.LSTM` fails in evaluation mode ("cudnn RNN backward can only be called in training mode") | Temporarily switch to training mode inside the Jacobian function, then restore the previous mode |
| A pre-sweep value was applied after the main grid had already run | Pre-sweep results are written into the experiment config before the main run; `train.py` refuses to run if a required value is still a placeholder |
| A run's settings silently differed from what was intended, giving an 8-hour job | `train.py` prints the resolved configuration and the estimated run time, and asks for confirmation before long runs |
| `.item()` rounded tiny gradients to print as 0.0 | Print gradient norms in scientific notation |
| Short `--quick` runs (errors ~10× the full runs) were compared as if meaningful | `--quick` output is labelled "code check only — do not interpret" |

The first four rows were all predictable with a mathematical check (Section 3) before any run.

## 12. What the first attempt established (prior evidence, not conclusions)

All on Van der Pol, mostly 2–3 seeds:
- The latent certificate holds by construction in every run.
- Under teacher forcing, chained-prediction error was 200–300× the one-step error.
- Training on chained predictions fixed most of the rollout error; the certificate did not.
- Removing the certificate entirely (free-sign decay parameter) changed nothing: training kept the decay negative by itself. A model *initialised* unstable diverged in every run.
- A plain Long Short-Term Memory network matched the certified model.
- Physical-space penalties (bounded output gate, spectral normalisation of the readout, a penalty on excess error amplification, and their combination) **did not beat the baseline**; spectral normalisation hurt accuracy.
- Diagnostic: the trained model's single-step Jacobian was already *below* the true system's, so rollout error was spreading through the **hidden state**, not the readout.
- A state-dependent correction inside the recurrence (motivated by Chung et al.) was only pilot-tested; inconclusive.

**Direction implied:** penalties on top of an affine recurrence are unlikely to work; changes to the recurrence itself, and a formal account of why the latent certificate leaks, are the open paths.

## 13. Plan for the restart

Each experiment starts with a mathematical signal, declares its decision rule in advance, runs one seed first, and goes to 8+ seeds only if it passes.

1. **exp01 — Certificate ablation:** certified versus free-sign versus unstable-initialised decay; teacher forcing and chained-prediction training; both prediction regimes.
2. **Formal analysis:** counterexample showing the latent certificate does not bound chained-prediction error (augmented-state Jacobian argument); attempt to extend Chung et al. to continuous dynamics. Scripts in `analysis/`, write-up in `docs/math_notes.md`.
3. **exp02+ — Error-dynamics certificate** that respects limit cycles, against the baseline and a parameter-matched control.
4. **Generalisation:** a second nonlinear system; the same method on a Long Short-Term Memory network.
5. **Closed loop:** model predictive control with the certificate, under disturbances and parameter drift, against unconstrained Mamba-based and Long Short-Term Memory-based control.
