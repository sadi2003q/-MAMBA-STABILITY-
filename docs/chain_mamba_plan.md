# Chain-Mamba — making Mamba work for chained prediction on Van der Pol

Written 3 November 2026. Diagnosis, literature, design, training recipe, results, and the plan from here.
Results are seed 0 unless marked otherwise, so they show **direction only** (CLAUDE.md Section 9).
Everything was trained on a 2-core cloud processor because the person who requested this work asked for the
design to be verified by running it. CLAUDE.md's rule that Claude never runs experiment code was set aside
for this one request. The runs reuse the repository's own data, windows, training loop and evaluation code
unchanged, so they compare directly with exp01d.
Abbreviations: LSTM = Long Short-Term Memory network; MPC = model predictive control; SwiGLU = gated
feed-forward layer with a SiLU gate; RMS = root mean square.

---

## 1. Result in one paragraph

The full Mamba in exp01d was weak at chained prediction (50-step error 0.465 against the LSTM's 0.267). Two
things fix it. (a) **Training.** Training used 1,250 updates, with the full 50-step chained loss from the
first update. Cevaal et al. trained for 4,000 epochs; this repository used 40. The new recipe grows the
rollout horizon from 5 to 50, keeps a one-step loss next to the chained loss, and runs 4 times as many
updates with a cosine learning-rate schedule. (b) **Architecture.** Chain-Mamba adds a memory-free linear
path from the current state to the next increment (the anchor), scales the output by the size of true
increments, adds a SwiGLU layer to every Mamba layer (Mamba has no state-times-input products inside a time
step), shortens the convolution from 10 to 4, and drops the RMS normalisation in front of the read-out.
Parameters stay matched: 3,310 against 3,282. On seed 0 the 50-step chained error falls from **0.465 to
0.066** (−86%). The one-step error falls from 0.094 to 0.014, and the free-running limit cycle is right to
0.5% in amplitude and 2.4% in period. The same recipe also lifts the LSTM a lot (0.267 → 0.098) and the
unchanged full Mamba (0.465 → 0.176). Chain-Mamba still beats them by 32% and 62%. On seed 1 Chain-Mamba
reaches 0.057 against the LSTM's 0.120. A declared control without the memory margin did as well or
better (0.054 and 0.056), so the margin is *not* what helps; Section 7 says what this means.

---

## 2. What "chained prediction" means here, and the target

Chained prediction is the one-step model applied to its own predictions: true states x_0 … x_10 warm up the
memory, then the model predicts x_11 and feeds that prediction back in, for 50 steps. This is the error a
model predictive controller sees when it rolls out a one-step model.
Target (CLAUDE.md default rule for a proposed method): 50-step error at least 20% below the baseline
(0.465 → 0.372 or lower), one-step error no more than 10% worse, no blow-up.

---

## 3. Why the current model is weak (measured, seed 0)

| Cause | Evidence | Fix |
|---|---|---|
| **Too little training, too hard from the start** | 40 epochs × 32 batches = 1,250 updates (Cevaal et al.: 4,000 epochs). Chained loss over all 50 steps from update 1 (first-epoch loss 140). Learning rate down to 0.30× by the end. Chained validation loss still noisy at epoch 40. | Rollout curriculum 5 → 50 steps, one-step loss kept (weight 0.5), 80 epochs at batch 128 (5,040 updates), cosine schedule, peak 0.002 |
| **No product between state and input inside a step** | Mamba's per-step map is linear projections, a convolution and one SiLU gate. Van der Pol's step needs dt·μ·x1²·x2. Mamba-X (2026) shows Mamba lacks state–input products and that adding them helps most in free-running rollout | SwiGLU feed-forward after the mixer in every layer (as in the Mamba-3 block) |
| **The read-out has to rebuild the identity-like linear step through 6 normalised layers** | x_{t+1} = x_t + 0.1·[x2, −x1 + u + μ(1−x1²)x2] is linear except one term. A final RMS normalisation also removes the size of the feature vector, but increment size varies more than 10× along the orbit | Anchor: increment = head(y_t) + W_skip [x_t, u_t]. Increment normalisation (output × std of true increments, as SUBNET's state-derivative normalisation). No final normalisation |
| **Long convolution mixes 9 earlier, possibly wrong, predictions into every step** | kernel 10 | kernel 4 |
| *Slow memory (hypothesis — not confirmed)* | Largest decay factor 0.996–0.999 in every layer of the trained full Mamba. Step size Δ median 0.006–0.023, never left its starting range. Memory steady-state gain 1/(1−ρ) = 237–1,017 (`tools/diagnose_memory_margin.py`) | Memory margin: Δ ≥ 0.05 and A ≤ −1, so ρ ≤ 0.951. **The control without it did better — see Section 7** |

---

## 4. What recent research says, and what was taken from it

- **Mamba-X: Bilinear Input Modulation for Mamba (arXiv 2604.17221, 2026).** Mamba's diagonal recurrence
  has no state–input product. Bilinear variants improved free-running rollout by about 3.5× on NARMA-10, and
  by up to two orders of magnitude with a larger state. → SwiGLU layer (cheap, keeps the parallel scan).
- **Mamba-3 (arXiv 2603.15569, 2026).** Its block puts a SwiGLU feed-forward after the state-space layer.
  Its complex, rotating state is the Stage 3 design and is not needed here.
- **SUBNET — Beintema, Schoukens, Tóth (2022–2023).** A truncated simulation loss over short subsections,
  plus state-derivative normalisation, which they show is essential for reliable training. → Chained loss on
  50-step windows; increment normalisation.
- **Pushforward and noise injection — Brandstetter et al. (2022); Sanchez-Gonzalez et al. (2020);
  scheduled sampling (Bengio et al., 2015).** Training must expose the model to its own errors, starting
  easy. → Horizon curriculum.
- **Wake-surrogate study (arXiv 2608.07189, 2026).** A ten-seed sweep of noise injection did not remove phase
  drift. → Noise injection left at 0 in the recipe (the option exists: `state_noise_std`).
- **PhyxMamba (arXiv 2505.23863, 2025)** uses a student-forcing stage to stop error accumulation in Mamba
  rollouts. **Mamba-assisted closure (arXiv 2606.05371, 2026)** trains sequence-to-sequence and deploys
  step by step. Both support "train on the regime you deploy".
- **Cevaal, de Jong, Lazar (arXiv 2604.13857, 2026).** RMS normalisation, a residual from the embedding, and
  4,000 training epochs for Van der Pol. The training budget is the main gap between their setup and exp01d.
- **Bonassi et al. (2024), deep state-space models as deep Wiener models.** Simulation models need fading
  memory and incremental input-to-state stability. This motivated the margin, which the results do not
  support as an accuracy fix.

---

## 5. The model: Chain-Mamba (`src/models/mamba_chain.py`, `configs/models/chain_mamba.yaml`)

Per step the input is s_t = [x̂_t, u_t, mask_t] (normalised). Each of the 2 layers (model width 12, state 8):

```
mixer:  (xi, z) = in_proj(RMSNorm(h));  xc = SiLU(causal depthwise conv_4(xi))
        Δ_t = δ_floor + softplus(W_Δ xc),  A = -(a_min + exp(A_log)),  Ā_t = exp(Δ_t A)   ≤ exp(-δ_floor a_min)
        H_t = Ā_t ⊙ H_{t-1} + (Δ_t xc) B_t,   y = (C_t·H_t + D xc) ⊙ SiLU(z);   h ← h + out_proj(y)
SwiGLU: (g, v) = W_in RMSNorm(h);  h ← h + W_out (SiLU(g) ⊙ v)
read-out (src/models/dynamics.py):
        x̂_{t+1} = x̂_t + σ_Δ ⊙ ( head(h_t) + W_skip s_t )      σ_Δ = std of true increments (training data)
```

- 3,310 parameters (full Mamba 3,282; LSTM 3,152; Cevaal et al. 3,418).
- Margin variant (`chain_mamba.yaml`): δ_floor = 0.05, a_min = 1, so every decay factor is ≤ 0.951 for every
  input and every set of weights. No-margin variant (`chain_mamba_no_margin.yaml`): δ_floor = a_min = 0 and
  the standard Δ range 0.001–0.1, which is exactly the plain certificate.
- The step-by-step and parallel forward passes agree to about 1e-16, and gradient reaches every parameter.
  The margin holds after random weight perturbation (`tests/test_chain_mamba.py`; `tools/check_model.py`
  passes for all arms).
- Old models and configs are unchanged: every new read-out and training option is off by default.

## 6. Training recipe (`configs/experiments/exp03_chain_mamba.yaml`, `&recipe`)

```
epochs 80, batch 128 (5,040 updates; exp01d: 1,250), Adam, peak learning rate 0.002, 3 warm-up epochs,
cosine decay to 1% · gradient clipping 1.0 · weight decay 1e-5 (as before)
chained loss over a horizon that grows linearly from 5 steps (epoch 1) to 50 steps (epoch 25)
+ 0.5 × one-step teacher-forced loss over the whole window · state noise 0 · model kept = best validation
(validation = full 50-step chained loss, unchanged)
```

### Mathematical signal (written before the runs, in the config)

```
Mathematical signal: POSITIVE (margin and anchor) / INCONCLUSIVE (size of the gain)
What was checked:    linearised chained-error recursion with one memory channel (analysis/a05)
Result:              growth per step = largest eigenvalue of [[1, c], [c, rho]]: 1 + c with no margin
                     (rho -> 1), about 1 + c^2/(1 - rho) with margin; c = 0.05: 1.0495 (rho 0.999) vs
                     1.0312 (rho 0.951); 50-step accumulation of a constant mistake 113 vs 88 (no leak: 50)
Assumptions:         linearisation; coupling c similar in both models (not measured)
Implication:         predicted the margin would lower 50-step error. It did NOT (Section 7): chained training
                     keeps the coupling c small by itself, so the slow memory does no harm once trained.
```

---

## 7. Results (test windows, physical units, smaller is better)

Seed 0. Baselines are the saved exp01d checkpoints, re-evaluated here (numbers reproduced to four digits).

| Model | Training | Params | 1-step | 10-step | 50-step (at step) | **50-step (average)** | vs full Mamba | Limit cycle amplitude / period error |
|---|---|---|---|---|---|---|---|---|
| Full Mamba (exp01d) | chained, old | 3,282 | 0.0939 | 0.2668 | 0.6261 | **0.4650** | — | 38.4% / 24.1% |
| LSTM (exp01d) | chained, old | 3,152 | 0.1184 | 0.1824 | 0.3620 | **0.2671** | −43% | 2.5% / 0.3% |
| Chain-Mamba (margin) | chained, old | 3,310 | 0.0711 | 0.2136 | 0.4092 | **0.3331** | −28% | 11.8% / 2.6% |
| LSTM | recipe | 3,152 | 0.0346 | 0.0680 | 0.1324 | **0.0980** | −79% | 2.5% / 0.2% |
| **Chain-Mamba (margin) — declared proposed** | recipe | 3,310 | **0.0135** | 0.0354 | 0.0948 | **0.0664** | **−86%** | 0.5% / 2.4% |
| Chain-Mamba, no margin — declared control | recipe | 3,310 | 0.0133 | 0.0304 | 0.0772 | **0.0539** | −88% | 6.9% / 0.8% |
| Full Mamba — declared control | recipe | 3,282 | 0.0369 | 0.1076 | 0.2160 | **0.1762** | −62% | 13.8% / 10.1% |

Seed 1 (robustness check, recipe arms only):

| Model | 1-step | 50-step (at step) | **50-step (average)** | Limit cycle amplitude / period error |
|---|---|---|---|---|
| Chain-Mamba (margin) | 0.0126 | 0.0823 | **0.0567** | 0.8% / 0.9% |
| Chain-Mamba, no margin | 0.0125 | 0.0800 | **0.0557** | 1.0% / 0.7% |
| LSTM | 0.0260 | 0.1704 | **0.1198** | 1.9% / 3.4% |

Over the two seeds, Chain-Mamba is 32% and 53% below the LSTM trained the same way. The no-margin variant is
19% and 2% below the margin variant.

Figures (`tools/compare_runs.py`): `docs/evidence/exp03_chain_mamba/error_vs_step.png`, `example_window.png`,
`free_running.png`.

**What this means**
- **The declared rule passes.** The proposed arm is 86% below the baseline (the bar was 20%). Its one-step
  error is 86% lower, not higher. Nothing blew up, and the certificate holds (largest decay value −1.55).
- **Both the recipe and the architecture matter.** With the old recipe, the architecture alone gives −28%
  (0.465 → 0.333). With the new recipe, Chain-Mamba is 32% below the LSTM trained the same way
  (0.066 vs 0.098), and its one-step error is 61% lower (0.0135 vs 0.0346). With the recipe alone, the
  unchanged full Mamba reaches 0.176 (−62%). Chain-Mamba is 62% below that, at the same parameter count.
  The recipe-trained full Mamba still has a slow memory (largest decay 0.996–0.999) and a poor limit cycle
  (13.8% / 10.1%).
- **The memory margin is not the reason it works.** Without the margin the 50-step error was 19% lower on
  seed 0 (0.054 vs 0.066) and about the same on seed 1 (0.056 vs 0.057), even though that model's memory is slow again (largest decay 0.9986, memory gain about
  700). This disagrees with the mathematical prediction. Once chained training has run, the slow memory does
  no harm, presumably because training keeps the coupling small. This matches the Stage 1 finding that the
  plain certificate is non-binding. The margin still gives a hard guarantee (ρ ≤ 0.951) and a better
  limit-cycle amplitude (0.5% vs 6.9%), but a worse period (2.4% vs 0.8%).
- Chain-Mamba's error still grows about 7× from step 1 to step 50. The true system itself magnifies
  errors (median 2.5× over 50 steps), and the model's own magnification is 1.9× (margin) and 2.3× (no margin).
- Training cost on the processor: 27 minutes (Chain-Mamba with margin), 22 minutes (no margin), 8 minutes
  (LSTM), 70 minutes (full Mamba). One chained update of Chain-Mamba is about 2.2 times faster than one of the full Mamba (0.76 s vs
  1.65 s at batch 256, horizon 50).

---

## 7b. Follow-up (3 November 2026, evening): memory-free controls and the four-tank system, seed 0

Two checks were requested: (1) does a network with **no memory** do as well? (2) does it work on another
system? For (2) the four-tank system from Cevaal et al. (Section VI.B, Equation 33, Table 5) was added to the
code. The new files are `src/systems/four_tank.py` and `configs/systems/four_tank.yaml`. It uses 4 measured
levels, 2 pump inputs in [0, 4], a 5 s sample time, and a piecewise-constant input held for 10 to 60
samples. There are 40,000 training samples (Cevaal et al. used 80,000). The windows, horizon (50 steps =
250 s), recipe and evaluation are the same as for Van der Pol. Configs: `exp04_mlp_control.yaml` and
`exp05_four_tank.yaml`.

Two memory-free controls, both with the same anchor, read-out scaling and recipe:
- **MLP**: a plain multilayer perceptron (2 SiLU layers), 3,358 / 3,412 parameters.
- **Chain-Mamba, memory removed**: Chain-Mamba with the selective state-space mixer switched off
  (`mixer: false`), leaving residual SwiGLU layers only, 3,286 / 3,368 parameters. This arm was added
  *after* seeing the MLP result, to find out what explains the gap.

**Van der Pol** (50-step chained error, average over the horizon; smaller is better)

| Model | 1-step | 50-step (at step) | 50-step (average) |
|---|---|---|---|
| Full Mamba, old training (exp01d) | 0.0939 | 0.6261 | 0.4650 |
| MLP + recipe | 0.0206 | 0.1357 | 0.1021 |
| LSTM + recipe | 0.0346 | 0.1324 | 0.0980 |
| Chain-Mamba + recipe | 0.0135 | 0.0948 | 0.0664 |
| Chain-Mamba, no margin + recipe | 0.0133 | 0.0772 | 0.0539 |
| **Chain-Mamba, memory removed + recipe** | **0.0116** | **0.0691** | **0.0484** |

**Four-tank** (levels in metres; level standard deviation about 0.4 m)

| Model | Params | 1-step | 50-step (at step) | 50-step (average) | vs full Mamba |
|---|---|---|---|---|---|
| Full Mamba, old training | 3,324 | 0.00610 | 0.02835 | 0.02200 | — |
| MLP + recipe | 3,412 | 0.00036 | 0.00556 | 0.00437 | −80% |
| LSTM + recipe | 3,504 | 0.00060 | 0.00425 | 0.00337 | −85% |
| Chain-Mamba + recipe | 3,392 | 0.00023 | 0.00309 | 0.00242 | −89% |
| Chain-Mamba, no margin + recipe | 3,392 | 0.00023 | 0.00310 | 0.00243 | −89% |
| **Chain-Mamba, memory removed + recipe** | 3,368 | **0.00021** | **0.00295** | **0.00229** | **−90%** |

Figures: `docs/evidence/exp04_mlp_control/` and `docs/evidence/exp05_four_tank/` (`error_vs_step.png`,
`example_window.png`).

**What this means**
- **It carries over to a second system.** On the four-tank system the ordering is the same as on Van der Pol.
  Chain-Mamba is 89% below the full Mamba trained the old way and 28% below the LSTM trained the same way.
  The margin makes no difference there (0.00242 vs 0.00243).
- **A plain MLP does not match it** (Van der Pol 0.102 vs 0.066; four-tank 0.0044 vs 0.0024). So the
  prediction in exp04 (MLP within 20%) did not hold.
- **But Chain-Mamba without its memory matches or beats Chain-Mamba on both systems** (0.048 vs 0.066;
  0.0023 vs 0.0024). The gain comes from the residual SwiGLU layers, the anchor, the output scaling and the
  recipe, **not** from Mamba's selective memory. This is expected for these two systems. Both are fully
  measured and Markov (the next state depends only on the current state and input), so the best chained
  predictor needs no memory at all.
- The four-tank system damps its own errors (true 50-step error magnification 0.52; Van der Pol 2.5).
  Errors there are small for every model, and the differences between the recipe-trained models are
  smaller than on Van der Pol.
- One seed. The seed-to-seed spread seen on Van der Pol (0.066 vs 0.057 for the same arm) is about 15%,
  so "memory removed is as good as Chain-Mamba" is safe to say, but "better" is not shown.

**Consequence for a paper.** "Mamba works for chained prediction" is not supported as a Mamba claim on
fully measured systems: the same network without the Mamba layer is as good. What *is* supported on two
systems is:
- training on chained predictions with a horizon curriculum, plus a state anchor and gated feed-forward
  layers, cuts chained error by about 90% at a matched parameter count;
- it beats a parameter-matched LSTM given the same training;
- the stability guarantee, with or without a margin, plays no part.

For Mamba's memory to earn its place, the test must be a system where memory is *needed*: only some states
measured (for example Van der Pol with x1 only, or the Quanser Aero2 outputs), or delays. That is now the
deciding experiment.

---

## 8. Honest limits

1. Two seeds for the recipe arms, one for the controls. Nothing here is significant until seeds 0–7.
2. The margin-free variant was better, but it was a control arm. Making it the new default after seeing
   the result is the kind of choice the project rules forbid without a new declared test. Step A below runs
   both variants at 8 seeds with the rule written first.
3. The recipe was set once, before any result, and was not tuned. It may not be the best recipe for either
   network.
4. A model trained for chained prediction is useless for direct multi-step prediction (2.56). That is
   expected, and Cevaal-style direct prediction needs its own training.
5. Processor numbers may differ slightly from the T4 graphics card's. Re-run seed 0 on Kaggle first.

---

## 9. Plan from here (each step: run → read → decide; rules fixed before running)

**Step A — Confirm (8 seeds), about 4–6 hours on a T4.**
`chain_mamba_recipe`, `chain_mamba_no_margin`, `lstm_recipe`, `full_mamba_recipe`, seeds 0–7.
Rule (written now): Chain-Mamba (either variant) is "better than the LSTM" only if the median 50-step error
is lower with Mann–Whitney p < 0.05 **and** |Cliff's delta| > 0.33. The margin question is answered by
`chain_mamba_recipe` vs `chain_mamba_no_margin` under the same test. If neither direction is significant,
report "the margin costs nothing detectable" and keep it for the guarantee.

**Step B — Which piece does what (one seed each, then 8 for any arm above 20%).**
Remove one at a time from Chain-Mamba with the recipe: anchor, increment normalisation, SwiGLU, short
kernel (back to 10), final normalisation (back on). This separates "Mamba with a feed-forward layer" from
"Mamba with a state anchor". Cost: about 25 minutes per arm on a processor.

**Step C — Closed loop.** Use Chain-Mamba as the predictor in MPC on Van der Pol: stabilisation from 100
initial states, tracking a piecewise-constant reference (Cevaal et al., Section VI.A), and noise at
signal-to-noise ratio 20. Compare it against the full Mamba and the LSTM, in both the chained form and
Cevaal's direct form. Warm-start the memory from the last 10 measured states. Pass if tracking error is
lower and solve time stays below the 0.1 s sampling time.

**Step D — Second system.** Done on seed 0 (Section 7b): it carries over. Eight seeds still needed.

**Step D2 — Partially measured system (now the deciding experiment for a Mamba-specific claim).** Van der
Pol with only x1 measured (the model sees x1 and u, predicts x1), and the four-tank system with only levels
1 and 2 measured (Cevaal et al.'s usual outputs). Arms: Chain-Mamba, Chain-Mamba with memory removed (given
the last 4 measurements as input, so it is not handicapped unfairly), LSTM, all with the recipe. Rule
(written now): Chain-Mamba must be at least 20% below the memory-removed control on both systems for the
memory to count as a contribution.

**Step E — Back to the research question.** The certificate (with or without margin) is not what makes
chained prediction work. Training on chained predictions plus a state anchor is. This supports the
"Paper B" line: the guarantee does not transfer, and here is what does. It also gives a positive design
result: a 3,300-parameter Mamba that beats a parameter-matched LSTM at chained prediction. Analysis a05
gives the algebra for when a margin *would* matter: a large coupling c that training does not suppress,
for example under teacher forcing.

---

## 10. How to run

```bash
# checks (seconds)
python tools/check_model.py --config configs/experiments/exp03_chain_mamba.yaml
pytest -q tests/test_chain_mamba.py
python analysis/a05_memory_margin_leak.py

# seed 0, all arms (T4: roughly 2-3 hours, most of it full_mamba_recipe; skip it with --arms)
python train.py \
    --config configs/experiments/exp03_chain_mamba.yaml \
    --seed 0

# only the main comparison (about 1 hour on a processor)
python train.py \
    --config configs/experiments/exp03_chain_mamba.yaml \
    --arms chain_mamba_recipe chain_mamba_no_margin lstm_recipe \
    --seed 0

# Step A: eight seeds
python train.py \
    --config configs/experiments/exp03_chain_mamba.yaml \
    --seeds 0 1 2 3 4 5 6 7

# figures and the side-by-side table (no training)
python tools/compare_runs.py \
    --runs "Full Mamba (exp01d)=models/seed0/exp01d_certificate_ablation_confirm/certified_chained_seed0" \
           "LSTM (exp01d)=models/seed0/exp01d_certificate_ablation_confirm/lstm_chained_seed0" \
           "Chain-Mamba=outputs/exp03_chain_mamba/runs/chain_mamba_recipe_seed0" \
           "LSTM + recipe=outputs/exp03_chain_mamba/runs/lstm_recipe_seed0" \
    --out outputs/exp03_chain_mamba/comparison

# how slowly does each memory fade?
python tools/diagnose_memory_margin.py \
    --runs outputs/exp03_chain_mamba/runs/chain_mamba_recipe_seed0 \
           outputs/exp03_chain_mamba/runs/chain_mamba_no_margin_seed0
```

## Sources

- Cevaal, de Jong, Lazar — Mamba Sequence Modeling meets Model Predictive Control (2026): https://arxiv.org/abs/2604.13857
- Bilinear Input Modulation for Mamba (Mamba-X) (2026): https://arxiv.org/abs/2604.17221
- Mamba-3: Improved Sequence Modeling using State Space Principles (2026): https://arxiv.org/abs/2603.15569
- Beintema, Schoukens, Tóth — Deep subspace encoders for nonlinear system identification: https://deepai.org/publication/deep-subspace-encoders-for-nonlinear-system-identification
- Beintema, Schoukens, Tóth — Continuous-time identification of dynamic state-space models by deep subspace encoding (ICLR 2023): https://iclr.cc/virtual/2023/poster/11475
- Brandstetter, Worrall, Welling — Message Passing Neural PDE Solvers (pushforward trick) (2022): https://arxiv.org/abs/2202.03376
- Autoregressive rollout error in latent-space reduced-order models of bluff-body wakes is accumulated phase drift (2026): https://arxiv.org/abs/2608.07189
- PhyxMamba — Mamba Integrated with Physics Principles Masters Long-term Chaotic System Forecasting (2025): https://arxiv.org/abs/2505.23863
- Mamba-Assisted Non-Markovian Closure for Reduced-Order Modeling (2026): https://arxiv.org/abs/2606.05371
- Bonassi, Andersson, Mattsson, Schön — Structured state-space models are deep Wiener models: https://arxiv.org/abs/2312.06211
- Hu et al. — State-space models are accurate and efficient neural operators for dynamical systems: https://arxiv.org/abs/2409.03231
- MamKO: Mamba-based Koopman operator for modeling and predictive control (ICLR 2025): https://iclr.cc/virtual/2025/poster/28767
