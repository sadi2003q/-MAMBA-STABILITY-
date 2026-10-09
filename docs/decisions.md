# Decisions

Dated record of what was decided and why. Newest last.

---

## 9 October 2026 — Benchmark set-up for exp01 (matched to Cevaal, de Jong and Lazar, 2026)

- **Equation.** Standard Van der Pol: dx2/dt = mu (1 - x1^2) x2 - x1 + u, mu = 1. Cevaal et al.'s Equation 31 (as printed in the HTML version of arXiv:2604.13857v2) and our earlier brief both omit "- x1". Checked numerically: with the printed form and their multisine input (peak 15), the forward-Euler simulation blows up within 211 to 950 steps in every one of 10 trials; even without input it has no limit cycle. With "- x1" it stays finite (|x1| at most 3.2). Their stated limit-cycle behaviour only exists with "- x1", so we treat the omission as a typo.
- **Time step 0.1, forward Euler** — as Cevaal et al. a02 shows the true system's own 50-step error amplification at this step is up to 5.2 (median 2.25).
- **Input.** Multisine, 30 harmonics from 0.0049 to 4.88 hertz, random phases, peak 15 (Cevaal et al.). Harmonic spacing is not stated in their paper; **linear spacing** chosen because logarithmic spacing with peak 15 blows up the Euler simulation in 4 of 5 trials (too much low-frequency energy).
- **Data.** 40,000 training samples (Cevaal et al.), 10,000 validation, 10,000 test, each its own trajectory with its own phases and initial state drawn from x1 in [-2.5, 2.5], x2 in [-2.0, 2.0] (their evaluation range). Data fixed across model seeds (data_seed 12345).
- **Windows.** 10 context steps (true states, to warm up the memory) + 50 prediction steps, stride 5. Cevaal et al. predict 10 steps; we report error at step 10 as well so results stay comparable.
- **Training.** Adam, learning rate 1e-3, weight decay 1e-5 (Cevaal et al.), not applied to decay parameters, biases or normalisation weights; learning rate times 0.97 per epoch; gradient clipping at 1.0; 40 epochs (first attempt: curves flat by 30–50; Cevaal et al.'s 4,000 epochs is far beyond our budget); model with the lowest validation loss is evaluated.
- **Sizes.** Full Mamba: model dimension 8, state 8, kernel 10, 6 layers (Cevaal et al.), expansion 1 (not reported by them) → 3,282 parameters (−4.0% from 3,418). Long Short-Term Memory network: hidden size 25 → 3,152 (+3.3% from 3,052). Minimal cell: width 32, state 16 → 2,850 (mechanism model; no published reference).
- **Model input** at every step is [x1, x2, u, mask]; the mask marks empty state slots in direct multi-step prediction. All three regimes use the same network and the same residual readout x_{t+1} = x_t + W_out y_t.
- **Direct multi-step regime** (our version of Cevaal et al.'s design): after the context, the network sees only future inputs; the readout increments are summed to give the trajectory, and predictions are never fed back. Under this regime the augmented-Jacobian leak of a01 cannot occur, so it is the regime where the latent certificate should matter most.
- **Unstable start.** Decay values mirrored to +1 … +16 (scale 1.0). Mathematical signal: non-finite loss from the first batch (exp(1.6 × 60) exceeds the float32 range), so this arm is expected to diverge in every regime; it is kept because it is declared in the plan.
- **Primary metric.** Root-mean-square error averaged over the 50 prediction steps, physical units, both state components (`rollout_rmse`). Next-step error = error at prediction step 1 in the same regime. Diverged runs are reported as counts.
- **Decision rule for exp01.** Ablation rule (any trained ablation arm more than 20% from its baseline in the same training regime). Blown-up arms are reported as counts and do not count towards "worth confirming". For a null result at Stage 2, the free-sign / certified error ratio must have its bootstrap 95% interval inside [0.9, 1.1] (plan.md, Step 7N).
- **exp01b** runs the same ablation on the full Mamba architecture as a separate experiment, so the minimal-cell result can be read before spending the longer full-Mamba time.
