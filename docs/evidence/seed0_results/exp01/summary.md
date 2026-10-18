RESULT — exp01_certificate_ablation   (1 seed: direction only)

Question: If we remove the built-in stability guarantee, do predictions get worse?
Smaller error is better. Errors are root-mean-square, in the system's own units.

Trained with the true previous state (teacher forcing); tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (baseline)                      0.0364           0.584          no
  Guarantee removed                            0.0373           0.619          no
  Started unstable                                  —               —         yes
  Long Short-Term Memory network (re           0.0162           0.193          no
  What this means:
    • Guarantee removed: 50-step error +6% against the baseline → about the same (next-step error +3%).
    • Started unstable blew up in 1 of 1 run(s); no error is reported for it.
    • Long Short-Term Memory network (reference): 50-step error -67% against the baseline → clearly better (on this seed) (next-step error -55%).
    • Baseline errors grow about 23× between 1 step and 50 steps.

Trained on its own chained predictions; tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (baseline)                        0.08           0.256          no
  Guarantee removed                            0.0451           0.209          no
  Started unstable                                  —               —         yes
  Long Short-Term Memory network (re            0.118           0.267          no
  What this means:
    • Guarantee removed: 50-step error -18% against the baseline → slightly better (next-step error -44%).
    • Started unstable blew up in 1 of 1 run(s); no error is reported for it.
    • Long Short-Term Memory network (reference): 50-step error +4% against the baseline → about the same (next-step error +48%).
    • Baseline errors grow about 4× between 1 step and 50 steps.

Direct multi-step prediction (whole horizon at once, nothing fed back)
                                      Next-step error   50-step error    Blew up?
  Guarantee on (baseline)                       0.522            1.09          no
  Guarantee removed                             0.554             1.1          no
  Started unstable                                  —               —         yes
  Long Short-Term Memory network (re            0.276           0.521          no
  What this means:
    • Guarantee removed: 50-step error +1% against the baseline → about the same (next-step error +6%).
    • Started unstable blew up in 1 of 1 run(s); no error is reported for it.
    • Long Short-Term Memory network (reference): 50-step error -52% against the baseline → clearly better (on this seed) (next-step error -47%).
    • Baseline errors grow about 3× between 1 step and 50 steps.

Math predicted: free_sign within 20% of certified in every training regime; unstable_init diverges in every training regime.
Result agreed with the prediction for every arm.

Decision: NOT worth confirming (no arm that trained differed by more than 20%). Blown-up arms are reported as counts, not errors.
