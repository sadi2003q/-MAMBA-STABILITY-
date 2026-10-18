RESULT — exp01b_certificate_ablation_full_mamba   (1 seed: direction only)

Question: Does removing the stability guarantee matter in the full Mamba architecture?
Smaller error is better. Errors are root-mean-square, in the system's own units.

Trained with the true previous state (teacher forcing); tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin           0.0339           0.383          no
  Guarantee removed (full Mamba)               0.0341            0.39          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +2% against the baseline → about the same (next-step error +1%).
    • Baseline errors grow about 17× between 1 step and 50 steps.

Trained on its own chained predictions; tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin           0.0939           0.465          no
  Guarantee removed (full Mamba)                0.101           0.483          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +4% against the baseline → about the same (next-step error +7%).
    • Baseline errors grow about 7× between 1 step and 50 steps.

Direct multi-step prediction (whole horizon at once, nothing fed back)
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin              0.3            1.05          no
  Guarantee removed (full Mamba)                0.297            1.05          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +0% against the baseline → about the same (next-step error -1%).
    • Baseline errors grow about 4× between 1 step and 50 steps.

Math predicted: free_sign within 20% of certified in every regime.
Result agreed with the prediction for every arm.

Decision: NOT worth confirming (no arm that trained differed by more than 20%). Blown-up arms are reported as counts, not errors.
