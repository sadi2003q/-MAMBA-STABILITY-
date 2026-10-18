RESULT — exp01d_certificate_ablation_confirm   (1 seed: direction only)

Question: Does removing the stability guarantee change how accurate the full Mamba's predictions are?
Smaller error is better. Errors are root-mean-square, in the system's own units.

Trained with the true previous state (teacher forcing); tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin           0.0339           0.383          no
  Guarantee removed (full Mamba)               0.0341            0.39          no
  Long Short-Term Memory network (re           0.0162           0.193          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +2% against the baseline → about the same (next-step error +1%).
    • Long Short-Term Memory network (reference): 50-step error -50% against the baseline → clearly better (on this seed) (next-step error -52%).
    • Baseline errors grow about 17× between 1 step and 50 steps.

Trained on its own chained predictions; tested on chained prediction
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin           0.0939           0.465          no
  Guarantee removed (full Mamba)                0.101           0.483          no
  Long Short-Term Memory network (re            0.118           0.267          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +4% against the baseline → about the same (next-step error +7%).
    • Long Short-Term Memory network (reference): 50-step error -43% against the baseline → clearly better (on this seed) (next-step error +26%).
    • Baseline errors grow about 7× between 1 step and 50 steps.

Direct multi-step prediction (whole horizon at once, nothing fed back)
                                      Next-step error   50-step error    Blew up?
  Guarantee on (full Mamba) (baselin              0.3            1.05          no
  Guarantee removed (full Mamba)                0.297            1.05          no
  Long Short-Term Memory network (re            0.276           0.521          no
  What this means:
    • Guarantee removed (full Mamba): 50-step error +0% against the baseline → about the same (next-step error -1%).
    • Long Short-Term Memory network (reference): 50-step error -50% against the baseline → clearly better (on this seed) (next-step error -8%).
    • Baseline errors grow about 4× between 1 step and 50 steps.

Math predicted: free_sign / certified error ratio has its bootstrap 95% interval inside [0.90, 1.10] in every training method; the decay stays negative in every free_sign run.
Result agreed with the prediction for every arm.

Decision: NOT worth confirming (no arm that trained differed by more than 20%). Blown-up arms are reported as counts, not errors.
