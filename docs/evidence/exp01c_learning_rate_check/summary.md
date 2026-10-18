RESULT — exp01c_learning_rate_check   (1 seed: direction only)

Question: Was the full Mamba trained fairly, or did the learning rate 0.001 hold it back?
Smaller error is better. Errors are root-mean-square, in the system's own units.

Full Mamba
                                      Next-step error   50-step error    Blew up?
  mamba_chained_lr_1e-3                   not run yet
  Full Mamba, learning rate 0.0001 (            0.232           0.999          no
  Full Mamba, learning rate 0.0003 (            0.134           0.631          no

Long Short-Term Memory network
                                      Next-step error   50-step error    Blew up?
  lstm_chained_lr_1e-3                    not run yet
  Long Short-Term Memory network, le            0.299           0.885          no
  Long Short-Term Memory network, le            0.227           0.471          no

Math predicted: If the training settings are the cause, a lower learning rate (0.0003 or 0.0001) lowers the full Mamba's best validation loss clearly (more than 3%) below the value at 0.001, and the gap between its validation and training loss closes. If the design is the cause, the best validation loss stays about the same at every learning rate.

Decision: none here. Run tools/pick_learning_rate.py to apply the rule that was written down before this check.
