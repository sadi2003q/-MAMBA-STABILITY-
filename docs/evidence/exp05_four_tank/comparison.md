Chained prediction on the test windows (smaller is better; root-mean-square error, physical units)

model                                         params   1-step  10-step  50-step  avg 1-50  vs first  cycle amp  cycle period
Full Mamba, old training                        3324   0.0061   0.0166   0.0283    0.0220       +0%          -             -
MLP + recipe                                    3412   0.0004   0.0028   0.0056    0.0044      -80%          -             -
Chain-Mamba + recipe                            3392   0.0002   0.0016   0.0031    0.0024      -89%          -             -
LSTM + recipe                                   3504   0.0006   0.0024   0.0043    0.0034      -85%          -             -
Chain-Mamba, memory removed                     3368   0.0002   0.0015   0.0029    0.0023      -90%          -             -

cycle amp / cycle period: free-running limit cycle (zero input) against the true one, relative error.
