Chained prediction on the test windows (smaller is better; root-mean-square error, physical units)

model                                         params   1-step  10-step  50-step  avg 1-50  vs first  cycle amp  cycle period
Full Mamba, old training (exp01d)               3282   0.0939   0.2668   0.6261    0.4650       +0%      38.4%         24.1%
MLP + recipe                                    3358   0.0206   0.0666   0.1357    0.1021      -78%       2.1%          0.5%
Chain-Mamba + recipe                            3310   0.0135   0.0354   0.0948    0.0664      -86%       0.5%          2.4%
LSTM + recipe                                   3152   0.0346   0.0680   0.1324    0.0980      -79%       2.5%          0.2%
Chain-Mamba, memory removed                     3286   0.0116   0.0260   0.0691    0.0484      -90%       1.2%          1.4%

cycle amp / cycle period: free-running limit cycle (zero input) against the true one, relative error.
