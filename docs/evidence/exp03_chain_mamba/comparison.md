Chained prediction on the test windows (smaller is better; root-mean-square error, physical units)

model                                         params   1-step  10-step  50-step  avg 1-50  vs first  cycle amp  cycle period
Full Mamba, old training (exp01d)               3282   0.0939   0.2668   0.6261    0.4650       +0%      38.4%         24.1%
LSTM, old training (exp01d)                     3152   0.1184   0.1824   0.3620    0.2671      -43%       2.5%          0.3%
Chain-Mamba + recipe                            3310   0.0135   0.0354   0.0948    0.0664      -86%       0.5%          2.4%
LSTM + recipe                                   3152   0.0346   0.0680   0.1324    0.0980      -79%       2.5%          0.2%
Full Mamba + recipe                             3282   0.0369   0.1076   0.2160    0.1762      -62%      13.8%         10.1%

cycle amp / cycle period: free-running limit cycle (zero input) against the true one, relative error.
