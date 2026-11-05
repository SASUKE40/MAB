Setting: Continuum-armed & GP. Horizon (T steps) = 500. Noise σ = 0.1. 20 runs, seed 7.
Ranked by cumulative regret (lower is better).

### Smooth unimodal (1-D)

f(x) = 0.05 + 0.9·exp(−(x−a)²/(2·0.15²)) with the peak a ~ U(0.2, 0.8). Every method should find it.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | StageOpt | 7.03 | 0.67 | 6.83 | 0.004 | 0.00 | 10.42 |
| 2 | UCT (dyadic tree) | 8.10 | 2.55 | 6.75 | 0.003 | 0.00 | 1.44 |
| 3 | GP-UCB | 8.14 | 0.75 | 7.63 | 0.001 | 0.00 | 9.05 |
| 4 | BKB (Nyström GP-UCB) | 8.52 | 0.65 | 8.29 | 0.002 | 0.00 | 4.44 |
| 5 | HOO | 10.2 | 0.79 | 10.3 | 0.002 | 0.00 | 2.47 |
| 6 | DOO | 11.2 | 4.72 | 6.54 | 0.016 | 0.00 | 1.26 |
| 7 | HCT | 14.0 | 1.31 | 14.3 | 0.002 | 0.00 | 0.64 |
| 8 | IGP-UCB | 14.9 | 0.54 | 15.0 | 0.001 | 0.00 | 9.23 |
| 9 | SafeOpt | 15.1 | 1.47 | 15.5 | 0.001 | 0.00 | 10.22 |
| 10 | Zooming | 16.1 | 0.63 | 16.0 | 0.003 | 0.00 | 3.60 |
| 11 | Contextual Zooming | 32.5 | 10.0 | 26.9 | 0.018 | 0.00 | 5.42 |
| 12 | Discretized UCB (Kleinberg) | 37.0 | 12.2 | 32.4 | 0.068 | 0.00 | 0.31 |
| 13 | POO (with HOO) | 40.0 | 2.03 | 39.7 | 0.006 | 0.00 | 1.37 |
| 14 | VHCT | 43.7 | 2.61 | 43.8 | 0.005 | 0.00 | 0.98 |
| 15 | SOO | 63.4 | 1.03 | 63.2 | 0.017 | 0.00 | 0.91 |
| 16 | GP-TS | 78.7 | 1.91 | 79.3 | 0.001 | 0.00 | 24.31 |
| 17 | Kernelized exp. weights (simplified) | 100 | 1.23 | 100 | 0.002 | 0.00 | 2.94 |
| 18 | BGD (two-point) | 103 | 0.25 | 103 | 0.022 | 0.00 | 0.25 |
| 19 | StoSOO | 168 | 19.6 | 190 | 0.020 | 0.00 | 2.08 |
| 20 | SCRiBLe-style (log-barrier) | 209 | 7.20 | 219 | 0.011 | 0.00 | 0.31 |
| 21 | FKM (one-point gradient) | 279 | 1.06 | 279 | 0.012 | 0.00 | 0.27 |
| 22 | Uniform random | 285 | 4.03 | 286 | 0.020 | 0.00 | 0.24 |

### Multimodal bumps (1-D)

A sum of four narrow Gaussian bumps (widths 0.025–0.06), one of height 0.9 and three decoys of height 0.45–0.75, at random positions.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | GP-UCB | 21.5 | 12.7 | 12.9 | 0.021 | 0.00 | 8.53 |
| 2 | StageOpt | 30.3 | 17.9 | 10.5 | 0.044 | 0.00 | 10.07 |
| 3 | BKB (Nyström GP-UCB) | 34.3 | 20.5 | 12.6 | 0.046 | 0.00 | 3.06 |
| 4 | IGP-UCB | 40.4 | 5.24 | 36.0 | 0.001 | 0.00 | 8.91 |
| 5 | SafeOpt | 41.2 | 19.6 | 19.6 | 0.053 | 0.00 | 9.83 |
| 6 | Zooming | 46.9 | 18.7 | 28.7 | 0.069 | 0.00 | 3.04 |
| 7 | DOO | 60.1 | 26.3 | 33.6 | 0.107 | 0.00 | 0.78 |
| 8 | HOO | 61.0 | 28.1 | 17.4 | 0.094 | 0.00 | 2.30 |
| 9 | HCT | 72.7 | 26.5 | 41.6 | 0.079 | 0.00 | 0.44 |
| 10 | SOO | 83.2 | 3.41 | 84.7 | 0.007 | 0.00 | 0.96 |
| 11 | UCT (dyadic tree) | 84.6 | 34.2 | 101 | 0.154 | 0.00 | 0.81 |
| 12 | Contextual Zooming | 97.9 | 28.6 | 103 | 0.131 | 0.00 | 3.18 |
| 13 | POO (with HOO) | 125 | 21.4 | 126 | 0.015 | 0.00 | 0.96 |
| 14 | GP-TS | 133 | 10.2 | 131 | 0.002 | 0.00 | 24.79 |
| 15 | VHCT | 143 | 23.6 | 142 | 0.087 | 0.00 | 0.42 |
| 16 | Discretized UCB (Kleinberg) | 149 | 45.5 | 153 | 0.293 | 0.00 | 0.24 |
| 17 | Kernelized exp. weights (simplified) | 171 | 9.75 | 170 | 0.006 | 0.00 | 1.62 |
| 18 | StoSOO | 186 | 23.4 | 181 | 0.022 | 0.00 | 0.98 |
| 19 | SCRiBLe-style (log-barrier) | 297 | 26.4 | 288 | 0.472 | 0.00 | 0.27 |
| 20 | Uniform random | 315 | 9.43 | 319 | 0.028 | 0.00 | 0.21 |
| 21 | BGD (two-point) | 330 | 33.9 | 361 | 0.301 | 0.00 | 0.21 |
| 22 | FKM (one-point gradient) | 387 | 25.6 | 408 | 0.635 | 0.00 | 0.23 |

### Needle in a haystack (1-D)

A broad decoy hill (peak 0.55) plus a needle of height +0.4 and half-width 0.015 elsewhere. Smoothness-based methods must keep exploring to find it.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | SOO | 83.9 | 4.67 | 82.1 | 0.046 | 0.00 | 0.76 |
| 2 | HOO | 87.1 | 10.0 | 93.3 | 0.123 | 0.00 | 2.71 |
| 3 | DOO | 93.4 | 8.22 | 90.9 | 0.178 | 0.00 | 0.84 |
| 4 | StageOpt | 94.8 | 5.11 | 93.5 | 0.176 | 0.00 | 9.40 |
| 5 | UCT (dyadic tree) | 95.0 | 5.20 | 89.0 | 0.182 | 0.00 | 0.83 |
| 6 | Discretized UCB (Kleinberg) | 96.9 | 6.40 | 94.8 | 0.187 | 0.00 | 0.21 |
| 7 | GP-UCB | 97.2 | 5.21 | 93.1 | 0.180 | 0.00 | 8.21 |
| 8 | BKB (Nyström GP-UCB) | 97.6 | 5.29 | 97.6 | 0.181 | 0.00 | 5.30 |
| 9 | Zooming | 101 | 9.34 | 101 | 0.169 | 0.00 | 3.21 |
| 10 | SafeOpt | 103 | 5.12 | 101 | 0.185 | 0.00 | 9.62 |
| 11 | Contextual Zooming | 106 | 7.96 | 111 | 0.154 | 0.00 | 5.63 |
| 12 | HCT | 107 | 5.26 | 102 | 0.184 | 0.00 | 0.48 |
| 13 | StoSOO | 107 | 8.51 | 106 | 0.073 | 0.00 | 1.47 |
| 14 | BGD (two-point) | 107 | 5.07 | 104 | 0.186 | 0.00 | 0.26 |
| 15 | IGP-UCB | 116 | 4.83 | 114 | 0.181 | 0.00 | 7.62 |
| 16 | POO (with HOO) | 123 | 5.07 | 117 | 0.182 | 0.00 | 0.77 |
| 17 | VHCT | 124 | 5.23 | 120 | 0.164 | 0.00 | 0.66 |
| 18 | SCRiBLe-style (log-barrier) | 129 | 5.16 | 126 | 0.189 | 0.00 | 0.20 |
| 19 | Kernelized exp. weights (simplified) | 135 | 5.24 | 131 | 0.196 | 0.00 | 2.07 |
| 20 | GP-TS | 137 | 4.93 | 133 | 0.180 | 0.00 | 21.93 |
| 21 | FKM (one-point gradient) | 141 | 4.47 | 139 | 0.189 | 0.00 | 0.18 |
| 22 | Uniform random | 152 | 5.04 | 147 | 0.121 | 0.00 | 0.20 |

### Branin, rescaled (2-D)

The Branin function on [0,1]², log-rescaled to f = 1 − ln(1 + (B − B_min)/10)/ln(1 + (B_max − B_min)/10) and randomly mirrored; three global optima with f* = 1.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | UCT (dyadic tree) | 20.8 | 4.02 | 19.8 | 0.017 | 0.00 | 0.82 |
| 2 | DOO | 22.6 | 5.94 | 25.2 | 0.028 | 0.00 | 0.81 |
| 3 | Discretized UCB (Kleinberg) | 32.0 | 0.28 | 32.2 | 0.053 | 0.00 | 0.44 |
| 4 | HOO | 38.9 | 1.22 | 38.7 | 0.014 | 0.00 | 2.20 |
| 5 | StageOpt | 43.7 | 1.78 | 43.1 | 0.008 | 0.00 | 22.17 |
| 6 | HCT | 43.7 | 1.83 | 44.3 | 0.009 | 0.00 | 0.62 |
| 7 | SOO | 53.6 | 1.51 | 53.0 | 0.013 | 0.00 | 0.65 |
| 8 | GP-UCB | 58.6 | 1.67 | 59.5 | 0.002 | 0.00 | 20.30 |
| 9 | BKB (Nyström GP-UCB) | 59.8 | 1.50 | 60.1 | 0.002 | 0.00 | 52.23 |
| 10 | Zooming | 69.4 | 1.95 | 68.1 | 0.012 | 0.00 | 3.94 |
| 11 | SafeOpt | 70.5 | 4.47 | 71.4 | 0.026 | 0.00 | 22.43 |
| 12 | POO (with HOO) | 88.0 | 1.54 | 88.3 | 0.022 | 0.00 | 0.98 |
| 13 | Contextual Zooming | 89.1 | 9.05 | 85.8 | 0.030 | 0.00 | 12.80 |
| 14 | VHCT | 91.6 | 1.42 | 90.8 | 0.020 | 0.00 | 0.49 |
| 15 | BGD (two-point) | 92.3 | 1.28 | 93.5 | 0.140 | 0.00 | 0.29 |
| 16 | IGP-UCB | 123 | 0.71 | 123 | 0.016 | 0.00 | 17.86 |
| 17 | StoSOO | 126 | 15.4 | 136 | 0.018 | 0.00 | 0.90 |
| 18 | SCRiBLe-style (log-barrier) | 146 | 4.93 | 145 | 0.198 | 0.00 | 0.30 |
| 19 | Kernelized exp. weights (simplified) | 154 | 1.71 | 153 | 0.051 | 0.00 | 4.11 |
| 20 | FKM (one-point gradient) | 176 | 5.10 | 178 | 0.270 | 0.00 | 0.28 |
| 21 | GP-TS | 182 | 0.79 | 182 | 0.018 | 0.00 | 44.79 |
| 22 | Uniform random | 222 | 2.70 | 222 | 0.042 | 0.00 | 0.21 |

### Concave quadratic (1-D, convex loss)

f(x) = 0.9 − 1.2(x − a)² with a ∈ [0.15, 0.3] ∪ [0.7, 0.85]: a convex loss for bandit convex optimisation (FKM, BGD, SCR) started at x = 0.5.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | UCT (dyadic tree) | 5.04 | 0.77 | 4.60 | 0.002 | 0.00 | 0.65 |
| 2 | StageOpt | 6.13 | 0.80 | 6.24 | 0.007 | 0.00 | 9.78 |
| 3 | Discretized UCB (Kleinberg) | 6.64 | 1.10 | 6.22 | 0.008 | 0.00 | 0.17 |
| 4 | DOO | 6.92 | 1.76 | 6.90 | 0.006 | 0.00 | 0.68 |
| 5 | GP-UCB | 7.60 | 0.65 | 7.75 | 0.004 | 0.00 | 8.92 |
| 6 | BKB (Nyström GP-UCB) | 7.86 | 0.98 | 7.47 | 0.003 | 0.00 | 4.71 |
| 7 | BGD (two-point) | 9.90 | 0.36 | 9.70 | 0.006 | 0.00 | 0.15 |
| 8 | SafeOpt | 10.9 | 1.36 | 11.2 | 0.003 | 0.00 | 9.72 |
| 9 | HOO | 11.1 | 0.52 | 10.9 | 0.004 | 0.00 | 1.91 |
| 10 | HCT | 12.0 | 0.59 | 11.9 | 0.004 | 0.00 | 0.41 |
| 11 | Zooming | 13.0 | 0.47 | 13.0 | 0.004 | 0.00 | 2.85 |
| 12 | IGP-UCB | 17.4 | 0.77 | 17.5 | 0.003 | 0.00 | 8.66 |
| 13 | POO (with HOO) | 22.3 | 0.67 | 22.0 | 0.006 | 0.00 | 0.85 |
| 14 | Contextual Zooming | 22.4 | 3.13 | 21.7 | 0.012 | 0.00 | 4.54 |
| 15 | SOO | 23.1 | 0.94 | 22.9 | 0.008 | 0.00 | 0.60 |
| 16 | VHCT | 25.7 | 0.95 | 25.5 | 0.003 | 0.00 | 0.36 |
| 17 | SCRiBLe-style (log-barrier) | 32.3 | 3.36 | 31.6 | 0.020 | 0.00 | 0.18 |
| 18 | StoSOO | 39.0 | 7.13 | 36.0 | 0.012 | 0.00 | 0.77 |
| 19 | Kernelized exp. weights (simplified) | 39.9 | 1.19 | 39.8 | 0.007 | 0.00 | 1.64 |
| 20 | GP-TS | 41.7 | 0.98 | 42.3 | 0.004 | 0.00 | 24.47 |
| 21 | FKM (one-point gradient) | 44.9 | 2.94 | 45.0 | 0.029 | 0.00 | 0.16 |
| 22 | Uniform random | 99.4 | 5.77 | 98.3 | 0.011 | 0.00 | 0.16 |

### Safe optimisation (1-D, h = 0.35)

A main bump (0.9) and a secondary bump (0.65) separated by a valley below the safety threshold h = 0.35. Evaluations with f(x) < h count as unsafe; a safe seed near the main bump is given.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | UCT (dyadic tree) | 7.79 | 0.77 | 7.53 | 0.004 | 2.25 | 0.73 |
| 2 | BKB (Nyström GP-UCB) | 7.88 | 0.43 | 7.90 | 0.002 | 4.80 | 3.22 |
| 3 | GP-UCB | 7.93 | 0.64 | 7.58 | 0.002 | 4.55 | 8.55 |
| 4 | HOO | 11.0 | 0.93 | 10.6 | 0.003 | 4.10 | 1.94 |
| 5 | HCT | 12.0 | 1.24 | 11.9 | 0.003 | 3.75 | 0.54 |
| 6 | Discretized UCB (Kleinberg) | 13.2 | 4.33 | 11.2 | 0.021 | 3.15 | 0.28 |
| 7 | Zooming | 14.9 | 0.71 | 14.7 | 0.001 | 5.95 | 2.80 |
| 8 | IGP-UCB | 16.1 | 0.58 | 15.9 | 0.000 | 6.75 | 8.77 |
| 9 | DOO | 19.3 | 12.5 | 8.45 | 0.032 | 2.85 | 0.58 |
| 10 | Contextual Zooming | 38.1 | 13.6 | 29.0 | 0.022 | 14.8 | 3.43 |
| 11 | VHCT | 42.9 | 2.33 | 41.6 | 0.004 | 24.1 | 0.41 |
| 12 | POO (with HOO) | 44.4 | 1.33 | 44.9 | 0.007 | 27.6 | 0.82 |
| 13 | StageOpt | 54.4 | 2.85 | 53.5 | 0.002 | 1.30 | 19.30 |
| 14 | SOO | 58.9 | 0.64 | 58.9 | 0.009 | 41.7 | 0.60 |
| 15 | GP-TS | 84.9 | 1.64 | 85.6 | 0.000 | 56.8 | 23.83 |
| 16 | Kernelized exp. weights (simplified) | 107 | 1.74 | 107 | 0.002 | 74.5 | 1.50 |
| 17 | SafeOpt | 134 | 16.6 | 143 | 0.003 | 5.20 | 20.95 |
| 18 | BGD (two-point) | 141 | 0.20 | 141 | 0.033 | 21.3 | 0.19 |
| 19 | StoSOO | 160 | 15.8 | 172 | 0.015 | 147 | 1.02 |
| 20 | SCRiBLe-style (log-barrier) | 228 | 6.77 | 227 | 0.009 | 106 | 0.19 |
| 21 | Uniform random | 252 | 3.02 | 251 | 0.027 | 254 | 0.18 |
| 22 | FKM (one-point gradient) | 329 | 2.01 | 330 | 0.012 | 462 | 0.21 |

### Contextual peak (1-D, context c)

Each step reveals c ~ U(0,1); f(x, c) = 0.05 + 0.9·exp(−(x − m(c))²/(2·0.1²)), m(c) = 0.5 + 0.3 sin(2πc + φ). Only Contextual Zooming uses c; the others chase the best fixed x.

| # | Algorithm | Regret | 95% CI | Median | Simple regret | Unsafe evals | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | Contextual Zooming | 176 | 3.80 | 174 | 0.069 | 0.00 | 5.49 |
| 2 | UCT (dyadic tree) | 294 | 5.84 | 294 | 0.592 | 0.00 | 0.96 |
| 3 | Discretized UCB (Kleinberg) | 298 | 7.00 | 298 | 0.583 | 0.00 | 0.24 |
| 4 | StageOpt | 300 | 7.27 | 301 | 0.627 | 0.00 | 11.82 |
| 5 | BKB (Nyström GP-UCB) | 301 | 4.22 | 305 | 0.576 | 0.00 | 5.47 |
| 6 | Zooming | 302 | 3.67 | 304 | 0.594 | 0.00 | 3.32 |
| 7 | HOO | 303 | 4.56 | 300 | 0.586 | 0.00 | 2.13 |
| 8 | GP-UCB | 305 | 6.10 | 303 | 0.613 | 0.00 | 10.51 |
| 9 | HCT | 305 | 3.92 | 306 | 0.600 | 0.00 | 0.51 |
| 10 | SCRiBLe-style (log-barrier) | 307 | 3.98 | 310 | 0.602 | 0.00 | 0.23 |
| 11 | IGP-UCB | 310 | 3.53 | 310 | 0.581 | 0.00 | 10.57 |
| 12 | SOO | 312 | 3.49 | 311 | 0.602 | 0.00 | 0.73 |
| 13 | DOO | 312 | 8.90 | 310 | 0.625 | 0.00 | 0.77 |
| 14 | VHCT | 316 | 3.84 | 316 | 0.592 | 0.00 | 0.45 |
| 15 | POO (with HOO) | 317 | 4.11 | 315 | 0.627 | 0.00 | 0.92 |
| 16 | SafeOpt | 319 | 12.8 | 313 | 0.638 | 0.00 | 12.16 |
| 17 | StoSOO | 320 | 5.28 | 321 | 0.611 | 0.00 | 1.01 |
| 18 | FKM (one-point gradient) | 321 | 3.46 | 323 | 0.616 | 0.00 | 0.23 |
| 19 | BGD (two-point) | 321 | 3.86 | 322 | 0.615 | 0.00 | 0.20 |
| 20 | Kernelized exp. weights (simplified) | 322 | 3.48 | 322 | 0.594 | 0.00 | 1.73 |
| 21 | GP-TS | 329 | 3.13 | 327 | 0.583 | 0.00 | 29.19 |
| 22 | Uniform random | 339 | 3.16 | 340 | 0.606 | 0.00 | 0.23 |

