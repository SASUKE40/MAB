Setting: Off-policy & offline. Actions (K) = 5. Logged rounds (max n) = 1,000. Context dimension d = 5. Distinct contexts = 100. 100 runs, seed 7.
Ranked by rmse / suboptimality at max n (lower is better).

### Linear, good overlap

Linear mean rewards, so ridge reward models are well specified. The logging policy is a soft, noisy version of the truth; the target policy is sharper and closer to it.

| # | Algorithm | RMSE / subopt. | 95% CI | Median |err| / subopt. | Bias of V̂ | SD | At smallest n | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | LinLCB | 0.008 | 0.0011 | 0.007 | -0.061 | 0.005 | 0.112 | 0.85 |
| 2 | DM-greedy | 0.008 | 0.0012 | 0.007 | 0.012 | 0.006 | 0.122 | 0.31 |
| 3 | DM (ridge) | 0.026 | 0.0048 | 0.016 | -0.005 | 0.026 | 0.219 | 0.23 |
| 4 | SWITCH-DR | 0.029 | 0.0062 | 0.015 | 0.002 | 0.029 | 0.167 | 1.32 |
| 5 | DR | 0.029 | 0.0062 | 0.015 | 0.002 | 0.030 | 0.162 | 0.24 |
| 6 | MRDR | 0.030 | 0.0064 | 0.017 | 0.002 | 0.030 | 0.165 | 0.23 |
| 7 | SNIPS | 0.033 | 0.0076 | 0.020 | 0.002 | 0.034 | 0.128 | 0.02 |
| 8 | CAB | 0.046 | 0.0067 | 0.025 | 0.004 | 0.046 | 0.205 | 0.23 |
| 9 | NeuralLCB | 0.049 | 0.0046 | 0.048 | -0.098 | 0.023 | 0.254 | 13.72 |
| 10 | POEM | 0.054 | 0.0042 | 0.049 | 0.053 | 0.022 | 0.207 | 12.82 |
| 11 | Clipped IPS | 0.062 | 0.01 | 0.032 | -0.000 | 0.062 | 0.289 | 0.02 |
| 12 | IPS-greedy | 0.062 | 0.0063 | 0.057 | 0.139 | 0.032 | 0.324 | 0.33 |
| 13 | IPS | 0.064 | 0.01 | 0.032 | 0.006 | 0.064 | 0.294 | 0.02 |
| 14 | SVP | 0.075 | 0.0091 | 0.071 | 0.070 | 0.046 | 0.210 | 13.23 |
| 15 | PEVI / LCB (tabular) | 0.098 | 0.0053 | 0.096 | -0.094 | 0.027 | 0.350 | 0.27 |

### Non-linear rewards (misspecified model)

Mean rewards are a sine of a linear score plus an interaction term, so every linear reward model is biased. Overlap is as in the linear case.

| # | Algorithm | RMSE / subopt. | 95% CI | Median |err| / subopt. | Bias of V̂ | SD | At smallest n | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | MRDR | 0.034 | 0.0050 | 0.021 | 0.002 | 0.034 | 0.170 | 0.22 |
| 2 | SNIPS | 0.037 | 0.0056 | 0.023 | 0.001 | 0.037 | 0.149 | 0.01 |
| 3 | SWITCH-DR | 0.038 | 0.0051 | 0.025 | -0.001 | 0.038 | 0.176 | 1.23 |
| 4 | DR | 0.041 | 0.0064 | 0.027 | -0.001 | 0.041 | 0.173 | 0.24 |
| 5 | CAB | 0.050 | 0.0066 | 0.033 | -0.004 | 0.050 | 0.183 | 0.27 |
| 6 | Clipped IPS | 0.051 | 0.0072 | 0.034 | -0.002 | 0.051 | 0.251 | 0.01 |
| 7 | IPS | 0.058 | 0.01 | 0.036 | 0.004 | 0.058 | 0.257 | 0.01 |
| 8 | DM (ridge) | 0.066 | 0.0080 | 0.054 | -0.041 | 0.051 | 0.239 | 0.24 |
| 9 | NeuralLCB | 0.157 | 0.01 | 0.154 | -0.097 | 0.054 | 0.449 | 14.76 |
| 10 | PEVI / LCB (tabular) | 0.167 | 0.0070 | 0.160 | -0.118 | 0.036 | 0.521 | 0.23 |
| 11 | POEM | 0.246 | 0.01 | 0.241 | 0.070 | 0.075 | 0.424 | 13.85 |
| 12 | SVP | 0.272 | 0.02 | 0.272 | 0.103 | 0.091 | 0.427 | 13.95 |
| 13 | LinLCB | 0.333 | 0.03 | 0.327 | 0.027 | 0.137 | 0.397 | 0.69 |
| 14 | DM-greedy | 0.340 | 0.03 | 0.325 | 0.142 | 0.142 | 0.416 | 0.31 |
| 15 | IPS-greedy | 0.377 | 0.03 | 0.369 | 0.236 | 0.128 | 0.511 | 0.30 |

### Low overlap (near-deterministic logging)

The logging policy is almost deterministic (temperature 5, 5% uniform floor), and the target policy often prefers other actions, so importance weights reach about 20·K and many actions are rarely logged.

| # | Algorithm | RMSE / subopt. | 95% CI | Median |err| / subopt. | Bias of V̂ | SD | At smallest n | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | LinLCB | 0.015 | 0.0022 | 0.012 | -0.063 | 0.011 | 0.096 | 0.70 |
| 2 | DM-greedy | 0.018 | 0.0025 | 0.015 | 0.025 | 0.013 | 0.105 | 0.32 |
| 3 | DM (ridge) | 0.037 | 0.0054 | 0.024 | -0.015 | 0.034 | 0.275 | 0.22 |
| 4 | CAB | 0.040 | 0.0054 | 0.030 | -0.008 | 0.039 | 0.225 | 0.26 |
| 5 | SWITCH-DR | 0.046 | 0.0077 | 0.029 | -0.007 | 0.046 | 0.227 | 1.39 |
| 6 | DR | 0.049 | 0.0084 | 0.029 | -0.006 | 0.049 | 0.219 | 0.23 |
| 7 | MRDR | 0.059 | 0.01 | 0.037 | -0.004 | 0.060 | 0.265 | 0.23 |
| 8 | SNIPS | 0.061 | 0.0095 | 0.035 | -0.003 | 0.061 | 0.159 | 0.01 |
| 9 | NeuralLCB | 0.079 | 0.0090 | 0.076 | -0.049 | 0.046 | 0.231 | 14.71 |
| 10 | PEVI / LCB (tabular) | 0.082 | 0.0071 | 0.078 | -0.124 | 0.036 | 0.312 | 0.21 |
| 11 | IPS | 0.121 | 0.02 | 0.072 | 0.001 | 0.121 | 0.371 | 0.01 |
| 12 | SVP | 0.164 | 0.02 | 0.146 | 0.168 | 0.093 | 0.309 | 14.18 |
| 13 | POEM | 0.167 | 0.02 | 0.153 | 0.162 | 0.089 | 0.288 | 14.05 |
| 14 | IPS-greedy | 0.186 | 0.01 | 0.189 | 0.480 | 0.070 | 0.369 | 0.29 |
| 15 | Clipped IPS | 0.193 | 0.02 | 0.145 | -0.159 | 0.110 | 0.335 | 0.01 |

### Small sample (n ≤ 150, noisy)

Linear rewards with twice the noise and at most 150 logged rounds (whatever the T control says), grid from n = 10: shows which methods degrade gracefully when data are scarce.

| # | Algorithm | RMSE / subopt. | 95% CI | Median |err| / subopt. | Bias of V̂ | SD | At smallest n | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | LinLCB | 0.127 | 0.01 | 0.117 | -0.115 | 0.058 | 0.324 | 0.51 |
| 2 | DM (ridge) | 0.143 | 0.02 | 0.116 | -0.044 | 0.136 | 0.769 | 0.12 |
| 3 | SWITCH-DR | 0.148 | 0.02 | 0.126 | -0.005 | 0.148 | 0.664 | 0.28 |
| 4 | DR | 0.149 | 0.02 | 0.125 | -0.002 | 0.150 | 0.642 | 0.11 |
| 5 | SNIPS | 0.155 | 0.02 | 0.111 | 0.008 | 0.156 | 0.445 | 0.00 |
| 6 | MRDR | 0.163 | 0.02 | 0.134 | -0.004 | 0.164 | 0.646 | 0.12 |
| 7 | DM-greedy | 0.167 | 0.02 | 0.153 | 0.339 | 0.085 | 0.329 | 0.15 |
| 8 | CAB | 0.171 | 0.03 | 0.128 | -0.002 | 0.172 | 0.592 | 0.13 |
| 9 | Clipped IPS | 0.215 | 0.04 | 0.150 | 0.004 | 0.217 | 0.893 | 0.00 |
| 10 | IPS | 0.219 | 0.04 | 0.140 | 0.012 | 0.219 | 0.917 | 0.01 |
| 11 | POEM | 0.252 | 0.02 | 0.247 | 0.510 | 0.099 | 0.394 | 9.38 |
| 12 | IPS-greedy | 0.254 | 0.02 | 0.230 | 0.802 | 0.127 | 0.408 | 0.13 |
| 13 | NeuralLCB | 0.276 | 0.02 | 0.264 | -0.022 | 0.116 | 0.409 | 13.09 |
| 14 | SVP | 0.292 | 0.02 | 0.287 | 0.531 | 0.116 | 0.397 | 9.46 |
| 15 | PEVI / LCB (tabular) | 0.310 | 0.01 | 0.312 | -0.503 | 0.070 | 0.433 | 0.10 |

