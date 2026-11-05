K = 10 arms, T = 1,000 steps, 200 runs per cell, seed 7.
Regret is mean cumulative pseudo-regret at T, ± a 95% confidence interval. "Best arm" is the share of the last 10% of steps spent on the best arm.

### Bernoulli, uniform means

Each arm pays 1 with probability μ ~ U(0, 1), else 0. Gaps are usually large.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Knowledge gradient | 17.4 | 3.95 | 76% | 1.31 |
| 2 | LinUCB | 18.2 | 2.65 | 83% | 0.24 |
| 3 | Information-directed sampling | 20.1 | 2.76 | 86% | 98.37 |
| 4 | Gittins index (approx.) | 20.3 | 2.55 | 87% | 0.94 |
| 5 | Thompson sampling | 28.8 | 1.95 | 86% | 1.82 |
| 6 | KL-UCB+ | 30.7 | 1.46 | 89% | 5.51 |
| 7 | IMED | 31.8 | 1.60 | 90% | 0.51 |
| 8 | KL-UCB | 31.8 | 1.89 | 88% | 5.23 |
| 9 | KL-UCB++ | 32.0 | 1.51 | 93% | 4.28 |
| 10 | Boltzmann–Gumbel | 33.6 | 1.18 | 86% | 0.76 |
| 11 | UCB1-Tuned | 34.4 | 1.41 | 81% | 0.25 |
| 12 | Bayes-UCB | 36.4 | 1.66 | 84% | 0.31 |
| 13 | Perturbed-history exploration | 38.7 | 1.69 | 81% | 1.94 |
| 14 | MOSS | 47.2 | 1.36 | 91% | 0.32 |
| 15 | MOSS-anytime | 50.0 | 1.37 | 84% | 0.31 |
| 16 | Decaying ε-greedy | 50.7 | 3.32 | 82% | 0.10 |
| 17 | Sequential halving + commit | 51.4 | 3.37 | 70% | 0.07 |
| 18 | Pursuit | 56.8 | 1.96 | 88% | 0.12 |
| 19 | ε-greedy | 69.0 | 4.36 | 76% | 0.09 |
| 20 | Tsallis-INF | 84.7 | 1.89 | 72% | 5.81 |
| 21 | Softmax (Boltzmann) | 85.0 | 12.0 | 47% | 0.26 |
| 22 | Discounted Thompson | 85.1 | 4.14 | 64% | 1.87 |
| 23 | Explore-then-commit | 99.6 | 4.94 | 71% | 0.04 |
| 24 | Annealed softmax | 108 | 5.11 | 46% | 0.29 |
| 25 | UCB2 | 109 | 1.64 | 76% | 0.52 |
| 26 | Optimistic initial values | 113 | 2.47 | 79% | 0.07 |
| 27 | Sliding-window UCB | 125 | 2.46 | 56% | 0.18 |
| 28 | ε-greedy, constant step | 126 | 10.6 | 55% | 0.10 |
| 29 | EXP3-IX | 128 | 2.91 | 66% | 0.23 |
| 30 | UCB1 | 130 | 2.62 | 56% | 0.19 |
| 31 | Monitored UCB | 131 | 2.61 | 57% | 0.39 |
| 32 | Reinforcement comparison | 132 | 2.73 | 75% | 0.25 |
| 33 | Gradient bandit | 137 | 2.83 | 76% | 0.29 |
| 34 | Linear reward-inaction | 169 | 5.64 | 54% | 0.11 |
| 35 | EXP3 | 184 | 5.36 | 51% | 0.30 |
| 36 | UCB-V | 231 | 5.33 | 41% | 0.22 |
| 37 | EXP3.S | 232 | 5.89 | 36% | 0.19 |
| 38 | Discounted UCB | 238 | 5.99 | 27% | 0.21 |
| 39 | Greedy | 245 | 31.8 | 20% | 0.15 |
| 40 | Rexp3 | 282 | 8.68 | 26% | 0.30 |
| 41 | Improved UCB | 283 | 6.69 | 29% | 0.07 |
| 42 | UCB1-Normal | 291 | 8.10 | 32% | 0.13 |
| 43 | EXP3.P | 334 | 9.57 | 18% | 0.29 |
| 44 | Successive elimination | 396 | 10.1 | 13% | 0.13 |
| 45 | Random | 419 | 11.4 | 10% | 0.08 |

### Bernoulli, close arms

One arm pays 0.55; the rest draw μ ~ U(0.40, 0.52). Small gaps make exploration expensive.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Gittins index (approx.) | 42.8 | 2.60 | 57% | 0.78 |
| 2 | Decaying ε-greedy | 42.9 | 3.61 | 44% | 0.10 |
| 3 | LinUCB | 44.1 | 3.10 | 52% | 0.21 |
| 4 | Knowledge gradient | 44.2 | 4.16 | 40% | 1.17 |
| 5 | Information-directed sampling | 46.7 | 2.37 | 50% | 97.12 |
| 6 | Pursuit | 47.0 | 3.15 | 49% | 0.15 |
| 7 | Boltzmann–Gumbel | 48.1 | 2.21 | 49% | 0.79 |
| 8 | ε-greedy | 48.4 | 3.34 | 43% | 0.10 |
| 9 | UCB1-Tuned | 51.7 | 1.98 | 44% | 0.20 |
| 10 | Bayes-UCB | 52.4 | 1.78 | 43% | 0.28 |
| 11 | KL-UCB++ | 52.5 | 1.90 | 64% | 4.59 |
| 12 | Sequential halving + commit | 53.3 | 4.92 | 30% | 0.08 |
| 13 | MOSS | 55.6 | 1.74 | 58% | 0.31 |
| 14 | Thompson sampling | 55.9 | 1.82 | 39% | 1.71 |
| 15 | MOSS-anytime | 56.1 | 2.00 | 46% | 0.28 |
| 16 | Perturbed-history exploration | 56.6 | 1.78 | 39% | 2.77 |
| 17 | IMED | 56.6 | 1.67 | 37% | 0.49 |
| 18 | KL-UCB+ | 57.0 | 1.84 | 35% | 6.06 |
| 19 | KL-UCB | 60.4 | 1.41 | 31% | 5.98 |
| 20 | Explore-then-commit | 61.6 | 4.92 | 25% | 0.10 |
| 21 | ε-greedy, constant step | 63.0 | 2.97 | 28% | 0.13 |
| 22 | Softmax (Boltzmann) | 63.5 | 2.69 | 21% | 0.26 |
| 23 | Tsallis-INF | 65.4 | 1.71 | 28% | 6.52 |
| 24 | UCB2 | 65.8 | 1.41 | 29% | 0.72 |
| 25 | Optimistic initial values | 66.7 | 1.56 | 28% | 0.09 |
| 26 | Gradient bandit | 69.9 | 1.44 | 28% | 0.29 |
| 27 | Annealed softmax | 70.2 | 1.55 | 17% | 0.29 |
| 28 | EXP3-IX | 70.8 | 1.52 | 20% | 0.26 |
| 29 | Reinforcement comparison | 70.9 | 1.48 | 26% | 0.25 |
| 30 | Sliding-window UCB | 71.1 | 1.34 | 18% | 0.20 |
| 31 | UCB1 | 71.5 | 1.31 | 18% | 0.18 |
| 32 | Linear reward-inaction | 71.6 | 1.60 | 21% | 0.11 |
| 33 | Monitored UCB | 71.8 | 1.30 | 19% | 0.24 |
| 34 | Discounted Thompson | 72.2 | 1.37 | 16% | 1.74 |
| 35 | EXP3 | 72.5 | 1.56 | 19% | 0.26 |
| 36 | UCB1-Normal | 74.0 | 1.36 | 15% | 0.13 |
| 37 | EXP3.S | 74.1 | 1.49 | 15% | 0.19 |
| 38 | UCB-V | 76.0 | 1.39 | 15% | 0.24 |
| 39 | Discounted UCB | 76.4 | 1.35 | 13% | 0.19 |
| 40 | Rexp3 | 77.1 | 1.45 | 14% | 0.30 |
| 41 | EXP3.P | 79.3 | 1.43 | 11% | 0.27 |
| 42 | Improved UCB | 81.2 | 1.45 | 10% | 0.06 |
| 43 | Successive elimination | 81.2 | 1.45 | 10% | 0.10 |
| 44 | Random | 81.3 | 1.49 | 10% | 0.09 |
| 45 | Greedy | 82.2 | 5.55 | 8% | 0.14 |

### Gaussian testbed

The classic 10-armed testbed: μ ~ N(0, 1), rewards ~ N(μ, 1).

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Gittins index (approx.) | 44.5 | 3.73 | 96% | 0.82 |
| 2 | Knowledge gradient | 47.8 | 10.7 | 87% | 1.21 |
| 3 | Boltzmann–Gumbel | 48.0 | 5.84 | 95% | 0.74 |
| 4 | Information-directed sampling | 53.6 | 3.89 | 95% | 37.17 |
| 5 | MOSS | 60.1 | 2.41 | 94% | 0.28 |
| 6 | Bayes-UCB | 63.8 | 5.59 | 94% | 0.22 |
| 7 | MOSS-anytime | 63.9 | 4.00 | 93% | 0.27 |
| 8 | Thompson sampling | 66.2 | 4.20 | 94% | 0.63 |
| 9 | Perturbed-history exploration | 78.9 | 3.40 | 93% | 0.61 |
| 10 | KL-UCB+ | 80.0 | 2.90 | 95% | 0.35 |
| 11 | IMED | 83.9 | 3.26 | 93% | 0.24 |
| 12 | KL-UCB++ | 84.1 | 3.96 | 94% | 0.61 |
| 13 | KL-UCB | 90.4 | 3.64 | 92% | 0.21 |
| 14 | UCB1-Tuned | 101 | 3.61 | 92% | 0.20 |
| 15 | UCB2 | 105 | 5.68 | 94% | 0.25 |
| 16 | Annealed softmax | 126 | 20.4 | 78% | 0.27 |
| 17 | UCB1 | 149 | 4.98 | 86% | 0.15 |
| 18 | Sliding-window UCB | 151 | 4.38 | 85% | 0.16 |
| 19 | Reinforcement comparison | 170 | 10.2 | 86% | 0.24 |
| 20 | Decaying ε-greedy | 171 | 10.5 | 86% | 0.08 |
| 21 | LinUCB | 172 | 43.5 | 68% | 0.24 |
| 22 | Monitored UCB | 184 | 3.49 | 89% | 0.43 |
| 23 | Gradient bandit | 186 | 11.8 | 83% | 0.26 |
| 24 | Pursuit | 187 | 8.45 | 91% | 0.12 |
| 25 | Sequential halving + commit | 187 | 13.4 | 83% | 0.07 |
| 26 | Softmax (Boltzmann) | 194 | 45.3 | 68% | 0.25 |
| 27 | Discounted Thompson | 220 | 9.01 | 74% | 0.71 |
| 28 | ε-greedy | 241 | 20.4 | 84% | 0.09 |
| 29 | Optimistic initial values | 243 | 6.21 | 82% | 0.07 |
| 30 | ε-greedy, constant step | 292 | 28.6 | 77% | 0.09 |
| 31 | Explore-then-commit | 327 | 15.2 | 84% | 0.05 |
| 32 | Improved UCB | 371 | 12.2 | 79% | 0.06 |
| 33 | Discounted UCB | 422 | 6.48 | 55% | 0.19 |
| 34 | UCB-V | 436 | 6.12 | 75% | 0.19 |
| 35 | Greedy | 506 | 83.6 | 35% | 0.11 |
| 36 | Successive elimination | 598 | 17.3 | 53% | 0.11 |
| 37 | Tsallis-INF | 644 | 17.4 | 55% | 5.90 |
| 38 | UCB1-Normal | 877 | 35.1 | 68% | 0.13 |
| 39 | EXP3-IX | 883 | 27.7 | 44% | 0.25 |
| 40 | EXP3 | 1,026 | 41.9 | 35% | 0.28 |
| 41 | Linear reward-inaction | 1,032 | 41.8 | 35% | 0.12 |
| 42 | EXP3.S | 1,077 | 44.4 | 29% | 0.19 |
| 43 | Rexp3 | 1,273 | 53.6 | 19% | 0.36 |
| 44 | EXP3.P | 1,387 | 60.7 | 13% | 0.26 |
| 45 | Random | 1,508 | 66.1 | 10% | 0.07 |

### Gaussian, drifting means

Starts like the Gaussian testbed, then every mean takes a random-walk step N(0, 0.03²) each round.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | UCB1-Tuned | 147 | 13.9 | 73% | 0.19 |
| 2 | KL-UCB | 148 | 16.5 | 72% | 0.21 |
| 3 | KL-UCB+ | 151 | 17.3 | 68% | 0.33 |
| 4 | IMED | 155 | 16.9 | 69% | 0.25 |
| 5 | Bayes-UCB | 155 | 20.2 | 68% | 0.23 |
| 6 | UCB1 | 161 | 10.8 | 75% | 0.16 |
| 7 | MOSS-anytime | 162 | 21.7 | 66% | 0.27 |
| 8 | MOSS | 172 | 23.3 | 62% | 0.27 |
| 9 | UCB2 | 177 | 19.2 | 69% | 0.23 |
| 10 | Thompson sampling | 178 | 23.0 | 65% | 0.63 |
| 11 | Gittins index (approx.) | 178 | 24.0 | 59% | 0.80 |
| 12 | KL-UCB++ | 180 | 20.9 | 65% | 0.56 |
| 13 | Information-directed sampling | 180 | 24.4 | 63% | 37.40 |
| 14 | Boltzmann–Gumbel | 181 | 24.8 | 61% | 0.79 |
| 15 | Sliding-window UCB | 185 | 9.38 | 76% | 0.16 |
| 16 | Perturbed-history exploration | 185 | 19.7 | 65% | 0.62 |
| 17 | Monitored UCB | 214 | 7.42 | 79% | 0.40 |
| 18 | Discounted Thompson | 220 | 10.3 | 77% | 0.71 |
| 19 | Annealed softmax | 225 | 24.2 | 63% | 0.27 |
| 20 | Knowledge gradient | 266 | 40.5 | 50% | 1.17 |
| 21 | Optimistic initial values | 286 | 14.0 | 70% | 0.07 |
| 22 | LinUCB | 309 | 46.2 | 49% | 0.21 |
| 23 | Reinforcement comparison | 322 | 28.6 | 58% | 0.23 |
| 24 | Decaying ε-greedy | 349 | 27.7 | 54% | 0.09 |
| 25 | Softmax (Boltzmann) | 355 | 61.7 | 49% | 0.24 |
| 26 | Gradient bandit | 363 | 33.5 | 56% | 0.26 |
| 27 | ε-greedy, constant step | 366 | 23.6 | 70% | 0.10 |
| 28 | ε-greedy | 368 | 21.7 | 57% | 0.08 |
| 29 | Pursuit | 391 | 34.1 | 52% | 0.10 |
| 30 | Discounted UCB | 408 | 4.22 | 66% | 0.19 |
| 31 | UCB-V | 414 | 10.8 | 78% | 0.17 |
| 32 | Sequential halving + commit | 426 | 48.2 | 49% | 0.07 |
| 33 | Improved UCB | 528 | 25.5 | 60% | 0.05 |
| 34 | Explore-then-commit | 625 | 50.2 | 43% | 0.05 |
| 35 | Greedy | 685 | 89.8 | 31% | 0.12 |
| 36 | Successive elimination | 693 | 21.2 | 57% | 0.10 |
| 37 | Tsallis-INF | 757 | 17.3 | 52% | 6.04 |
| 38 | UCB1-Normal | 987 | 36.3 | 72% | 0.13 |
| 39 | EXP3-IX | 1,021 | 22.7 | 44% | 0.26 |
| 40 | EXP3 | 1,199 | 37.7 | 34% | 0.26 |
| 41 | Linear reward-inaction | 1,202 | 38.7 | 35% | 0.11 |
| 42 | EXP3.S | 1,255 | 42.6 | 30% | 0.17 |
| 43 | Rexp3 | 1,481 | 53.1 | 22% | 0.26 |
| 44 | EXP3.P | 1,639 | 60.7 | 14% | 0.26 |
| 45 | Random | 1,805 | 67.7 | 10% | 0.07 |

### Bernoulli, abrupt changes

Bernoulli arms whose means are all redrawn from U(0, 1) at four evenly spaced change points.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | KL-UCB | 107 | 4.82 | 55% | 5.42 |
| 2 | KL-UCB+ | 115 | 5.38 | 56% | 5.65 |
| 3 | IMED | 116 | 5.53 | 49% | 0.48 |
| 4 | Bayes-UCB | 121 | 5.92 | 45% | 0.23 |
| 5 | KL-UCB++ | 128 | 5.62 | 45% | 3.95 |
| 6 | UCB1-Tuned | 129 | 6.23 | 47% | 0.19 |
| 7 | MOSS-anytime | 132 | 4.42 | 50% | 0.27 |
| 8 | Discounted Thompson | 141 | 3.16 | 60% | 1.74 |
| 9 | Optimistic initial values | 142 | 3.85 | 57% | 0.08 |
| 10 | Gittins index (approx.) | 142 | 7.22 | 38% | 0.85 |
| 11 | MOSS | 143 | 4.60 | 43% | 0.28 |
| 12 | Sliding-window UCB | 147 | 2.95 | 50% | 0.17 |
| 13 | UCB1 | 153 | 3.03 | 62% | 0.16 |
| 14 | Monitored UCB | 157 | 3.36 | 62% | 0.26 |
| 15 | LinUCB | 162 | 8.76 | 31% | 0.22 |
| 16 | Information-directed sampling | 168 | 6.70 | 34% | 92.07 |
| 17 | UCB2 | 181 | 4.39 | 49% | 0.68 |
| 18 | Boltzmann–Gumbel | 201 | 6.98 | 28% | 0.75 |
| 19 | Thompson sampling | 202 | 6.81 | 31% | 1.72 |
| 20 | Knowledge gradient | 209 | 10.0 | 23% | 1.28 |
| 21 | ε-greedy, constant step | 209 | 7.82 | 33% | 0.11 |
| 22 | Perturbed-history exploration | 215 | 6.54 | 27% | 2.03 |
| 23 | Decaying ε-greedy | 234 | 9.14 | 21% | 0.10 |
| 24 | Discounted UCB | 241 | 2.70 | 26% | 0.19 |
| 25 | ε-greedy | 241 | 9.24 | 19% | 0.11 |
| 26 | Tsallis-INF | 272 | 7.04 | 18% | 5.77 |
| 27 | UCB-V | 277 | 3.38 | 35% | 0.20 |
| 28 | Softmax (Boltzmann) | 284 | 9.66 | 21% | 0.27 |
| 29 | Annealed softmax | 292 | 6.20 | 18% | 0.29 |
| 30 | EXP3.S | 308 | 5.41 | 23% | 0.19 |
| 31 | EXP3-IX | 309 | 6.52 | 19% | 0.26 |
| 32 | Pursuit | 310 | 11.2 | 16% | 0.12 |
| 33 | Rexp3 | 322 | 5.09 | 25% | 0.29 |
| 34 | Reinforcement comparison | 331 | 7.51 | 19% | 0.25 |
| 35 | Gradient bandit | 335 | 7.36 | 18% | 0.30 |
| 36 | UCB1-Normal | 336 | 6.18 | 30% | 0.12 |
| 37 | EXP3 | 345 | 6.41 | 15% | 0.28 |
| 38 | Linear reward-inaction | 350 | 6.78 | 14% | 0.11 |
| 39 | Sequential halving + commit | 369 | 16.5 | 11% | 0.07 |
| 40 | Greedy | 375 | 18.6 | 8% | 0.13 |
| 41 | EXP3.P | 378 | 4.99 | 13% | 0.28 |
| 42 | Improved UCB | 410 | 5.66 | 10% | 0.06 |
| 43 | Explore-then-commit | 413 | 16.1 | 10% | 0.06 |
| 44 | Successive elimination | 414 | 4.98 | 10% | 0.10 |
| 45 | Random | 414 | 5.15 | 10% | 0.08 |

### Linear contextual

Each round shows a context x = (1, z₁…z₄), z ~ N(0, 1). Arm a pays N(θₐ·x, 1) with a hidden θₐ ~ N(0, 0.5²I). Without the context, only the average arm can be learned.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | LinUCB | 170 | 4.02 | 82% | 0.95 |
| 2 | Gittins index (approx.) | 956 | 34.2 | 29% | 0.75 |
| 3 | Information-directed sampling | 968 | 34.8 | 29% | 35.68 |
| 4 | Knowledge gradient | 971 | 38.4 | 27% | 1.18 |
| 5 | MOSS-anytime | 976 | 33.3 | 29% | 0.26 |
| 6 | MOSS | 978 | 34.8 | 29% | 0.28 |
| 7 | Bayes-UCB | 979 | 34.4 | 29% | 0.21 |
| 8 | Boltzmann–Gumbel | 981 | 36.6 | 28% | 0.70 |
| 9 | KL-UCB++ | 983 | 32.7 | 29% | 0.59 |
| 10 | Thompson sampling | 985 | 34.2 | 29% | 0.65 |
| 11 | KL-UCB+ | 988 | 33.9 | 30% | 0.34 |
| 12 | Perturbed-history exploration | 989 | 33.9 | 29% | 0.59 |
| 13 | IMED | 990 | 33.5 | 29% | 0.25 |
| 14 | KL-UCB | 1,000 | 33.2 | 28% | 0.20 |
| 15 | Softmax (Boltzmann) | 1,020 | 40.2 | 26% | 0.26 |
| 16 | UCB1-Tuned | 1,020 | 32.1 | 29% | 0.20 |
| 17 | Pursuit | 1,022 | 30.9 | 29% | 0.12 |
| 18 | Decaying ε-greedy | 1,026 | 35.1 | 28% | 0.10 |
| 19 | UCB2 | 1,027 | 33.2 | 29% | 0.37 |
| 20 | Sequential halving + commit | 1,035 | 35.2 | 27% | 0.07 |
| 21 | ε-greedy | 1,043 | 34.1 | 27% | 0.10 |
| 22 | Gradient bandit | 1,045 | 32.7 | 30% | 0.27 |
| 23 | Reinforcement comparison | 1,053 | 32.4 | 28% | 0.26 |
| 24 | Annealed softmax | 1,056 | 34.9 | 25% | 0.28 |
| 25 | UCB1 | 1,061 | 32.7 | 28% | 0.16 |
| 26 | Sliding-window UCB | 1,061 | 32.6 | 27% | 0.17 |
| 27 | ε-greedy, constant step | 1,069 | 32.8 | 24% | 0.10 |
| 28 | Monitored UCB | 1,080 | 31.1 | 28% | 0.34 |
| 29 | Explore-then-commit | 1,127 | 33.8 | 27% | 0.06 |
| 30 | Optimistic initial values | 1,144 | 32.6 | 25% | 0.08 |
| 31 | Greedy | 1,145 | 49.0 | 21% | 0.10 |
| 32 | Discounted Thompson | 1,163 | 30.9 | 22% | 0.68 |
| 33 | Improved UCB | 1,236 | 30.5 | 27% | 0.07 |
| 34 | Discounted UCB | 1,276 | 27.3 | 19% | 0.18 |
| 35 | UCB-V | 1,307 | 26.4 | 24% | 0.19 |
| 36 | Tsallis-INF | 1,389 | 28.2 | 19% | 6.30 |
| 37 | UCB1-Normal | 1,417 | 23.8 | 23% | 0.13 |
| 38 | Successive elimination | 1,461 | 27.3 | 18% | 0.12 |
| 39 | EXP3-IX | 1,479 | 24.3 | 16% | 0.25 |
| 40 | EXP3 | 1,514 | 26.0 | 15% | 0.26 |
| 41 | Linear reward-inaction | 1,523 | 25.4 | 15% | 0.12 |
| 42 | EXP3.S | 1,524 | 26.2 | 13% | 0.20 |
| 43 | Rexp3 | 1,581 | 27.4 | 12% | 0.28 |
| 44 | EXP3.P | 1,619 | 28.3 | 11% | 0.26 |
| 45 | Random | 1,649 | 29.2 | 10% | 0.08 |

