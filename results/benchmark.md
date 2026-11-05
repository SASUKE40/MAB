K = 10 arms, T = 1,000 steps, 200 runs per cell, seed 7.
Regret is mean cumulative pseudo-regret at T, ± a 95% confidence interval. "Best arm" is the share of the last 10% of steps spent on the best arm.

### Bernoulli, uniform means

Each arm pays 1 with probability μ ~ U(0, 1), else 0. Gaps are usually large.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Thompson sampling | 28.8 | 1.95 | 86% | 1.63 |
| 2 | KL-UCB | 31.8 | 1.89 | 88% | 5.00 |
| 3 | UCB1-Tuned | 34.4 | 1.41 | 81% | 0.22 |
| 4 | Bayes-UCB | 36.4 | 1.66 | 84% | 0.31 |
| 5 | MOSS | 47.2 | 1.36 | 91% | 0.28 |
| 6 | Decaying ε-greedy | 50.7 | 3.32 | 82% | 0.08 |
| 7 | ε-greedy | 69.0 | 4.36 | 76% | 0.09 |
| 8 | Softmax (Boltzmann) | 85.0 | 12.0 | 47% | 0.24 |
| 9 | Discounted Thompson | 85.1 | 4.14 | 64% | 1.72 |
| 10 | Explore-then-commit | 99.6 | 4.94 | 71% | 0.04 |
| 11 | Optimistic initial values | 113 | 2.47 | 79% | 0.07 |
| 12 | Sliding-window UCB | 125 | 2.46 | 56% | 0.16 |
| 13 | UCB1 | 130 | 2.62 | 56% | 0.16 |
| 14 | Gradient bandit | 137 | 2.83 | 76% | 0.27 |
| 15 | EXP3 | 184 | 5.36 | 51% | 0.26 |
| 16 | Greedy | 245 | 31.8 | 20% | 0.14 |
| 17 | Random | 419 | 11.4 | 10% | 0.07 |

### Bernoulli, close arms

One arm pays 0.55; the rest draw μ ~ U(0.40, 0.52). Small gaps make exploration expensive.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Decaying ε-greedy | 42.9 | 3.61 | 44% | 0.09 |
| 2 | ε-greedy | 48.4 | 3.34 | 43% | 0.10 |
| 3 | UCB1-Tuned | 51.7 | 1.98 | 44% | 0.19 |
| 4 | Bayes-UCB | 52.4 | 1.78 | 43% | 0.26 |
| 5 | MOSS | 55.6 | 1.74 | 58% | 0.28 |
| 6 | Thompson sampling | 55.9 | 1.82 | 39% | 1.58 |
| 7 | KL-UCB | 60.4 | 1.41 | 31% | 5.60 |
| 8 | Explore-then-commit | 61.6 | 4.92 | 25% | 0.04 |
| 9 | Softmax (Boltzmann) | 63.5 | 2.69 | 21% | 0.25 |
| 10 | Optimistic initial values | 66.7 | 1.56 | 28% | 0.08 |
| 11 | Gradient bandit | 69.9 | 1.44 | 28% | 0.25 |
| 12 | Sliding-window UCB | 71.1 | 1.34 | 18% | 0.18 |
| 13 | UCB1 | 71.5 | 1.31 | 18% | 0.16 |
| 14 | Discounted Thompson | 72.2 | 1.37 | 16% | 1.62 |
| 15 | EXP3 | 72.5 | 1.56 | 19% | 0.24 |
| 16 | Random | 81.3 | 1.49 | 10% | 0.07 |
| 17 | Greedy | 82.2 | 5.55 | 8% | 0.13 |

### Gaussian testbed

The classic 10-armed testbed: μ ~ N(0, 1), rewards ~ N(μ, 1).

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | MOSS | 60.1 | 2.41 | 94% | 0.26 |
| 2 | Bayes-UCB | 63.8 | 5.59 | 94% | 0.20 |
| 3 | Thompson sampling | 66.2 | 4.20 | 94% | 0.67 |
| 4 | KL-UCB | 90.4 | 3.64 | 92% | 0.18 |
| 5 | UCB1-Tuned | 101 | 3.61 | 92% | 0.18 |
| 6 | UCB1 | 149 | 4.98 | 86% | 0.13 |
| 7 | Sliding-window UCB | 151 | 4.38 | 85% | 0.14 |
| 8 | Decaying ε-greedy | 171 | 10.5 | 86% | 0.07 |
| 9 | Gradient bandit | 186 | 11.8 | 83% | 0.25 |
| 10 | Softmax (Boltzmann) | 194 | 45.3 | 68% | 0.23 |
| 11 | Discounted Thompson | 220 | 9.01 | 74% | 0.70 |
| 12 | ε-greedy | 241 | 20.4 | 84% | 0.08 |
| 13 | Optimistic initial values | 243 | 6.21 | 82% | 0.06 |
| 14 | Explore-then-commit | 327 | 15.2 | 84% | 0.04 |
| 15 | Greedy | 506 | 83.6 | 35% | 0.10 |
| 16 | EXP3 | 1,026 | 41.9 | 35% | 0.25 |
| 17 | Random | 1,508 | 66.1 | 10% | 0.05 |

### Gaussian, drifting means

Starts like the Gaussian testbed, then every mean takes a random-walk step N(0, 0.03²) each round.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | UCB1-Tuned | 147 | 13.9 | 73% | 0.18 |
| 2 | KL-UCB | 148 | 16.5 | 72% | 0.18 |
| 3 | Bayes-UCB | 155 | 20.2 | 68% | 0.19 |
| 4 | UCB1 | 161 | 10.8 | 75% | 0.13 |
| 5 | MOSS | 172 | 23.3 | 62% | 0.26 |
| 6 | Thompson sampling | 178 | 23.0 | 65% | 0.65 |
| 7 | Sliding-window UCB | 185 | 9.38 | 76% | 0.14 |
| 8 | Discounted Thompson | 220 | 10.3 | 77% | 0.67 |
| 9 | Optimistic initial values | 286 | 14.0 | 70% | 0.06 |
| 10 | Decaying ε-greedy | 349 | 27.7 | 54% | 0.07 |
| 11 | Softmax (Boltzmann) | 355 | 61.7 | 49% | 0.22 |
| 12 | Gradient bandit | 363 | 33.5 | 56% | 0.24 |
| 13 | ε-greedy | 368 | 21.7 | 57% | 0.08 |
| 14 | Explore-then-commit | 625 | 50.2 | 43% | 0.04 |
| 15 | Greedy | 685 | 89.8 | 31% | 0.09 |
| 16 | EXP3 | 1,199 | 37.7 | 34% | 0.25 |
| 17 | Random | 1,805 | 67.7 | 10% | 0.05 |

### Bernoulli, abrupt changes

Bernoulli arms whose means are all redrawn from U(0, 1) at four evenly spaced change points.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | KL-UCB | 107 | 4.82 | 55% | 5.15 |
| 2 | Bayes-UCB | 121 | 5.92 | 45% | 0.22 |
| 3 | UCB1-Tuned | 129 | 6.23 | 47% | 0.18 |
| 4 | Discounted Thompson | 141 | 3.16 | 60% | 1.61 |
| 5 | Optimistic initial values | 142 | 3.85 | 57% | 0.08 |
| 6 | MOSS | 143 | 4.60 | 43% | 0.27 |
| 7 | Sliding-window UCB | 147 | 2.95 | 50% | 0.15 |
| 8 | UCB1 | 153 | 3.03 | 62% | 0.14 |
| 9 | Thompson sampling | 202 | 6.81 | 31% | 1.56 |
| 10 | Decaying ε-greedy | 234 | 9.14 | 21% | 0.08 |
| 11 | ε-greedy | 241 | 9.24 | 19% | 0.09 |
| 12 | Softmax (Boltzmann) | 284 | 9.66 | 21% | 0.23 |
| 13 | Gradient bandit | 335 | 7.36 | 18% | 0.26 |
| 14 | EXP3 | 345 | 6.41 | 15% | 0.25 |
| 15 | Greedy | 375 | 18.6 | 8% | 0.12 |
| 16 | Explore-then-commit | 413 | 16.1 | 10% | 0.04 |
| 17 | Random | 414 | 5.15 | 10% | 0.06 |

