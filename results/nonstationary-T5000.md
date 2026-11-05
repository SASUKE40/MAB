K = 10 arms, T = 5,000 steps, 100 runs per cell, seed 7.
Regret is mean cumulative pseudo-regret at T, ± a 95% confidence interval. "Best arm" is the share of the last 10% of steps spent on the best arm.

### Gaussian, drifting means

Starts like the Gaussian testbed, then every mean takes a random-walk step N(0, 0.03²) each round.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | Discounted Thompson | 782 | 83.7 | 80% | 3.17 |
| 2 | Sliding-window UCB | 857 | 37.3 | 81% | 0.82 |
| 3 | UCB1 | 1,635 | 343 | 54% | 0.85 |
| 4 | KL-UCB | 1,786 | 370 | 54% | 1.03 |
| 5 | UCB1-Tuned | 1,853 | 381 | 51% | 1.06 |
| 6 | Thompson sampling | 2,350 | 459 | 45% | 3.10 |
| 7 | ε-greedy | 2,971 | 235 | 49% | 0.48 |
| 8 | EXP3 | 5,245 | 201 | 50% | 1.29 |

### Bernoulli, abrupt changes

Bernoulli arms whose means are all redrawn from U(0, 1) at four evenly spaced change points.

| # | Algorithm | Regret | ±95% CI | Best arm | ms / run |
|--:|---|--:|--:|--:|--:|
| 1 | KL-UCB | 423 | 35.5 | 54% | 24.21 |
| 2 | UCB1 | 454 | 22.5 | 66% | 0.80 |
| 3 | Discounted Thompson | 458 | 14.8 | 65% | 9.51 |
| 4 | UCB1-Tuned | 485 | 37.8 | 46% | 1.02 |
| 5 | Sliding-window UCB | 647 | 7.72 | 45% | 0.93 |
| 6 | Thompson sampling | 808 | 45.4 | 32% | 9.13 |
| 7 | ε-greedy | 1,168 | 60.0 | 20% | 0.47 |
| 8 | EXP3 | 1,458 | 59.6 | 16% | 1.28 |

