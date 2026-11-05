Setting: Bandits with knapsacks. Arms (K) = 5. Horizon (T rounds) = 1,000. Budget per round β (B = β·T) = 0.25. 50 runs, seed 7.
Ranked by regret vs opt_lp (lower is better).

### One resource, tight budget

Bernoulli rewards and Bernoulli consumption of a single resource. Cost grows with reward and the most rewarding arm is the most expensive, so a cost-blind learner burns the budget early. Budget B = β·T.

| # | Algorithm | Regret | 95% CI | Median | Reward | % of OPT_LP | Budget left | Stop round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | CBwK (linear, simplified) | 49.4 | 9.14 | 43.1 | 410 | 89% | 17% | 1,000 | 0.93 |
| 2 | UCB-BwK | 57.1 | 7.76 | 55.8 | 402 | 88% | 0.0% | 958 | 1.26 |
| 3 | PD-BwK | 122 | 10.1 | 118 | 337 | 74% | 0.0% | 637 | 0.57 |
| 4 | Reward-per-cost UCB | 138 | 16.2 | 136 | 321 | 70% | 47% | 984 | 0.48 |
| 5 | LagrangeBwK | 140 | 13.1 | 149 | 319 | 70% | 18% | 977 | 0.62 |
| 6 | UCB1 (ignores costs) | 209 | 15.4 | 221 | 250 | 55% | 0.0% | 389 | 0.24 |

### Two resources

Each pull consumes two resources with independent Bernoulli draws; arms trade one resource against the other, so the LP optimum mixes two arms. Both budgets are β·T.

| # | Algorithm | Regret | 95% CI | Median | Reward | % of OPT_LP | Budget left | Stop round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | UCB-BwK | 37.2 | 5.17 | 36.8 | 344 | 90% | 0.0% | 960 | 1.03 |
| 2 | CBwK (linear, simplified) | 46.5 | 5.39 | 44.8 | 335 | 88% | 7.3% | 1,000 | 0.43 |
| 3 | PD-BwK | 47.1 | 5.77 | 42.2 | 334 | 88% | 0.0% | 599 | 0.39 |
| 4 | UCB1 (ignores costs) | 81.0 | 11.5 | 83.1 | 301 | 79% | 0.0% | 484 | 0.17 |
| 5 | LagrangeBwK | 82.8 | 9.82 | 82.7 | 299 | 78% | 4.6% | 921 | 0.39 |
| 6 | Reward-per-cost UCB | 98.2 | 16.0 | 82.1 | 283 | 74% | 8.1% | 838 | 0.26 |

### Shifting (adversarial-ish)

One resource. Halfway through, the reward means are reversed by rank (the best arm becomes the worst) while costs stay put. Benchmark: the LP over (phase, arm), i.e. the best policy that knows the switch and may split the budget between the two halves; it upper-bounds every policy. Methods that average all history adapt slowly; LagrangeBwK's adversarial (EXP3) primal is built for changing outcomes.

| # | Algorithm | Regret | 95% CI | Median | Reward | % of OPT_LP | Budget left | Stop round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | UCB-BwK | 133 | 15.4 | 124 | 526 | 80% | 2.0% | 969 | 0.97 |
| 2 | Reward-per-cost UCB | 139 | 14.4 | 138 | 520 | 79% | 49% | 987 | 0.32 |
| 3 | CBwK (linear, simplified) | 192 | 23.7 | 179 | 467 | 71% | 28% | 997 | 0.54 |
| 4 | PD-BwK | 255 | 24.1 | 271 | 405 | 62% | 0.4% | 734 | 0.46 |
| 5 | LagrangeBwK | 261 | 19.5 | 259 | 399 | 60% | 21% | 977 | 0.48 |
| 6 | UCB1 (ignores costs) | 409 | 18.6 | 420 | 251 | 38% | 0.0% | 390 | 0.13 |

### Contextual, linear rewards & costs

Each round shows one of 8 context types x = (1, z₁, z₂), drawn uniformly. Mean reward θₐ·x and mean cost wₐ·x are linear in x; one resource. The LP benchmark ranges over context-dependent policies p(a | x), so it pays to save budget for good contexts.

| # | Algorithm | Regret | 95% CI | Median | Reward | % of OPT_LP | Budget left | Stop round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | CBwK (linear, simplified) | 55.0 | 11.6 | 47.3 | 364 | 86% | 9.8% | 1,000 | 0.48 |
| 2 | PD-BwK | 77.5 | 8.52 | 74.7 | 342 | 81% | 1.2% | 789 | 0.30 |
| 3 | UCB-BwK | 79.1 | 10.3 | 71.4 | 340 | 81% | 1.5% | 974 | 0.47 |
| 4 | Reward-per-cost UCB | 104 | 13.4 | 93.9 | 315 | 75% | 5.5% | 868 | 0.15 |
| 5 | UCB1 (ignores costs) | 115 | 13.5 | 106 | 304 | 73% | 0.6% | 753 | 0.21 |
| 6 | LagrangeBwK | 155 | 16.1 | 150 | 264 | 64% | 6.8% | 944 | 0.24 |

