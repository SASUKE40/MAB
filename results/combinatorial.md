Setting: Combinatorial & ranking. Items (L) = 16. Horizon (T rounds) = 2,000. List size (m) = 4. 50 runs, seed 7.
Ranked by final regret (lower is better).

### Top-m semi-bandit

L items with Bernoulli weights wᵢ ~ U(0.1, 0.8). Choose any m items; reward is the sum of their realised weights and every chosen item's weight is observed (semi-bandit). Best action: the m heaviest items.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | CascadeLinTS | 103 | 6.31 | 103 | 2.706 | 67% | 3.09 |
| 2 | CTS | 109 | 10.1 | 101 | 2.702 | 70% | 6.63 |
| 3 | ESCB | 116 | 7.14 | 115 | 2.699 | 58% | 3.31 |
| 4 | PBM-UCB | 132 | 7.44 | 131 | 2.691 | 62% | 1.36 |
| 5 | PBM-PIE | 178 | 20.2 | 167 | 2.668 | 19% | 8.42 |
| 6 | CascadeTS | 180 | 7.43 | 179 | 2.667 | 61% | 1.46 |
| 7 | dcmKL-UCB | 183 | 7.20 | 185 | 2.666 | 52% | 12.55 |
| 8 | CascadeKL-UCB | 183 | 8.04 | 181 | 2.666 | 51% | 12.05 |
| 9 | DCM-UCB | 260 | 8.16 | 261 | 2.627 | 36% | 1.87 |
| 10 | CombUCB1 | 264 | 9.42 | 261 | 2.625 | 38% | 1.79 |
| 11 | CascadeUCB1 | 265 | 10.1 | 266 | 2.625 | 37% | 2.03 |
| 12 | CUCB | 267 | 9.47 | 269 | 2.624 | 38% | 1.95 |
| 13 | OSMD (negentropy) | 300 | 9.18 | 301 | 2.607 | 50% | 3.25 |
| 14 | OMM | 307 | 8.86 | 309 | 2.604 | 33% | 1.67 |
| 15 | CascadeLinUCB | 406 | 75.5 | 421 | 2.554 | 8.0% | 1.56 |
| 16 | EXP3.M | 542 | 14.2 | 554 | 2.486 | 36% | 3.12 |
| 17 | Greedy | 546 | 130 | 449 | 2.484 | 8.0% | 1.74 |
| 18 | TopRank | 595 | 31.1 | 589 | 2.460 | 21% | 3.53 |
| 19 | ComBand (CExp3) | 887 | 54.5 | 846 | 2.314 | 8.4% | 17.44 |
| 20 | Ranked bandits (RBA) | 1,047 | 32.5 | 1,051 | 2.234 | 3.6% | 2.60 |
| 21 | Random | 1,966 | 76.7 | 2,009 | 1.774 | 0.1% | 1.82 |

### Partition matroid

Items are dealt at random into m blocks; an action is a basis of the partition matroid (exactly one item per block). Linear Bernoulli rewards with semi-bandit feedback, wᵢ ~ U(0.1, 0.8). Best action: the heaviest item of each block. Top-m algorithms use the matroid oracle; adversarial ones factorise per block.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | CascadeLinTS | 93.3 | 9.52 | 85.5 | 2.648 | 66% | 2.76 |
| 2 | CTS | 98.4 | 8.83 | 90.6 | 2.646 | 65% | 5.96 |
| 3 | ESCB | 110 | 7.53 | 105 | 2.640 | 72% | 2.96 |
| 4 | PBM-UCB | 130 | 7.15 | 128 | 2.630 | 67% | 1.04 |
| 5 | CascadeKL-UCB | 174 | 7.56 | 172 | 2.607 | 61% | 12.67 |
| 6 | dcmKL-UCB | 177 | 6.84 | 172 | 2.606 | 58% | 12.12 |
| 7 | CascadeTS | 180 | 5.11 | 179 | 2.605 | 73% | 1.17 |
| 8 | PBM-PIE | 200 | 42.4 | 155 | 2.595 | 25% | 2.88 |
| 9 | CUCB | 266 | 7.99 | 261 | 2.562 | 49% | 1.69 |
| 10 | CascadeUCB1 | 267 | 8.13 | 267 | 2.561 | 47% | 1.62 |
| 11 | DCM-UCB | 267 | 7.76 | 266 | 2.561 | 47% | 1.46 |
| 12 | CombUCB1 | 269 | 8.49 | 269 | 2.560 | 47% | 1.35 |
| 13 | OSMD (negentropy) | 282 | 8.10 | 279 | 2.554 | 56% | 2.64 |
| 14 | Ranked bandits (RBA) | 319 | 9.00 | 321 | 2.535 | 44% | 1.06 |
| 15 | OMM | 321 | 7.60 | 319 | 2.534 | 40% | 1.62 |
| 16 | TopRank | 402 | 24.0 | 399 | 2.494 | 44% | 2.26 |
| 17 | CascadeLinUCB | 410 | 106 | 328 | 2.490 | 18% | 1.10 |
| 18 | Greedy | 517 | 132 | 354 | 2.436 | 8.0% | 1.22 |
| 19 | EXP3.M | 547 | 18.7 | 547 | 2.421 | 36% | 2.32 |
| 20 | ComBand (CExp3) | 729 | 44.5 | 695 | 2.330 | 23% | 10.52 |
| 21 | Random | 1,781 | 86.6 | 1,791 | 1.804 | 0.4% | 1.26 |

### Cascading clicks

Cascade model: the user scans the list top-down and clicks the first attractive item (attraction wᵢ ~ U(0.05, 0.35)), then leaves. Reward 1 if any click, so E[r] = 1 − Π(1 − wₐₖ). Items below the click are unobserved.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | ESCB | 52.5 | 3.18 | 52.2 | 0.736 | 29% | 3.47 |
| 2 | CascadeLinUCB | 64.0 | 9.48 | 60.9 | 0.731 | 18% | 1.51 |
| 3 | CTS | 66.5 | 4.33 | 66.4 | 0.729 | 30% | 6.10 |
| 4 | CascadeLinTS | 70.4 | 3.95 | 67.4 | 0.727 | 25% | 3.18 |
| 5 | CascadeTS | 78.8 | 3.50 | 78.2 | 0.723 | 23% | 1.45 |
| 6 | PBM-PIE | 94.0 | 8.11 | 89.4 | 0.716 | 15% | 7.22 |
| 7 | CascadeKL-UCB | 104 | 3.96 | 103 | 0.711 | 20% | 11.09 |
| 8 | dcmKL-UCB | 105 | 3.83 | 103 | 0.710 | 18% | 10.98 |
| 9 | PBM-UCB | 105 | 4.47 | 107 | 0.710 | 8.1% | 1.39 |
| 10 | CascadeUCB1 | 149 | 4.91 | 148 | 0.688 | 8.0% | 2.08 |
| 11 | DCM-UCB | 149 | 5.46 | 149 | 0.688 | 6.7% | 2.13 |
| 12 | CombUCB1 | 150 | 5.04 | 148 | 0.688 | 7.5% | 1.98 |
| 13 | CUCB | 151 | 4.99 | 154 | 0.687 | 7.4% | 1.99 |
| 14 | OMM | 169 | 5.93 | 170 | 0.678 | 4.0% | 2.01 |
| 15 | EXP3.M | 169 | 8.36 | 168 | 0.678 | 7.3% | 2.83 |
| 16 | ComBand (CExp3) | 178 | 9.02 | 181 | 0.674 | 3.0% | 16.98 |
| 17 | TopRank | 208 | 10.6 | 202 | 0.658 | 7.6% | 3.41 |
| 18 | Greedy | 231 | 36.0 | 224 | 0.647 | 2.0% | 1.82 |
| 19 | Ranked bandits (RBA) | 279 | 11.6 | 283 | 0.623 | 0.3% | 2.73 |
| 20 | OSMD (negentropy) | 319 | 13.4 | 329 | 0.603 | 0.2% | 3.44 |
| 21 | Random | 356 | 16.0 | 366 | 0.585 | 0.1% | 1.72 |

### Dependent click model

DCM (Guo et al. 2009): the user clicks every attractive item (wᵢ ~ U(0.05, 0.4)) while scanning; after a click at position k she is satisfied and stops with probability vₖ = 0.9·0.8ᵏ. Reward = satisfaction (hidden; learners see clicks and "any click"), E[r] = 1 − Π(1 − vₖ wₐₖ).

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | ESCB | 55.8 | 2.82 | 53.7 | 0.632 | 15% | 3.55 |
| 2 | CascadeLinUCB | 59.0 | 8.65 | 63.7 | 0.630 | 2.7% | 1.44 |
| 3 | CTS | 59.7 | 3.61 | 57.7 | 0.630 | 7.2% | 6.18 |
| 4 | CascadeLinTS | 65.6 | 3.69 | 61.9 | 0.627 | 6.2% | 3.12 |
| 5 | CascadeTS | 70.1 | 2.99 | 68.1 | 0.625 | 9.1% | 1.45 |
| 6 | PBM-PIE | 71.6 | 6.14 | 69.8 | 0.624 | 2.7% | 7.12 |
| 7 | PBM-UCB | 84.4 | 4.90 | 81.5 | 0.618 | 1.1% | 1.40 |
| 8 | CascadeKL-UCB | 95.6 | 3.80 | 94.0 | 0.612 | 3.5% | 10.71 |
| 9 | dcmKL-UCB | 96.8 | 3.03 | 97.9 | 0.611 | 4.2% | 10.65 |
| 10 | EXP3.M | 134 | 6.81 | 136 | 0.593 | 1.1% | 2.70 |
| 11 | CUCB | 136 | 3.65 | 135 | 0.592 | 1.5% | 2.14 |
| 12 | DCM-UCB | 136 | 3.89 | 138 | 0.592 | 1.5% | 2.15 |
| 13 | CascadeUCB1 | 137 | 4.18 | 137 | 0.591 | 1.2% | 2.18 |
| 14 | CombUCB1 | 137 | 3.23 | 138 | 0.591 | 0.7% | 1.94 |
| 15 | OMM | 154 | 4.23 | 156 | 0.583 | 0.6% | 2.00 |
| 16 | ComBand (CExp3) | 164 | 10.2 | 160 | 0.578 | 0.2% | 16.26 |
| 17 | Greedy | 176 | 27.1 | 168 | 0.572 | 0.0% | 1.64 |
| 18 | TopRank | 203 | 9.31 | 201 | 0.558 | 0.5% | 2.86 |
| 19 | Ranked bandits (RBA) | 278 | 10.2 | 281 | 0.521 | 0.0% | 2.56 |
| 20 | OSMD (negentropy) | 306 | 11.8 | 318 | 0.507 | 0.0% | 3.51 |
| 21 | Random | 373 | 15.5 | 382 | 0.473 | 0.0% | 1.67 |

### Position-based model

PBM: position k is examined with probability κₖ = 1/(1 + k/2) (known to PBM learners), item i is attractive with probability wᵢ ~ U(0.05, 0.7), and a click needs both. Reward = number of clicks, E[r] = Σ κₖ wₐₖ; the best list puts the most attractive items on top.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | PBM-PIE | 140 | 11.0 | 132 | 1.504 | 19% | 6.83 |
| 2 | CTS | 167 | 14.8 | 154 | 1.491 | 8.3% | 5.80 |
| 3 | PBM-UCB | 202 | 6.75 | 202 | 1.473 | 9.3% | 1.16 |
| 4 | CascadeLinUCB | 260 | 32.7 | 264 | 1.444 | 0.0% | 1.39 |
| 5 | CascadeLinTS | 274 | 16.2 | 275 | 1.437 | 2.6% | 2.99 |
| 6 | CascadeTS | 274 | 16.8 | 267 | 1.437 | 3.2% | 1.37 |
| 7 | dcmKL-UCB | 274 | 10.2 | 281 | 1.437 | 3.8% | 11.52 |
| 8 | CUCB | 282 | 12.2 | 280 | 1.433 | 2.3% | 1.89 |
| 9 | ESCB | 283 | 29.7 | 258 | 1.432 | 0.3% | 3.24 |
| 10 | CombUCB1 | 287 | 15.6 | 283 | 1.430 | 1.8% | 1.77 |
| 11 | OSMD (negentropy) | 310 | 19.6 | 295 | 1.419 | 1.3% | 2.85 |
| 12 | OMM | 318 | 15.4 | 307 | 1.415 | 1.1% | 1.88 |
| 13 | DCM-UCB | 344 | 10.6 | 348 | 1.402 | 4.1% | 1.91 |
| 14 | EXP3.M | 345 | 19.1 | 323 | 1.401 | 2.3% | 2.68 |
| 15 | CascadeKL-UCB | 349 | 16.3 | 351 | 1.400 | 1.4% | 9.95 |
| 16 | CascadeUCB1 | 430 | 13.3 | 435 | 1.359 | 0.7% | 1.85 |
| 17 | Greedy | 522 | 82.5 | 461 | 1.313 | 0.0% | 1.57 |
| 18 | TopRank | 536 | 25.6 | 531 | 1.306 | 0.8% | 2.75 |
| 19 | ComBand (CExp3) | 835 | 52.5 | 814 | 1.156 | 0.1% | 15.48 |
| 20 | Ranked bandits (RBA) | 890 | 28.2 | 894 | 1.129 | 0.0% | 2.61 |
| 21 | Random | 1,253 | 45.9 | 1,308 | 0.947 | 0.0% | 1.63 |

### Cascade with item features

Cascade clicks where attraction is linear in known item features: xᵢ on the 5-dimensional simplex, wᵢ = θᵀxᵢ with θⱼ ~ U(0, 0.5). CascadeLinUCB/LinTS read the features; the rest treat items as unrelated.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | CascadeLinUCB | 26.5 | 5.94 | 19.0 | 0.727 | 38% | 5.99 |
| 2 | CascadeLinTS | 30.9 | 3.69 | 28.3 | 0.725 | 37% | 4.06 |
| 3 | ESCB | 52.6 | 4.97 | 54.7 | 0.714 | 14% | 3.77 |
| 4 | CTS | 59.4 | 4.70 | 59.4 | 0.711 | 11% | 6.09 |
| 5 | CascadeTS | 64.6 | 4.37 | 65.1 | 0.708 | 12% | 1.56 |
| 6 | PBM-PIE | 67.9 | 6.23 | 66.3 | 0.707 | 3.1% | 7.51 |
| 7 | PBM-UCB | 78.7 | 5.09 | 77.8 | 0.701 | 4.4% | 1.55 |
| 8 | dcmKL-UCB | 86.3 | 5.98 | 88.4 | 0.698 | 4.1% | 10.85 |
| 9 | CascadeKL-UCB | 86.4 | 6.18 | 86.5 | 0.697 | 4.3% | 10.54 |
| 10 | DCM-UCB | 110 | 8.61 | 114 | 0.686 | 1.0% | 2.05 |
| 11 | CUCB | 110 | 8.89 | 113 | 0.685 | 1.6% | 2.11 |
| 12 | CombUCB1 | 110 | 9.01 | 115 | 0.685 | 2.0% | 2.10 |
| 13 | CascadeUCB1 | 111 | 8.70 | 115 | 0.685 | 0.9% | 2.01 |
| 14 | EXP3.M | 111 | 9.62 | 119 | 0.685 | 2.3% | 2.88 |
| 15 | OMM | 119 | 9.76 | 120 | 0.681 | 1.0% | 2.09 |
| 16 | ComBand (CExp3) | 126 | 11.8 | 122 | 0.678 | 1.5% | 16.77 |
| 17 | Greedy | 143 | 23.5 | 134 | 0.669 | 0.0% | 1.65 |
| 18 | Ranked bandits (RBA) | 158 | 15.1 | 160 | 0.662 | 0.2% | 2.79 |
| 19 | TopRank | 161 | 14.2 | 168 | 0.660 | 0.4% | 3.86 |
| 20 | OSMD (negentropy) | 172 | 16.9 | 172 | 0.654 | 0.1% | 3.29 |
| 21 | Random | 181 | 18.5 | 181 | 0.650 | 0.1% | 1.60 |

### Adversarial full-bandit

Full-bandit linear rewards: only the total Σ yₜᵢ of the chosen m items is observed (no per-item feedback). Means switch at T/3: a decoy set is best early (0.85 → 0.15) while the target set starts low (0.35 → 0.75); others ~ U(0.1, 0.5). Regret is against the best fixed set in hindsight and can dip below zero early.

| # | Algorithm | Regret | 95% CI | Median | Reward / round | Best list, last 10% | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | CUCB | 203 | 92.8 | 230 | 2.365 | 37% | 2.28 |
| 2 | ESCB | 690 | 182 | 625 | 2.122 | 10% | 3.52 |
| 3 | ComBand (CExp3) | 716 | 71.4 | 669 | 2.109 | 14% | 15.29 |
| 4 | PBM-PIE | 737 | 125 | 729 | 2.099 | 2.9% | 9.17 |
| 5 | PBM-UCB | 758 | 128 | 912 | 2.088 | 2.0% | 1.70 |
| 6 | dcmKL-UCB | 873 | 121 | 898 | 2.031 | 0.0% | 11.83 |
| 7 | CombUCB1 | 914 | 113 | 1,021 | 2.010 | 0.0% | 2.35 |
| 8 | CascadeKL-UCB | 934 | 90.3 | 970 | 2.000 | 0.0% | 11.71 |
| 9 | CascadeUCB1 | 945 | 109 | 1,027 | 1.995 | 0.0% | 2.22 |
| 10 | OMM | 968 | 122 | 1,063 | 1.983 | 3.9% | 2.29 |
| 11 | DCM-UCB | 1,042 | 97.2 | 1,078 | 1.946 | 0.0% | 2.25 |
| 12 | CascadeLinUCB | 1,093 | 94.8 | 1,076 | 1.921 | 0.0% | 2.53 |
| 13 | CascadeTS | 1,096 | 107 | 1,132 | 1.919 | 0.0% | 1.85 |
| 14 | OSMD (negentropy) | 1,099 | 54.1 | 1,124 | 1.918 | 3.6% | 3.22 |
| 15 | Ranked bandits (RBA) | 1,135 | 64.7 | 1,179 | 1.900 | 16% | 2.26 |
| 16 | CTS | 1,177 | 97.6 | 1,227 | 1.879 | 8.2% | 6.36 |
| 17 | CascadeLinTS | 1,340 | 55.9 | 1,378 | 1.797 | 3.9% | 3.20 |
| 18 | EXP3.M | 1,419 | 48.3 | 1,465 | 1.758 | 0.9% | 2.81 |
| 19 | Greedy | 1,508 | 137 | 1,517 | 1.713 | 0.0% | 1.78 |
| 20 | Random | 1,720 | 49.8 | 1,747 | 1.607 | 0.1% | 1.66 |
| 21 | TopRank | 1,723 | 50.1 | 1,726 | 1.606 | 0.0% | 6.59 |

