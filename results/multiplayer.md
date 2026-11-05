Setting: Multi-player & federated. Arms (K) = 8. Horizon (T rounds) = 2,000. Players (M) = 3. 50 runs, seed 7.
Ranked by system regret (lower is better).

### Bernoulli, homogeneous players

K Bernoulli arms with means in [0.1, 0.9], the same for every player. Colliding players get 0 (and are told). Regret is against the M best arms each held by one player: OPT = sum of the top-M means.

| # | Algorithm | Regret | 95% CI | Median | Collisions | Coll. / round | Accidental coll. | Signalling coll. | Reward / round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | DPE (simplified) | 189 | 16.8 | 183 | 84.2 | 0.042 | 0.48 | 83.7 | 1.999 | 3.75 |
| 2 | MCTopM | 262 | 23.6 | 258 | 221 | 0.111 | 221 | 0.00 | 1.963 | 13.49 |
| 3 | RandTopM | 265 | 22.7 | 242 | 214 | 0.107 | 214 | 0.00 | 1.961 | 13.78 |
| 4 | Musical Chairs | 486 | 82.3 | 396 | 217 | 0.109 | 217 | 0.00 | 1.851 | 0.80 |
| 5 | Game of Thrones (simplified) | 617 | 49.9 | 597 | 343 | 0.172 | 343 | 0.00 | 1.785 | 1.25 |
| 6 | Selfish UCB1 | 794 | 26.4 | 812 | 747 | 0.374 | 747 | 0.00 | 1.697 | 1.52 |
| 7 | SIC-MMAB (simplified) | 1,674 | 88.0 | 1,689 | 926 | 0.463 | 0.82 | 925 | 1.257 | 1.25 |
| 8 | FedUCB (periodic sync) | 4,144 | 159 | 4,307 | 5,909 | 2.955 | 5,909 | 0.00 | 0.022 | 2.04 |
| 9 | FedLinUCB (periodic sync) | 4,149 | 159 | 4,296 | 5,921 | 2.960 | 5,921 | 0.00 | 0.019 | 1.89 |

### Player-specific means

Each player has its own mean for each arm, drawn uniformly in [0.05, 0.95]. Collisions give 0. OPT is the max-weight matching of players to distinct arms (Hungarian algorithm) — the target of Game of Thrones.

| # | Algorithm | Regret | 95% CI | Median | Collisions | Coll. / round | Accidental coll. | Signalling coll. | Reward / round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | MCTopM | 303 | 27.1 | 281 | 156 | 0.078 | 156 | 0.00 | 2.375 | 16.81 |
| 2 | RandTopM | 409 | 42.6 | 376 | 151 | 0.076 | 151 | 0.00 | 2.322 | 16.04 |
| 3 | Selfish UCB1 | 594 | 17.6 | 592 | 349 | 0.174 | 349 | 0.00 | 2.230 | 2.35 |
| 4 | Game of Thrones (simplified) | 816 | 50.7 | 822 | 266 | 0.133 | 266 | 0.00 | 2.119 | 1.87 |
| 5 | Musical Chairs | 920 | 105 | 876 | 216 | 0.108 | 216 | 0.00 | 2.067 | 1.64 |
| 6 | DPE (simplified) | 1,744 | 127 | 1,698 | 69.0 | 0.034 | 0.48 | 68.5 | 1.654 | 4.32 |
| 7 | SIC-MMAB (simplified) | 2,544 | 135 | 2,540 | 939 | 0.469 | 0.82 | 938 | 1.255 | 1.75 |
| 8 | FedUCB (periodic sync) | 5,007 | 94.9 | 5,057 | 5,914 | 2.957 | 5,914 | 0.00 | 0.023 | 2.34 |
| 9 | FedLinUCB (periodic sync) | 5,013 | 94.4 | 5,059 | 5,929 | 2.965 | 5,929 | 0.00 | 0.020 | 2.59 |

### Federated Bernoulli (no collisions)

M agents each face the same K Bernoulli arms with independent reward draws and never collide. Regret is summed over agents, each against its own best arm (here the same arm for all), so pooling statistics helps everyone.

| # | Algorithm | Regret | 95% CI | Median | Collisions | Coll. / round | Accidental coll. | Signalling coll. | Reward / round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | FedLinUCB (periodic sync) | 59.2 | 23.2 | 46.3 | 0.00 | 0.000 | 0.00 | 0.00 | 2.329 | 2.25 |
| 2 | MCTopM | 254 | 67.7 | 130 | 0.00 | 0.000 | 0.00 | 0.00 | 2.232 | 11.81 |
| 3 | FedUCB (periodic sync) | 272 | 12.6 | 274 | 0.00 | 0.000 | 0.00 | 0.00 | 2.223 | 2.24 |
| 4 | RandTopM | 300 | 88.1 | 154 | 0.00 | 0.000 | 0.00 | 0.00 | 2.209 | 11.66 |
| 5 | Musical Chairs | 370 | 31.7 | 359 | 0.00 | 0.000 | 0.00 | 0.00 | 2.174 | 1.44 |
| 6 | Selfish UCB1 | 486 | 19.6 | 496 | 0.00 | 0.000 | 0.00 | 0.00 | 2.116 | 2.01 |
| 7 | Game of Thrones (simplified) | 659 | 66.3 | 642 | 0.00 | 0.000 | 0.00 | 0.00 | 2.030 | 1.54 |
| 8 | DPE (simplified) | 676 | 78.9 | 627 | 0.00 | 0.000 | 0.00 | 0.00 | 2.021 | 4.32 |
| 9 | SIC-MMAB (simplified) | 1,742 | 125 | 1,740 | 0.00 | 0.000 | 0.00 | 0.00 | 1.488 | 1.68 |

### Federated linear contextual

Each round every agent sees K fresh 5-dimensional contexts x ∈ [0,1]⁵; reward = θ·x + N(0, 0.1²) with one θ shared by all agents (Σθ = 1, so means lie in [0, 1]). No collisions. Regret is summed over agents against each round's best context.

| # | Algorithm | Regret | 95% CI | Median | Collisions | Coll. / round | Accidental coll. | Signalling coll. | Reward / round | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | FedLinUCB (periodic sync) | 11.0 | 0.25 | 11.1 | 0.00 | 0.000 | 0.00 | 0.00 | 2.119 | 8.78 |
| 2 | RandTopM | 1,247 | 27.0 | 1,246 | 0.00 | 0.000 | 0.00 | 0.00 | 1.501 | 8.95 |
| 3 | Game of Thrones (simplified) | 1,247 | 26.4 | 1,244 | 0.00 | 0.000 | 0.00 | 0.00 | 1.501 | 1.40 |
| 4 | DPE (simplified) | 1,248 | 27.0 | 1,251 | 0.00 | 0.000 | 0.00 | 0.00 | 1.501 | 3.97 |
| 5 | SIC-MMAB (simplified) | 1,249 | 27.1 | 1,243 | 0.00 | 0.000 | 0.00 | 0.00 | 1.500 | 1.55 |
| 6 | FedUCB (periodic sync) | 1,249 | 26.5 | 1,248 | 0.00 | 0.000 | 0.00 | 0.00 | 1.500 | 2.08 |
| 7 | Musical Chairs | 1,251 | 26.6 | 1,238 | 0.00 | 0.000 | 0.00 | 0.00 | 1.499 | 1.54 |
| 8 | MCTopM | 1,251 | 27.2 | 1,253 | 0.00 | 0.000 | 0.00 | 0.00 | 1.499 | 8.99 |
| 9 | Selfish UCB1 | 1,251 | 26.7 | 1,254 | 0.00 | 0.000 | 0.00 | 0.00 | 1.499 | 2.31 |

