Setting: Restless bandits. Arms (N) = 10. Horizon (T rounds) = 500. Pulls per round (M) = 3. 50 runs, seed 7.
Ranked by regret vs whittle oracle (lower is better).

### Adherence (indexable)

Healthcare-adherence style arms: "good" = adherent. Adherence is sticky (Pr(stay good) 0.6–0.95 when left alone), recovery is rare without help (0.02–0.3), and a call raises both chances (more so from the bad state). Acting always helps and the good state is always better, the structure under which the arms are indexable.

| # | Algorithm | Regret | 95% CI | Median | Good arms / round | Share good | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | Whittle index (oracle) | 0.00 | 0.00 | 0.00 | 6.45 | 65% | 0.62 |
| 2 | UCWhittle (simplified) | 66.8 | 8.42 | 59.1 | 6.32 | 63% | 1.06 |
| 3 | Plug-in Whittle (greedy) | 93.6 | 11.8 | 84.4 | 6.26 | 63% | 0.81 |
| 4 | Thompson-sampling Whittle | 103 | 7.24 | 103 | 6.25 | 62% | 4.09 |
| 5 | Myopic (greedy one-step) | 224 | 30.7 | 194 | 6.00 | 60% | 0.67 |
| 6 | Round-robin | 436 | 16.9 | 442 | 5.58 | 56% | 0.10 |
| 7 | Random | 485 | 17.4 | 479 | 5.48 | 55% | 0.60 |

### Heterogeneous arm types

Each arm is one of four types: responds mainly in the bad state, responds mainly in the good state (prevents relapse), barely responds, or recovers on its own. Effects range from 0 to 0.6, so knowing which arm is which matters more than in the adherence instance.

| # | Algorithm | Regret | 95% CI | Median | Good arms / round | Share good | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | Whittle index (oracle) | 0.00 | 0.00 | 0.00 | 5.96 | 60% | 0.58 |
| 2 | Thompson-sampling Whittle | 97.1 | 10.3 | 92.1 | 5.77 | 58% | 3.70 |
| 3 | Plug-in Whittle (greedy) | 110 | 22.7 | 97.5 | 5.74 | 57% | 0.79 |
| 4 | Myopic (greedy one-step) | 114 | 25.6 | 86.5 | 5.73 | 57% | 0.67 |
| 5 | UCWhittle (simplified) | 188 | 25.1 | 177 | 5.59 | 56% | 0.95 |
| 6 | Round-robin | 563 | 33.6 | 561 | 4.84 | 48% | 0.10 |
| 7 | Random | 579 | 31.8 | 597 | 4.80 | 48% | 0.69 |

### Intervention fatigue (hard)

Hard, non-monotone instance: acting helps a lot in the bad state but hurts in the good state (fatigue: Pr(stay good) drops when pulled), and a third of the arms are near-copies with tiny differences. These arms break the usual sufficient conditions for indexability (acting always helps); a numerical check still finds a single Whittle threshold per state, so the hardness is in learning sign-flipping effects and near-ties, and Whittle remains only a heuristic for the coupled M-of-N problem — regret can go negative.

| # | Algorithm | Regret | 95% CI | Median | Good arms / round | Share good | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | Whittle index (oracle) | 0.00 | 0.00 | 0.00 | 7.26 | 73% | 0.81 |
| 2 | UCWhittle (simplified) | 58.1 | 5.92 | 55.5 | 7.15 | 71% | 1.26 |
| 3 | Plug-in Whittle (greedy) | 58.2 | 7.61 | 56.4 | 7.15 | 71% | 1.14 |
| 4 | Myopic (greedy one-step) | 79.4 | 12.4 | 75.6 | 7.11 | 71% | 0.87 |
| 5 | Thompson-sampling Whittle | 83.2 | 6.02 | 84.0 | 7.10 | 71% | 4.86 |
| 6 | Round-robin | 783 | 19.8 | 776 | 5.70 | 57% | 0.16 |
| 7 | Random | 831 | 20.8 | 825 | 5.60 | 56% | 0.93 |

