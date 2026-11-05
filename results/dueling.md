Setting: Dueling bandits. Arms (K) = 8. Horizon (duels) = 2,000. Preference strength = 1. 50 runs, seed 7.
Ranked by final strong regret (lower is better).

### Bradley–Terry (Condorcet)

Utility-based preferences P(i ≻ j) = σ(vᵢ − vⱼ) (the pairwise marginals of a Plackett–Luce model). The highest-utility arm is the Condorcet winner and preferences are transitive. Regret: P(w ≻ i) − ½.

| # | Algorithm | Strong regret | 95% CI | Median | Weak regret | (w, w), last 10% | (w, w), all | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | Double Thompson Sampling | 48.5 | 5.31 | 42.6 | 16.6 | 94% | 76% | 16.79 |
| 2 | RCS | 93.9 | 7.32 | 88.7 | 15.8 | 88% | 45% | 12.61 |
| 3 | Versatile-DB (Tsallis-INF) | 105 | 5.21 | 104 | 39.8 | 74% | 52% | 1.33 |
| 4 | RMED2 | 130 | 9.94 | 126 | 46.2 | 65% | 35% | 3.30 |
| 5 | RMED1 | 148 | 12.2 | 148 | 63.9 | 58% | 32% | 2.57 |
| 6 | CW-RMED | 174 | 16.0 | 172 | 86.3 | 59% | 24% | 13.47 |
| 7 | RMED2FH | 182 | 15.6 | 176 | 102 | 46% | 26% | 2.54 |
| 8 | Borda successive elimination | 190 | 20.1 | 210 | 107 | 73% | 34% | 0.61 |
| 9 | Sparring UCB1 | 192 | 4.62 | 194 | 86.5 | 32% | 20% | 0.89 |
| 10 | CCB | 194 | 12.7 | 200 | 101 | 59% | 22% | 3.35 |
| 11 | RUCB | 198 | 12.4 | 201 | 104 | 47% | 20% | 1.71 |
| 12 | Sparring EXP3 | 223 | 7.96 | 231 | 125 | 34% | 15% | 1.28 |
| 13 | DBGD | 231 | 11.8 | 230 | 133 | 13% | 11% | 1.79 |
| 14 | Doubler | 275 | 9.53 | 277 | 166 | 13% | 7.1% | 0.65 |
| 15 | MultiSBM | 289 | 9.63 | 290 | 186 | 9.7% | 6.2% | 0.56 |
| 16 | Interleaved Filter | 296 | 18.5 | 312 | 177 | 2.6% | 0.3% | 0.26 |
| 17 | Uniform pairs | 354 | 13.9 | 357 | 258 | 1.7% | 1.6% | 0.34 |
| 18 | Beat the Mean | 354 | 13.8 | 354 | 259 | 1.7% | 1.6% | 0.98 |
| 19 | SAVAGE | 354 | 13.8 | 356 | 246 | 0.0% | 0.0% | 2.10 |

### Condorcet with cycles

A Condorcet winner beats every arm by a narrow margin (5–20 points) while the other arms form a strong rock-paper-scissors tournament among themselves. Breaks the transitivity that IF and BTM rely on. Regret: P(w ≻ i) − ½.

| # | Algorithm | Strong regret | 95% CI | Median | Weak regret | (w, w), last 10% | (w, w), all | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | RMED1 | 48.0 | 4.86 | 43.5 | 31.5 | 94% | 76% | 3.11 |
| 2 | RMED2 | 48.4 | 4.02 | 47.2 | 27.6 | 93% | 73% | 3.39 |
| 3 | RMED2FH | 53.9 | 4.37 | 54.1 | 30.1 | 93% | 69% | 3.26 |
| 4 | Double Thompson Sampling | 55.1 | 4.03 | 53.4 | 17.4 | 83% | 59% | 17.61 |
| 5 | RCS | 108 | 4.39 | 106 | 17.0 | 23% | 5.1% | 13.67 |
| 6 | Versatile-DB (Tsallis-INF) | 112 | 8.82 | 110 | 55.3 | 50% | 32% | 1.18 |
| 7 | CCB | 129 | 5.53 | 128 | 42.1 | 15% | 3.5% | 3.09 |
| 8 | RUCB | 130 | 4.58 | 131 | 40.9 | 14% | 2.8% | 1.53 |
| 9 | Interleaved Filter | 160 | 10.2 | 156 | 80.1 | 30% | 6.4% | 0.19 |
| 10 | Borda successive elimination | 164 | 14.2 | 172 | 106 | 45% | 19% | 0.63 |
| 11 | Sparring UCB1 | 165 | 5.01 | 163 | 94.8 | 17% | 9.9% | 0.93 |
| 12 | CW-RMED | 166 | 10.7 | 166 | 98.2 | 25% | 8.8% | 13.20 |
| 13 | Sparring EXP3 | 173 | 6.35 | 176 | 105 | 20% | 9.1% | 1.20 |
| 14 | DBGD | 175 | 7.85 | 175 | 114 | 8.5% | 7.8% | 1.49 |
| 15 | Doubler | 193 | 6.85 | 195 | 128 | 6.4% | 4.2% | 0.72 |
| 16 | SAVAGE | 197 | 8.72 | 202 | 130 | 32% | 5.2% | 1.81 |
| 17 | MultiSBM | 214 | 8.01 | 214 | 155 | 2.3% | 2.3% | 0.54 |
| 18 | Uniform pairs | 215 | 8.43 | 212 | 156 | 1.4% | 1.5% | 0.28 |
| 19 | Beat the Mean | 215 | 8.55 | 215 | 157 | 1.5% | 1.6% | 0.68 |

### Copeland (no Condorcet winner)

Synthetic version of the MSLR/Yandex ranker cycles: three top arms A ≻ B ≻ C ≻ A beat the rest, but the two weakest arms ("spoilers") beat B and C by 15–25 points, so A has the most wins without beating everyone. Regret: normalised Copeland-score gap (Cpld(w) − Cpld(i))/(K − 1).

| # | Algorithm | Strong regret | 95% CI | Median | Weak regret | (w, w), last 10% | (w, w), all | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | Double Thompson Sampling | 254 | 14.6 | 242 | 134 | 91% | 58% | 14.54 |
| 2 | RCS | 367 | 9.90 | 356 | 81.3 | 0.0% | 0.0% | 12.84 |
| 3 | CW-RMED | 369 | 34.4 | 363 | 215 | 89% | 44% | 12.55 |
| 4 | CCB | 404 | 24.1 | 414 | 220 | 91% | 37% | 2.88 |
| 5 | Versatile-DB (Tsallis-INF) | 442 | 23.2 | 431 | 252 | 15% | 15% | 1.23 |
| 6 | Sparring EXP3 | 467 | 11.5 | 470 | 248 | 18% | 11% | 1.18 |
| 7 | RMED1 | 468 | 55.0 | 538 | 239 | 25% | 19% | 2.79 |
| 8 | Sparring UCB1 | 482 | 9.46 | 484 | 246 | 11% | 9.3% | 0.82 |
| 9 | Borda successive elimination | 494 | 23.7 | 517 | 213 | 29% | 8.9% | 0.58 |
| 10 | RMED2 | 510 | 39.0 | 550 | 233 | 9.7% | 7.7% | 3.57 |
| 11 | DBGD | 528 | 23.3 | 532 | 307 | 11% | 8.8% | 1.42 |
| 12 | RMED2FH | 547 | 44.1 | 571 | 295 | 9.7% | 7.4% | 2.76 |
| 13 | Doubler | 563 | 8.20 | 566 | 326 | 9.0% | 5.6% | 0.62 |
| 14 | Interleaved Filter | 575 | 43.5 | 589 | 373 | 38% | 11% | 0.17 |
| 15 | RUCB | 601 | 8.56 | 605 | 316 | 0.0% | 0.0% | 1.31 |
| 16 | MultiSBM | 619 | 5.64 | 623 | 379 | 2.0% | 2.7% | 0.56 |
| 17 | SAVAGE | 706 | 4.51 | 711 | 452 | 1.3% | 0.1% | 1.69 |
| 18 | Beat the Mean | 715 | 1.40 | 715 | 491 | 1.5% | 1.5% | 0.60 |
| 19 | Uniform pairs | 716 | 1.75 | 716 | 492 | 1.5% | 1.5% | 0.23 |

### Borda ≠ Condorcet

A Condorcet winner beats everyone by only 2–8 points, while a second arm loses to it narrowly but crushes all others (80–90%), so the Borda winner (best average win rate) differs from the Condorcet winner. Regret: Borda-score gap b(w) − b(i), b(i) = Σⱼ P(i ≻ j)/(K − 1).

| # | Algorithm | Strong regret | 95% CI | Median | Weak regret | (w, w), last 10% | (w, w), all | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | Borda successive elimination | 114 | 12.1 | 113 | 75.0 | 100% | 78% | 0.23 |
| 2 | Versatile-DB (Tsallis-INF) | 259 | 19.6 | 265 | 137 | 26% | 32% | 1.12 |
| 3 | Sparring UCB1 | 318 | 7.31 | 320 | 151 | 28% | 22% | 0.73 |
| 4 | RCS | 327 | 17.3 | 302 | 107 | 0.0% | 0.0% | 11.72 |
| 5 | Sparring EXP3 | 328 | 8.97 | 332 | 176 | 46% | 25% | 1.01 |
| 6 | Double Thompson Sampling | 333 | 28.7 | 331 | 176 | 8.4% | 14% | 14.70 |
| 7 | Interleaved Filter | 365 | 19.6 | 375 | 162 | 16% | 7.5% | 0.19 |
| 8 | RMED1 | 384 | 17.0 | 394 | 250 | 2.0% | 4.5% | 2.59 |
| 9 | RMED2 | 386 | 11.9 | 384 | 246 | 0.3% | 2.6% | 2.79 |
| 10 | RMED2FH | 401 | 19.4 | 403 | 274 | 5.3% | 7.7% | 2.47 |
| 11 | DBGD | 403 | 12.7 | 409 | 244 | 13% | 11% | 1.26 |
| 12 | Doubler | 413 | 5.60 | 411 | 237 | 22% | 12% | 0.59 |
| 13 | CCB | 449 | 12.1 | 438 | 269 | 0.0% | 0.0% | 2.56 |
| 14 | RUCB | 450 | 11.5 | 438 | 270 | 0.0% | 0.0% | 1.17 |
| 15 | MultiSBM | 464 | 5.11 | 466 | 300 | 8.4% | 7.4% | 0.48 |
| 16 | CW-RMED | 468 | 25.4 | 502 | 335 | 4.0% | 4.7% | 12.57 |
| 17 | SAVAGE | 585 | 5.98 | 585 | 436 | 0.0% | 0.0% | 1.59 |
| 18 | Beat the Mean | 585 | 5.95 | 586 | 447 | 1.7% | 1.6% | 0.58 |
| 19 | Uniform pairs | 586 | 5.93 | 585 | 448 | 1.4% | 1.5% | 0.21 |

### Continuous concave utility

Arms are points in the unit ball of ℝ⁵ with utility v(x) = −κ‖x − x*‖² and P(x ≻ y) = σ(v(x) − v(y)). One of the K arms sits at the optimum x*. DBGD searches the whole ball; every other method duels the K arms. Regret: P(x* ≻ x) − ½.

| # | Algorithm | Strong regret | 95% CI | Median | Weak regret | (w, w), last 10% | (w, w), all | ms / run |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | Double Thompson Sampling | 37.0 | 3.45 | 32.6 | 10.8 | 97% | 88% | 15.20 |
| 2 | RCS | 55.3 | 4.76 | 50.3 | 7.84 | 96% | 79% | 11.65 |
| 3 | RMED2 | 62.7 | 3.94 | 63.6 | 18.2 | 97% | 78% | 2.72 |
| 4 | RMED1 | 83.9 | 7.15 | 78.2 | 37.0 | 96% | 74% | 2.64 |
| 5 | CCB | 91.4 | 7.17 | 88.2 | 36.6 | 96% | 73% | 2.40 |
| 6 | Versatile-DB (Tsallis-INF) | 94.9 | 4.26 | 93.6 | 27.9 | 88% | 71% | 0.91 |
| 7 | CW-RMED | 98.3 | 11.2 | 85.1 | 48.4 | 93% | 71% | 12.60 |
| 8 | RUCB | 99.3 | 9.60 | 98.9 | 40.9 | 95% | 70% | 1.16 |
| 9 | RMED2FH | 126 | 17.5 | 104 | 76.9 | 89% | 68% | 2.65 |
| 10 | DBGD | 173 | 15.8 | 165 | 122 | 64% | 49% | 0.95 |
| 11 | Sparring UCB1 | 184 | 3.65 | 182 | 51.2 | 66% | 46% | 0.72 |
| 12 | Borda successive elimination | 188 | 31.6 | 148 | 89.8 | 86% | 59% | 0.31 |
| 13 | Interleaved Filter | 254 | 20.7 | 247 | 141 | 84% | 36% | 0.18 |
| 14 | Sparring EXP3 | 262 | 7.83 | 256 | 130 | 68% | 35% | 1.05 |
| 15 | Doubler | 364 | 8.81 | 367 | 186 | 31% | 17% | 0.54 |
| 16 | MultiSBM | 368 | 6.71 | 373 | 217 | 41% | 21% | 0.51 |
| 17 | SAVAGE | 503 | 17.0 | 495 | 365 | 38% | 9.1% | 1.47 |
| 18 | Uniform pairs | 609 | 20.2 | 612 | 462 | 1.5% | 1.6% | 0.26 |
| 19 | Beat the Mean | 610 | 19.6 | 613 | 462 | 1.5% | 1.6% | 0.64 |

