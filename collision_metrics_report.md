# Collision Resolution Metrics Report

*Generated: 2026-09-25 18:29:15*

## 1. Collision Rates (Before / After Resolution)

| Room | n | ColObj (Before) | ColScene (Before) | ColObj (After) | ColScene (After) |
|:-----|--:|:----:|:----:|:----:|:----:|
| bedroom | 162 | 0.0148 | 0.0864 | 0.0000 | 0.0000 |
| diningroom | 177 | 0.0093 | 0.2881 | 0.0023 | 0.0565 |
| livingroom | 192 | 0.0066 | 0.2552 | 0.0004 | 0.0208 |

## 2. Penetration Depth Metrics (Before Resolution)

| Room | n | Colliding Pairs | Mean Depth (m) | Max Depth (m) | Depth/ObjSize (%) |
|:-----|--:|---:|:----:|:----:|:----:|
| bedroom | 162 | 15 | 0.2388 Â± 0.2566 | 0.9144 | 51.5 Â± 46.2 |
| diningroom | 177 | 70 | 0.2204 Â± 0.2096 | 1.1336 | 53.8 Â± 44.8 |
| livingroom | 192 | 65 | 0.2084 Â± 0.2153 | 1.5129 | 52.4 Â± 44.5 |

## 3. Convergence Metrics

| Room | n | Mean Iter | Median Iter | Min | Max | Std | Failure Rate |
|:-----|--:|:----:|:----:|:----:|:----:|:----:|:----:|
| bedroom | 162 | 1.7 | 2 | 1 | 5 | 0.7 | 0.0% (0/162) |
| diningroom | 177 | 18.5 | 2 | 1 | 500 | 82.3 | 2.8% (5/177) |
| livingroom | 192 | 6.5 | 2 | 1 | 500 | 36.0 | 0.5% (1/192) |

### Iteration Distribution

**bedroom** (n=162):

| Iterations | Count | Pct |
|:-----------|------:|----:|
| 1 | 61 | 37.7% |
| 2-5 | 101 | 62.3% |
| 6-10 | 0 | 0.0% |
| 11-50 | 0 | 0.0% |
| 51-100 | 0 | 0.0% |
| 101-499 | 0 | 0.0% |
| 500 | 0 | 0.0% |

**diningroom** (n=177):

| Iterations | Count | Pct |
|:-----------|------:|----:|
| 1 | 49 | 27.7% |
| 2-5 | 86 | 48.6% |
| 6-10 | 21 | 11.9% |
| 11-50 | 16 | 9.0% |
| 51-100 | 0 | 0.0% |
| 101-499 | 0 | 0.0% |
| 500 | 5 | 2.8% |

**livingroom** (n=192):

| Iterations | Count | Pct |
|:-----------|------:|----:|
| 1 | 52 | 27.1% |
| 2-5 | 92 | 47.9% |
| 6-10 | 34 | 17.7% |
| 11-50 | 13 | 6.8% |
| 51-100 | 0 | 0.0% |
| 101-499 | 0 | 0.0% |
| 500 | 1 | 0.5% |

## 4. Displacement Cost Metrics

| Room | n | Objects Moved (%) | Mean Disp (cm) | Max Disp (cm) | Mean Disp/ObjSize |
|:-----|--:|:----:|:----:|:----:|:----:|
| bedroom | 162 | 67.1% (565/842) | 33.74 Â± 28.74 | 194.48 | 0.776 Â± 0.836 |
| diningroom | 177 | 80.1% (1527/1906) | 81.12 Â± 80.19 | 550.22 | 2.015 Â± 2.563 |
| livingroom | 192 | 75.4% (1736/2302) | 73.69 Â± 72.33 | 430.50 | 1.697 Â± 1.890 |

## 5. Resolution Success Rate

| Room | n | Success Rate | Solver Converged |
|:-----|--:|:----:|:----:|
| bedroom | 162 | 100.0% (162/162) | 162/162 |
| diningroom | 177 | 94.4% (167/177) | 172/177 |
| livingroom | 192 | 97.9% (188/192) | 191/192 |

## 6. Stress Test Results

- **Scenes**: 50, **Objects/scene**: 20, **Room half-extent**: 2.5 m
- **Mean collisions before**: 48.3
- **Mean collisions after**: 0.0
- **Mean iterations**: 14.1 (median: 14, max: 19)
- **Convergence failure rate**: 0.0% (0/50)
