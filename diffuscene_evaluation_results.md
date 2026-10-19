# DiffuScene Out-of-Bound (OOB) and Collision Evaluation Report

## 1. Summary of Results (Benchmark: 20 Scenes per Category, 60 Scenes Total)

| Room Category | Method | Avg #OOB $\downarrow$ | OOB Objs (%) $\downarrow$ | OOB Scenes (%) $\downarrow$ | ColObj $\downarrow$ | ColScene $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bedroom** ($N=20$) | DiffuScene (Raw) | **2.2000** | 44/98 (44.9%) | 17/20 (85.0%) | 0.5000 | 0.8500 |
| | DiffuScene + Resolver (Ours) | **0.0000** | **0/98 (0.0%)** | **0/20 (0.0%)** | **0.0408** | **0.1000** |
| **Living Room** ($N=20$) | DiffuScene (Raw) | **4.4000** | 88/237 (37.1%) | 15/20 (75.0%) | 0.3418 | 0.9000 |
| | DiffuScene + Resolver (Ours) | **0.0500** | **1/237 (0.4%)** | **1/20 (5.0%)** | **0.0844** | **0.2000** |
| **Dining Room** ($N=20$) | DiffuScene (Raw) | **4.4000** | 88/215 (40.9%) | 14/20 (70.0%) | 0.4744 | 0.9500 |
| | DiffuScene + Resolver (Ours) | **0.0500** | **1/215 (0.5%)** | **1/20 (5.0%)** | **0.2884** | **0.4000** |
| **Overall** ($N=60$) | **DiffuScene (Raw)** | **3.6667** | **220/550 (40.0%)** | **46/60 (76.7%)** | **0.4218** | **0.9000** |
| | **DiffuScene + Resolver (Ours)** | **0.0333** | **2/550 (0.4%)** | **2/60 (3.3%)** | **0.1564** | **0.2333** |
| **Relative Reduction** | | **-99.1%** | **-99.0%** | **-95.7%** | **-92.4%** | **-73.9%** |

---

## 2. Extended Results (All 70 Scenes in Google Drive: 20B, 20L, 30D)

| Room Category | Method | Avg #OOB $\downarrow$ | OOB Objs (%) $\downarrow$ | OOB Scenes (%) $\downarrow$ | ColObj $\downarrow$ | ColScene $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bedroom** ($N=20$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | 2.2000<br>**0.0000** | 44/98 (44.9%)<br>**0/98 (0.0%)** | 17/20 (85.0%)<br>**0/20 (0.0%)** | 0.5000<br>**0.0408** | 0.8500<br>**0.1000** |
| **Living Room** ($N=20$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | 4.4000<br>**0.0500** | 88/237 (37.1%)<br>**1/237 (0.4%)** | 15/20 (75.0%)<br>**1/20 (5.0%)** | 0.3418<br>**0.0844** | 0.9000<br>**0.2000** |
| **Dining Room** ($N=30$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | 4.2333<br>**0.0333** | 127/328 (38.7%)<br>**1/328 (0.3%)** | 23/30 (76.7%)<br>**1/30 (3.3%)** | 0.4512<br>**0.2561** | 0.9667<br>**0.3667** |
| **Overall** ($N=70$) | **DiffuScene (Raw)**<br>**DiffuScene + Ours (PP)** | 3.7000<br>**0.0286** | 259/663 (39.1%)<br>**2/663 (0.3%)** | 55/70 (78.6%)<br>**2/70 (2.9%)** | 0.4193<br>**0.1629** | 0.9143<br>**0.2429** |
| **Relative Reduction** | | **-99.2%** | **-99.2%** | **-96.4%** | **-89.9%** | **-69.1%** |
