# Week 4 Assignment — EKF vs UKF

**Course:** Autonomous Driving  
**Program:** M.S. Mechanical Engineering  
**University:** Kongju National University

## Overview

This assignment compares the Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) in a nonlinear range-bearing tracking system.

## Experiments

1. Baseline EKF and UKF performance comparison.
2. Effect of disabling bearing-angle normalization.
3. Sensitivity to initial covariance.
4. Sensitivity to the UKF scaling parameter alpha.
5. Statistical consistency analysis using NEES.
6. Information-form measurement update and commutativity proof.

## Files

- `Week4 Assignment.pdf` — Assignment report.
- `ekf_vs_ukf_modified.py` — Python simulation code.
- `corrected.csv` — Experimental results.
- `corrected.png` — Results figure.

## Running the Code

The main Python script is:

`ekf_vs_ukf_modified.py`

Run it from the assignment directory:

```bash
python ekf_vs_ukf_modified.py
```

The exact experiment options and required Python packages should be checked against the script before reproducing all results.

## Code Repository

https://github.com/Chal29/Autonomous/tree/main/Autonomous%20assignment
