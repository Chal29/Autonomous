
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import lsq_linear

# ============================================================
# WEEK 7 - QUESTION 4
# Real Robot Odometry vs Ground Truth
# ============================================================



# Search local data first, then the original repository layout.
def locate(filename):
    here = Path(__file__).resolve().parent
    candidates = [here / filename,
                  here / "Probabilistic-Robotics-Algorithms-master" / "0.Dataset1" / filename,
                  Path.cwd() / filename,
                  Path.cwd() / "Probabilistic-Robotics-Algorithms-master" / "0.Dataset1" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename}; place it next to this script or in the dataset folder")

odom = np.loadtxt(locate("Odometry.dat"))
gt = np.loadtxt(locate("Groundtruth.dat"))

print("Odometry shape:", odom.shape)
print("Ground truth shape:", gt.shape)

# Expected odometry columns:
# timestamp, linear velocity, angular velocity
t = odom[:, 0]
v = odom[:, 1]
w = odom[:, 2]

# Expected ground truth columns:
# timestamp, x, y, heading
gt_t = gt[:, 0]

# 2. Angle wrapping function
def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# 3. Verify velocity data
print("\n--- Velocity verification ---")

print("Linear velocity range (m/s):",
      np.min(v), "to", np.max(v))

print("Angular velocity range (rad/s):",
      np.min(w), "to", np.max(w))

print("Mean absolute linear velocity (m/s):",
      np.mean(np.abs(v)))

print("Mean absolute angular velocity (rad/s):",
      np.mean(np.abs(w)))

print("Nonzero linear velocity samples:",
      np.count_nonzero(v))

print("Nonzero angular velocity samples:",
      np.count_nonzero(w))


# 4. Verify timestamp alignment
print("\n--- Timestamp alignment verification ---")

print("First odometry timestamp:", t[0])
print("First ground-truth timestamp:", gt_t[0])
print("Start-time difference (s):", t[0] - gt_t[0])

print("Last odometry timestamp:", t[-1])
print("Last ground-truth timestamp:", gt_t[-1])
print("End-time difference (s):", t[-1] - gt_t[-1])


# 5. Interpolate ground truth to odometry timestamps
gt_x = np.interp(t, gt_t, gt[:, 1])
gt_y = np.interp(t, gt_t, gt[:, 2])

# Interpolate heading using unwrapped angles
gt_theta = np.interp(
    t,
    gt_t,
    np.unwrap(gt[:, 3])
)
gt_theta = wrap(gt_theta)


# 6. Initialize odometry trajectory
poses = np.zeros((len(t), 3))

# Start from time-aligned ground-truth pose
poses[0] = [
    gt_x[0],
    gt_y[0],
    gt_theta[0]
]

print("\n--- Initial pose verification ---")
print("Time-aligned initial pose:", poses[0])


# 7. Integrate recorded odometry
# Midpoint integration method
for i in range(1, len(t)):

    dt = t[i] - t[i - 1]

    if dt <= 0:
        poses[i] = poses[i - 1]
        continue

    x, y, theta = poses[i - 1]

    dtheta = w[i - 1] * dt
    theta_mid = theta + 0.5 * dtheta

    poses[i, 0] = (
        x + v[i - 1] * dt * np.cos(theta_mid)
    )

    poses[i, 1] = (
        y + v[i - 1] * dt * np.sin(theta_mid)
    )

    poses[i, 2] = wrap(theta + dtheta)


# 8. Calculate trajectory errors
errors = np.sqrt(
    (poses[:, 0] - gt_x) ** 2 +
    (poses[:, 1] - gt_y) ** 2
)

mean_error = np.mean(errors)
rmse = np.sqrt(np.mean(errors ** 2))
final_error = errors[-1]

print("\n--- Trajectory comparison ---")
print("Mean position error: {:.4f} m".format(mean_error))
print("Position RMSE: {:.4f} m".format(rmse))
print("Final position error: {:.4f} m".format(final_error))


# 9. Dataset timing verification
dt_values = np.diff(t)

print("\n--- Dataset timing check ---")

print("Odometry duration: {:.2f} s".format(
    t[-1] - t[0]
))

print("Ground truth duration: {:.2f} s".format(
    gt_t[-1] - gt_t[0]
))

print("Median odometry dt: {:.4f} s".format(
    np.median(dt_values)
))

print("Maximum odometry dt: {:.4f} s".format(
    np.max(dt_values)
))

print("Nonpositive odometry intervals:",
      np.sum(dt_values <= 0))

print("Odometry timestamps outside ground-truth range:",
      np.sum((t < gt_t[0]) | (t > gt_t[-1])))


# 10. Least-squares estimation of odometry noise coefficients (alpha1-alpha4)
# Lecture Eq. (6.8):
# Var(rot1 residual) = alpha1*rot1^2 + alpha2*trans^2
# Var(trans residual) = alpha3*trans^2 + alpha4*(rot1^2+rot2^2)
# Var(rot2 residual) = alpha1*rot2^2 + alpha2*trans^2
# Use nonoverlapping fixed-duration windows to reduce effects of ground-truth
# interpolation and avoid dividing small displacements by tiny time steps.
# These are empirical conditional-variance estimates, not guaranteed sensor
# calibration; residual biases and temporal correlation remain possible.

def estimate_alpha(window_seconds=1.0):
    anchors = [0]
    for i in range(1, len(t)):
        if t[i] - t[anchors[-1]] >= window_seconds:
            anchors.append(i)
    rows_rot, target_rot = [], []
    rows_trans, target_trans = [], []
    signed_residuals = []
    for i0, i1 in zip(anchors[:-1], anchors[1:]):
        # Ground-truth relative displacement in its initial local frame
        dx, dy = gt_x[i1]-gt_x[i0], gt_y[i1]-gt_y[i0]
        gth0 = gt_theta[i0]
        gx = np.cos(gth0)*dx + np.sin(gth0)*dy
        gy = -np.sin(gth0)*dx + np.cos(gth0)*dy
        gtrans = np.hypot(gx, gy)
        if gtrans < 1e-5:
            continue  # heading of zero-length motion is undefined
        grot1 = wrap(np.arctan2(gy, gx))
        grot2 = wrap(gt_theta[i1]-gt_theta[i0]-grot1)
        # Integrate recorded odometry over the same window in a local frame
        ox = oy = oth = 0.0
        for j in range(i0, i1):
            dtj = t[j+1]-t[j]
            if dtj <= 0:
                continue
            dth = w[j]*dtj
            ox += v[j]*dtj*np.cos(oth+0.5*dth)
            oy += v[j]*dtj*np.sin(oth+0.5*dth)
            oth += dth
        otrans = np.hypot(ox, oy)
        if otrans < 1e-5:
            continue
        orot1 = wrap(np.arctan2(oy, ox))
        orot2 = wrap(oth-orot1)
        e1 = wrap(grot1-orot1)
        et = gtrans-otrans
        e2 = wrap(grot2-orot2)
        # Regress squared residuals on lecture's squared-motion features.
        rows_rot.extend([[orot1**2, otrans**2], [orot2**2, otrans**2]])
        target_rot.extend([e1**2, e2**2])
        rows_trans.append([otrans**2, orot1**2+orot2**2])
        target_trans.append(et**2)
        signed_residuals.append([e1, et, e2])
    Arot = np.asarray(rows_rot)
    Atrans = np.asarray(rows_trans)
    brot = np.asarray(target_rot)
    btrans = np.asarray(target_trans)
    if len(Atrans) < 10:
        raise RuntimeError('Not enough valid motion windows to estimate coefficients')
    # Nonnegative bounded least squares (variance coefficients cannot be negative).
    rfit = lsq_linear(Arot, brot, bounds=(0, np.inf), tol=1e-12)
    tfit = lsq_linear(Atrans, btrans, bounds=(0, np.inf), tol=1e-12)
    alpha_hat = np.r_[rfit.x, tfit.x]
    # Reorder to alpha1,alpha2,alpha3,alpha4 (already in that order).
    residuals = np.asarray(signed_residuals)
    print('\n--- Q4: nonnegative least-squares alpha estimates ---')
    print(f'Window duration: {window_seconds:.2f} s; valid windows: {len(Atrans)}')
    for k, a in enumerate(alpha_hat, start=1):
        print(f'alpha{k}: {a:.8f}')
    print('Mean signed residual [rot1 rad, trans m, rot2 rad]:',
          np.array2string(residuals.mean(axis=0), precision=5))
    print('Design matrix rank (rotation/translation):',
          np.linalg.matrix_rank(Arot), np.linalg.matrix_rank(Atrans))
    print('NOTE: Estimates depend on the window length and include unmodeled bias.')
    return alpha_hat, Arot, brot, Atrans, btrans

alpha_hat, Arot, brot, Atrans, btrans = estimate_alpha(window_seconds=1.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, A, b, fit, title in [
    (axes[0], Arot, brot, alpha_hat[:2], 'Rotation noise variance'),
    (axes[1], Atrans, btrans, alpha_hat[2:], 'Translation noise variance')]:
    predicted = A @ fit
    ax.scatter(predicted, b, s=7, alpha=0.25)
    limit = max(np.percentile(b, 99), np.percentile(predicted, 99), 1e-9)
    ax.plot([0, limit], [0, limit], '--', color='black', lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('Predicted squared residual')
    ax.set_ylabel('Observed squared residual')
    ax.set_title(title)
    ax.grid(alpha=0.25)
fig.suptitle('Q4: Least-squares fit of odometry noise coefficients')
fig.tight_layout()
fig.savefig('week7_q4_alpha_least_squares.png', dpi=200)
plt.close(fig)

# 11. Test angle wrapping
print("\n--- Angle wrapping test ---")

test_angles = np.array([
    -4 * np.pi,
    -3 * np.pi / 2,
    -np.pi,
    0,
    np.pi,
    3 * np.pi / 2,
    4 * np.pi
])

for angle in test_angles:
    print(
        "Input: {:.4f} rad -> Wrapped: {:.4f} rad".format(
            angle, wrap(angle)
        )
    )


# 11. Plot trajectory comparison
plt.figure(figsize=(9, 7))

plt.plot(
    gt_x,
    gt_y,
    color="blue",
    linewidth=1.5,
    alpha=0.7,
    label="Ground truth",
    zorder=2
)

plt.plot(
    poses[:, 0],
    poses[:, 1],
    color="red",
    linewidth=1.5,
    alpha=0.9,
    label="Integrated odometry",
    zorder=3
)

# Mark starting positions
plt.scatter(
    gt_x[0],
    gt_y[0],
    color="green",
    marker="o",
    s=70,
    label="Start",
    zorder=5
)

# Mark final positions
plt.scatter(
    gt_x[-1],
    gt_y[-1],
    color="blue",
    marker="x",
    s=100,
    label="Ground-truth end",
    zorder=5
)

plt.scatter(
    poses[-1, 0],
    poses[-1, 1],
    color="red",
    marker="x",
    s=100,
    label="Odometry end",
    zorder=5
)

plt.xlabel("x (m)")
plt.ylabel("y (m)")

plt.title(
    "Week 7 Q4: Real Robot Odometry vs Ground Truth"
)

plt.axis("equal")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "week7_q4_trajectory_comparison.png",
    dpi=200
)

plt.show()
