
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# WEEK 7 - QUESTION 4
# Real Robot Odometry vs Ground Truth
# ============================================================

# 1. Load datasets
# Change filenames if your CSV files have different names.

odom = np.loadtxt(
    "Probabilistic-Robotics-Algorithms-master/0.Dataset1/Odometry.dat"
)

gt = np.loadtxt(
    "Probabilistic-Robotics-Algorithms-master/0.Dataset1/Groundtruth.dat"
)

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


# 10. Test angle wrapping
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
