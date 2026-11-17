"""Week 7 Assignment , Question 1 and 2: velocity-model sample distributions.
Based on Section 6.5 of the lecture. Run: python week7_motion_models_q1.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def sample_velocity(x, u, dt, a):
    v, w = u
    n = x.shape[0]
    vh = v + rng.normal(0, np.sqrt(a[0] * v**2 + a[1] * w**2), n)
    wh = w + rng.normal(0, np.sqrt(a[2] * v**2 + a[3] * w**2), n)
    gh = rng.normal(0, np.sqrt(a[4] * v**2 + a[5] * w**2), n)
    wh = np.where(np.abs(wh) < 1e-6, 1e-6, wh)
    th = x[:, 2]
    out = np.empty_like(x)
    out[:, 0] = x[:, 0] - vh / wh * np.sin(th) + vh / wh * np.sin(th + wh * dt)
    out[:, 1] = x[:, 1] + vh / wh * np.cos(th) - vh / wh * np.cos(th + wh * dt)
    out[:, 2] = wrap(th + wh * dt + gh * dt)
    return out

M = 3000
X0 = np.tile(np.array([0., 0., 0.]), (M, 1))
alpha = [0.02, 0.02, 0.02, 0.02, 0.005, 0.005]
velocity = (1.0, 0.35)
dts = [1, 3, 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
for ax, dt in zip(axes, dts):
    samples = sample_velocity(X0, velocity, dt, alpha)
    xy = samples[:, :2]
    mean = xy.mean(axis=0)
    std = xy.std(axis=0, ddof=1)
    print(f'dt={dt:>2} s | mean x,y = ({mean[0]:.4f}, {mean[1]:.4f}) m | std x,y = ({std[0]:.4f}, {std[1]:.4f}) m')
    ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.22)
    ax.scatter(mean[0], mean[1], c='red', marker='x', s=90, label='sample mean')
    ax.set_title(f'dt = {dt} s')
    ax.set_xlabel('x (m)')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
axes[0].set_ylabel('y (m)')
fig.suptitle('Week 7 Assignment 2 — Q1: Velocity motion model')
fig.tight_layout()
fig.savefig('week7_q1_motion_distributions.png', dpi=200)
print('Saved: week7_q1_motion_distributions.png')
plt.show()

# Week 7 Assignment 2 - Question 2
# Effect of alpha1 through alpha4 on motion samples

import numpy as np
import matplotlib.pyplot as plt

M = 3000
X0 = np.zeros((M, 3))
base_alpha = [0.02, 0.02, 0.02, 0.02, 0.005, 0.005]

cases = [("Baseline", base_alpha.copy())]

for i in range(4):
    modified = base_alpha.copy()
    modified[i] *= 10
    cases.append(("alpha{} x10".format(i + 1), modified))

fig, axes = plt.subplots(1, 5, figsize=(19, 4),
                         sharex=True, sharey=True)

for ax, (name, alpha) in zip(axes, cases):
    rng = np.random.default_rng(1)
    samples = sample_velocity(X0, (1.0, 0.35), 3.0, alpha)

    ax.scatter(samples[:, 0], samples[:, 1],
               s=2, alpha=0.35)
    ax.plot(samples[:, 0].mean(),
            samples[:, 1].mean(), "rx", markersize=8)

    ax.set_title(name)
    ax.set_xlim(-3, 8)
    ax.set_ylim(-3, 7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x (m)")

    print("{}: mean={}, std={}".format(
        name,
        np.round(samples[:, :2].mean(axis=0), 3),
        np.round(samples[:, :2].std(axis=0), 3)
    ))

axes[0].set_ylabel("y (m)")
fig.suptitle("Week 7 Assignment 2 - Q2")
plt.tight_layout()
plt.savefig("week7_q2_alpha_comparison.png", dpi=200)
plt.show()
    