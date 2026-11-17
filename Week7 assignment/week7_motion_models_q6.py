"""Week7 Assignment  Q6: Map-aware motion model (lecture Eq. 6.9).
Demonstrates why endpoint-only occupancy checks can miss wall crossings,
and why checking intermediate states with smaller dt helps.
Requires numpy and matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

# Map: vertical wall at x=0, except a narrow doorway at y in [-0.25, 0.25].
WALL_X = 0.0
DOOR_HALF_WIDTH = 0.25
ROBOT_RADIUS = 0.08

def is_free(x, y):
    """Point robot-center collision check against a wall with doorway."""
    # This idealized wall has zero thickness; check when crossing x=0 separately.
    return True

def crosses_wall_illegally(p0, p1):
    """Exact segment collision test for the idealized wall."""
    x0, y0 = p0
    x1, y1 = p1
    if (x0 < WALL_X and x1 > WALL_X) or (x0 > WALL_X and x1 < WALL_X):
        f = (WALL_X - x0) / (x1 - x0)
        y_cross = y0 + f * (y1 - y0)
        return abs(y_cross) > DOOR_HALF_WIDTH - ROBOT_RADIUS
    return False

# Smooth path: endpoints are in free space, but its middle crosses the wall outside door.
# The trajectory is deliberately chosen to cross x=0 at y=1.0.
T = 2.0

def trajectory(t):
    return np.array([-1.0 + t, 1.0 + 0.35 * np.sin(np.pi * t / T)])

# Endpoint-only approximation of Eq. (6.9): free final poses get nonzero score.
# A coarse 2-second step misses the wall crossing because both endpoints are free.
steps = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
print('dt (s) | checked segments | collision detected')
print('-' * 47)
results = []
for dt in steps:
    times = np.linspace(0.0, T, int(round(T / dt)) + 1)
    poses = np.array([trajectory(t) for t in times])
    # For demonstration, a discrete endpoint-only occupancy check needs a finite wall width.
    # Wall obstacle: |x| <= 0.12 and |y| >= 0.25 (robot-radius inflated).
    hits = (np.abs(poses[:, 0]) <= 0.12 + ROBOT_RADIUS) & (np.abs(poses[:, 1]) >= DOOR_HALF_WIDTH - ROBOT_RADIUS)
    detected = bool(np.any(hits))
    results.append((dt, len(times)-1, detected))
    print('{:6.2f} | {:16d} | {}'.format(dt, len(times)-1, detected))

fine_t = np.linspace(0, T, 501)
curve = np.array([trajectory(t) for t in fine_t])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax in axes:
    ax.plot([0, 0], [-1.5, -DOOR_HALF_WIDTH], 'k-', lw=8)
    ax.plot([0, 0], [DOOR_HALF_WIDTH, 2.0], 'k-', lw=8)
    ax.axhspan(-DOOR_HALF_WIDTH, DOOR_HALF_WIDTH, xmin=0.48, xmax=0.52, color='lightgreen', alpha=.3)
    ax.plot(curve[:, 0], curve[:, 1], color='steelblue', lw=2, label='Actual continuous path')
    ax.set(xlim=(-1.4, 1.4), ylim=(-0.8, 1.9), xlabel='x (m)', ylabel='y (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
for ax, dt in zip(axes, [2.0, 0.1]):
    times = np.linspace(0, T, int(round(T/dt))+1)
    sampled = np.array([trajectory(t) for t in times])
    ax.scatter(sampled[:, 0], sampled[:, 1], s=24, color='darkorange', zorder=5, label='Checked poses')
    ax.set_title('Endpoint checks, dt = {:.1f} s'.format(dt))
    ax.legend(fontsize=8, loc='lower left')
fig.suptitle('Question 6: map-aware endpoint approximation and wall crossing')
fig.tight_layout()
fig.savefig("week7_q6_map_aware_comparison.png", dpi=180)
plt.show()
plt.show()
print('\nExact segment collision test:', crosses_wall_illegally(trajectory(0), trajectory(T)))
