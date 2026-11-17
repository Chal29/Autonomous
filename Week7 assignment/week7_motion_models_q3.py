"""Week 7 Assignment  - Question 3: Gaussian approximation checks.
Compatible with Python 3.6 and NumPy/Matplotlib. Run: py week7_motion_models_q3.py
"""
import numpy as np
import matplotlib.pyplot as plt

ALPHA = [0.02, 0.02, 0.02, 0.02, 0.005, 0.005]
N = 3000
DT = 3.0

def wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def sample_velocity(x, u, dt, a, rng):
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

def sample_odometry(x, odom, a, rng):
    p0, p1 = odom
    drot1 = wrap(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]) - p0[2])
    dtrans = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
    drot2 = wrap(p1[2]-p0[2]-drot1)
    n = x.shape[0]
    s1 = np.sqrt(a[0]*drot1**2 + a[1]*dtrans**2)
    s2 = np.sqrt(a[2]*dtrans**2 + a[3]*(drot1**2 + drot2**2))
    s3 = np.sqrt(a[0]*drot2**2 + a[1]*dtrans**2)
    r1 = drot1 - rng.normal(0, s1, n)
    tr = dtrans - rng.normal(0, s2, n)
    r2 = drot2 - rng.normal(0, s3, n)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0] + tr * np.cos(x[:, 2] + r1)
    out[:, 1] = x[:, 1] + tr * np.sin(x[:, 2] + r1)
    out[:, 2] = wrap(x[:, 2] + r1 + r2)
    return out

def analyze(name, samples):
    xy = samples[:, :2]
    mean = xy.mean(axis=0)
    cov = np.cov(xy, rowvar=False, ddof=1)
    residuals = xy - mean
    # Solve is numerically preferable to explicitly forming the inverse.
    d2 = np.sum(residuals * np.linalg.solve(cov, residuals.T).T, axis=1)
    threshold = 5.991464547
    coverage = 100.0 * np.mean(d2 <= threshold)
    # For df=2, the chi-square CDF is F(z)=1-exp(-z/2), z>=0.
    ordered = np.sort(d2)
    expected = 1.0 - np.exp(-ordered / 2.0)
    empirical_hi = np.arange(1, len(d2)+1) / float(len(d2))
    empirical_lo = np.arange(len(d2)) / float(len(d2))
    ecdf_gap = max(np.max(np.abs(empirical_hi-expected)),
                   np.max(np.abs(empirical_lo-expected)))
    print('\n' + name)
    print('Mean [x, y] (m):', np.round(mean, 4))
    print('Covariance (m^2):\n', np.round(cov, 4))
    print('95% ellipse coverage: {:.2f}% (Gaussian reference: 95%)'.format(coverage))
    print('Max empirical CDF difference from chi-square(2): {:.4f}'.format(ecdf_gap))
    return xy, mean, cov, d2

def main():
    X0 = np.zeros((N, 3))
    # Separate deterministic random streams for repeatable comparison.
    vel = sample_velocity(X0, (1.0, 0.35), DT, ALPHA, np.random.default_rng(1))
    odom = sample_odometry(X0,
                           (np.array([0., 0., 0.]), np.array([2.60, 1.55, 1.05])),
                           ALPHA, np.random.default_rng(2))
    cases = [('Velocity model', analyze('Velocity model', vel)),
             ('Odometry model', analyze('Odometry model', odom))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    theta = np.linspace(0, 2*np.pi, 240)
    unit_circle = np.vstack((np.cos(theta), np.sin(theta)))
    for row, (name, (xy, mean, cov, d2)) in enumerate(cases):
        ax = axes[row, 0]
        ax.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.2)
        ax.plot(mean[0], mean[1], 'rx', markersize=9)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        ellipse = mean[:, None] + eigenvectors.dot(
            np.diag(np.sqrt(5.991464547*eigenvalues))).dot(unit_circle)
        ax.plot(ellipse[0], ellipse[1], 'r-', linewidth=2, label='95% Gaussian ellipse')
        ax.set_title(name + ': samples and Gaussian ellipse')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        ax = axes[row, 1]
        ax.hist(d2, bins=45, density=True, alpha=0.55, label='Sample squared distances')
        z = np.linspace(0, max(15, np.percentile(d2, 99.5)), 300)
        ax.plot(z, 0.5*np.exp(-z/2.0), 'r-', linewidth=2, label='Chi-square(2) density')
        ax.axvline(5.991464547, color='k', linestyle='--', label='95% threshold')
        ax.set_title(name + ': squared Mahalanobis distances')
        ax.set_xlabel('Squared Mahalanobis distance')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
    fig.tight_layout()
    filename = 'week7_q3_gaussian_comparison.png'
    fig.savefig(filename, dpi=180)
    print('\nSaved:', filename)
    plt.show()

if __name__ == '__main__':
    main()
