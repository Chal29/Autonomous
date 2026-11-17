"""Assignment 2 Q5: textbook velocity-motion likelihood vs Monte Carlo samples.
Based on Probabilistic Robotics, Chapter 5, Tables 5.1-5.3.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def normal_density(a, variance):
    return np.exp(-0.5 * np.asarray(a)**2 / variance) / np.sqrt(2*np.pi*variance)

def noise_variances(v, w, alpha):
    return (alpha[0]*abs(v) + alpha[1]*abs(w),
            alpha[2]*abs(v) + alpha[3]*abs(w),
            alpha[4]*abs(v) + alpha[5]*abs(w))

def sample_velocity(x0, v, w, dt, alpha, n, rng):
    bv, bw, bg = noise_variances(v, w, alpha)
    vh = v + rng.normal(0, np.sqrt(bv), n)
    wh = w + rng.normal(0, np.sqrt(bw), n)
    gh = rng.normal(0, np.sqrt(bg), n)
    # Stable limit of the circular-arc model when angular velocity is near zero
    half = wh*dt/2
    scale = np.sinc(half/np.pi)  # sin(half)/half
    distance = vh*dt*scale
    heading_mid = x0[2] + half
    out = np.column_stack((x0[0]+distance*np.cos(heading_mid),
                           x0[1]+distance*np.sin(heading_mid),
                           wrap(x0[2]+wh*dt+gh*dt)))
    return out, (vh, wh, gh)

def motion_model_velocity(x1, v, w, dt, x0, alpha):
    """Table 5.1: relative likelihood at a proposed pose (not a normalized pose PDF)."""
    x, y, theta = x0
    xp, yp, thetap = x1
    dx, dy = xp-x, yp-y
    denom = dy*np.cos(theta) - dx*np.sin(theta)
    if abs(denom) < 1e-10:
        return np.nan  # straight-line limiting case requires separate treatment
    mu = 0.5*(dx*np.cos(theta)+dy*np.sin(theta))/denom
    xc = (x+xp)/2 + mu*(y-yp)
    yc = (y+yp)/2 + mu*(xp-x)
    r = np.hypot(x-xc, y-yc)
    dtheta = wrap(np.arctan2(yp-yc, xp-xc)-np.arctan2(y-yc, x-xc))
    vh = dtheta*r/dt
    # Signed velocity for reverse motion
    if dx*np.cos(theta)+dy*np.sin(theta) < 0:
        vh = -abs(vh)
    else:
        vh = abs(vh)
    wh = dtheta/dt
    gh = wrap(thetap-theta-dtheta)/dt
    bv, bw, bg = noise_variances(v, w, alpha)
    return float(normal_density(v-vh,bv)*normal_density(w-wh,bw)*normal_density(gh,bg))

if __name__ == '__main__':
    start = np.array([0.0, 0.0, 0.0])
    v, w, dt = 1.0, 0.35, 3.0
    alpha = np.array([0.02, 0.02, 0.02, 0.02, 0.005, 0.005])
    n = 3000
    samples, (vh, wh, gh) = sample_velocity(start,v,w,dt,alpha,n,rng)
    bv, bw, bg = noise_variances(v,w,alpha)
    scores = np.array([motion_model_velocity(p,v,w,dt,start,alpha) for p in samples])
    finite = np.isfinite(scores)
    print('Monte Carlo samples:',n)
    print('Sample mean pose:', np.round(samples.mean(axis=0),4))
    print('Control-noise variances:', np.round([bv,bw,bg],5))
    print('Velocity sample mean/std:',round(vh.mean(),4),round(vh.std(),4))
    print('Theoretical velocity mean/std:',v,round(np.sqrt(bv),4))
    print('Finite Table 5.1 likelihood evaluations:',finite.sum())
    print('Mean relative likelihood score:',round(scores[finite].mean(),5))
    fig, ax = plt.subplots(1,3,figsize=(15,4.5))
    ax[0].scatter(samples[:,0],samples[:,1],s=3,alpha=.25)
    ax[0].scatter([start[0]],[start[1]],marker='o',s=65,label='Start')
    ax[0].set(xlabel='x (m)',ylabel='y (m)',title='Monte Carlo final poses')
    ax[0].axis('equal');ax[0].grid(alpha=.2);ax[0].legend()
    xs=np.linspace(v-4*np.sqrt(bv),v+4*np.sqrt(bv),300)
    ax[1].hist(vh,bins=50,density=True,alpha=.55,label='Sampled velocity')
    ax[1].plot(xs,normal_density(xs-v,bv),lw=2,label='Textbook Gaussian density')
    ax[1].set(xlabel='Sampled linear velocity (m/s)',ylabel='Density',title='Monte Carlo vs theoretical noise')
    ax[1].legend();ax[1].grid(alpha=.2)
    ax[2].hist(scores[finite],bins=45,alpha=.7)
    ax[2].set(xlabel='Table 5.1 relative likelihood',ylabel='Number of poses',title='Closed-form scores of samples')
    ax[2].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig('week7_q5_comparison.png',dpi=180)
    plt.show()
