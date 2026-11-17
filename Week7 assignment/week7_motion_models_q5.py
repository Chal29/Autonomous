"""Week7 Assignment , Question 5: velocity-model likelihood vs Monte Carlo poses.
Uses the squared-velocity variance convention in the professor's lecture.
The Table 5.1 score is a relative likelihood, not a normalized pose density.
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def noise_variances(v, w, alpha):
    return (alpha[0]*v**2 + alpha[1]*w**2,
            alpha[2]*v**2 + alpha[3]*w**2,
            alpha[4]*v**2 + alpha[5]*w**2)

def gaussian(error, variance):
    return np.exp(-0.5 * np.asarray(error)**2 / variance) / np.sqrt(2*np.pi*variance)

def sample_velocity(x0, v, w, dt, alpha, n):
    bv, bw, bg = noise_variances(v, w, alpha)
    vh = v + rng.normal(0, np.sqrt(bv), n)
    wh = w + rng.normal(0, np.sqrt(bw), n)
    gh = rng.normal(0, np.sqrt(bg), n)
    half = wh * dt / 2
    distance = vh * dt * np.sinc(half / np.pi)
    return np.column_stack((x0[0] + distance*np.cos(x0[2]+half),
                            x0[1] + distance*np.sin(x0[2]+half),
                            wrap(x0[2] + wh*dt + gh*dt))), vh

def table51_score(xp, yp, tp, x0, v, w, dt, alpha):
    """Vectorized Table 5.1 relative likelihood; returns NaN at singular points."""
    x, y, theta = x0
    dx, dy = xp-x, yp-y
    denom = dy*np.cos(theta) - dx*np.sin(theta)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = 0.5*(dx*np.cos(theta)+dy*np.sin(theta))/denom
        xc = (x+xp)/2 + mu*(y-yp)
        yc = (y+yp)/2 + mu*(xp-x)
        radius = np.hypot(x-xc, y-yc)
        dtheta = wrap(np.arctan2(yp-yc, xp-xc)-np.arctan2(y-yc, x-xc))
        vh = np.where(dx*np.cos(theta)+dy*np.sin(theta)<0, -1, 1)*abs(dtheta)*radius/dt
        wh = dtheta/dt
        gh = wrap(tp-theta-dtheta)/dt
        bv,bw,bg = noise_variances(v,w,alpha)
        score = gaussian(v-vh,bv)*gaussian(w-wh,bw)*gaussian(gh,bg)
    return np.where(abs(denom)<1e-10, np.nan, score)

def main():
    start = np.array([0.,0.,0.])
    v,w,dt = 1.,0.35,3.
    alpha = np.array([.02,.02,.02,.02,.005,.005])
    n=3000
    poses, sampled_v = sample_velocity(start,v,w,dt,alpha,n)
    bv,bw,bg = noise_variances(v,w,alpha)
    scores = table51_score(poses[:,0],poses[:,1],poses[:,2],start,v,w,dt,alpha)
    expected_heading=wrap(start[2]+w*dt)
    heading_tolerance=.15
    near_heading=np.abs(wrap(poses[:,2]-expected_heading))<=heading_tolerance
    x_grid=np.linspace(np.percentile(poses[:,0],.5)-.4,np.percentile(poses[:,0],99.5)+.4,150)
    y_grid=np.linspace(np.percentile(poses[:,1],.5)-.4,np.percentile(poses[:,1],99.5)+.4,150)
    X,Y=np.meshgrid(x_grid,y_grid)
    Z=table51_score(X,Y,expected_heading,start,v,w,dt,alpha)
    Z=np.ma.masked_invalid(Z)

    fig, ax=plt.subplots(1,3,figsize=(15,4.8))
    ax[0].scatter(poses[:,0],poses[:,1],s=3,alpha=.24,color='tab:blue')
    ax[0].plot(start[0],start[1],'o',color='tab:orange',label='Start')
    ax[0].set(title='Monte Carlo final positions',xlabel='x (m)',ylabel='y (m)')
    ax[0].axis('equal');ax[0].grid(alpha=.2);ax[0].legend()

    xs=np.linspace(v-4*np.sqrt(bv),v+4*np.sqrt(bv),300)
    ax[1].hist(sampled_v,bins=50,density=True,alpha=.5,label='Monte Carlo')
    ax[1].plot(xs,gaussian(xs-v,bv),lw=2,label='Theoretical Gaussian')
    ax[1].set(title='Check of sampled velocity noise',xlabel='Velocity (m/s)',ylabel='Density')
    ax[1].grid(alpha=.2);ax[1].legend()

    levels=np.linspace(float(Z.min()),float(Z.max()),12)
    contour=ax[2].contourf(X,Y,Z,levels=levels,cmap='viridis',alpha=.85)
    ax[2].scatter(poses[near_heading,0],poses[near_heading,1],s=4,color='white',alpha=.45,
                  label='Samples within ±0.15 rad')
    ax[2].set(title='Table 5.1 score at fixed heading',xlabel='x (m)',ylabel='y (m)')
    ax[2].axis('equal');ax[2].legend(loc='lower right',fontsize=8)
    fig.colorbar(contour,ax=ax[2],label='Relative likelihood')
    fig.suptitle('Q5: Velocity motion model — Monte Carlo vs closed-form score',fontsize=13)
    fig.tight_layout()
    fig.savefig('week7_q5_corrected_comparison.png',dpi=200)

    print('Monte Carlo samples:',n)
    print('Sample mean pose:',np.round(poses.mean(axis=0),4))
    print('Velocity mean/std:',round(sampled_v.mean(),4),round(sampled_v.std(),4))
    print('Theoretical velocity mean/std:',v,round(np.sqrt(bv),4))
    print('Finite Table 5.1 evaluations:',int(np.isfinite(scores).sum()))
    print('Mean relative likelihood score:',round(float(np.nanmean(scores)),5))
    print('Samples in fixed-heading slice:',int(near_heading.sum()))
    print('Saved: week7_q5_corrected_comparison.png')
    plt.show()

if __name__=='__main__':
    main()
