"""Week 4 Assignment 3: reconstructed EKF/UKF experiment.
This is NOT the professor's original distributed script.
Usage: python ekf_vs_ukf.py --runs 3 --seed 42 [--sigma0 10] [--alpha 0.1] [--nowrap]
"""
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import chi2

DT, STEPS = 0.1, 200
A = np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]],float)
PROCESS = np.diag([0.001,0.001,0.002,0.002])
Q = np.diag([0.5**2,np.deg2rad(4)**2])

def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def h(x):
    return np.array([np.hypot(x[0],x[1]),np.arctan2(x[1],x[0])])

def H_jac(x):
    px,py=x[:2]
    r2=max(px*px+py*py,1e-10)
    r=np.sqrt(r2)
    return np.array([[px/r,py/r,0,0],[-py/r2,px/r2,0,0]])

def sigma_points(mu,P,alpha=1e-3,kappa=0.,beta=2.):
    n=len(mu)
    lam=alpha**2*(n+kappa)-n
    L=np.linalg.cholesky((n+lam)*P)
    X=np.vstack([mu,mu+L.T,mu-L.T])
    wm=np.full(2*n+1,1/(2*(n+lam)))
    wm[0]=lam/(n+lam)
    wc=wm.copy()
    wc[0]+=1-alpha**2+beta
    return X,wm,wc

def ekf_correct(mu,P,z,nowrap=False):
    H=H_jac(mu)
    S=H@P@H.T+Q
    K=np.linalg.solve(S,H@P).T
    innovation=z-h(mu)
    if not nowrap: innovation[1]=wrap(innovation[1])
    m=mu+K@innovation
    I=np.eye(4)
    B=I-K@H
    cov=B@P@B.T+K@Q@K.T  # Joseph form
    return m,0.5*(cov+cov.T),innovation

def ukf_correct(mu,P,z,alpha=1e-3,nowrap=False):
    X,wm,wc=sigma_points(mu,P,alpha=alpha)
    Z=np.array([h(x) for x in X])
    # Circular mean for bearing avoids the branch-cut artifact.
    zhat=np.array([wm@Z[:,0],np.arctan2(wm@np.sin(Z[:,1]),wm@np.cos(Z[:,1]))])
    dZ=Z-zhat
    if not nowrap: dZ[:,1]=wrap(dZ[:,1])
    dX=X-mu
    S=np.einsum('i,ij,ik->jk',wc,dZ,dZ)+Q
    Cxz=np.einsum('i,ij,ik->jk',wc,dX,dZ)
    K=np.linalg.solve(S,Cxz.T).T
    innovation=z-zhat
    if not nowrap: innovation[1]=wrap(innovation[1])
    m=mu+K@innovation
    cov=P-K@S@K.T
    return m,0.5*(cov+cov.T),innovation

def simulate(seed,sigma0=1.,alpha=1e-3,nowrap=False):
    rng=np.random.default_rng(seed)
    # Target passes 0.5 m above sensor origin at t=10 s.
    truth=np.array([-10.,0.5,1.,0.])
    prior=np.array([-10.2,0.7,1.,0.])
    P0=np.diag([0.3**2,0.3**2,0.1**2,0.1**2])*sigma0
    means={'EKF':prior.copy(),'UKF':prior.copy()}
    covs={'EKF':P0.copy(),'UKF':P0.copy()}
    sqerr={k:[] for k in means}; nees={k:[] for k in means}
    first_large_innov={k:None for k in means}
    for t in range(STEPS):
        truth=A@truth
        z=h(truth)+rng.multivariate_normal(np.zeros(2),Q)
        z[1]=wrap(z[1])
        for key in means:
            mp=A@means[key]; Pp=A@covs[key]@A.T+PROCESS
            if key=='EKF': m,P,innov=ekf_correct(mp,Pp,z,nowrap)
            else: m,P,innov=ukf_correct(mp,Pp,z,alpha,nowrap)
            if first_large_innov[key] is None and abs(innov[1])>np.pi:
                first_large_innov[key]=(t+1,float(innov[1]))
            means[key],covs[key]=m,P
            err=m-truth
            sqerr[key].append(float(err[:2]@err[:2]))
            nees[key].append(float(err@np.linalg.solve(P,err)))
    return {k:{'nees_series':nees[k], 'rmse_m':float(np.sqrt(np.mean(sqerr[k]))),'mean_nees':float(np.mean(nees[k])),
               'first_bearing_innovation_over_pi':first_large_innov[k]} for k in means}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--sigma0',type=float,default=1.)
    ap.add_argument('--alpha',type=float,default=1e-3)
    ap.add_argument('--nowrap',action='store_true')
    ap.add_argument('--plot',action='store_true',help='Save ANEES graph and per-step CSV')
    ap.add_argument('--output-prefix',default='nees_consistency')
    a=ap.parse_args()
    rows=[]
    successful_seeds=[]
    for j in range(a.runs):
        try:
            rows.append(simulate(a.seed+j,a.sigma0,a.alpha,a.nowrap))
            successful_seeds.append(a.seed+j)
        except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:
            print(f'seed {a.seed+j}: numerical failure: {exc}')
    if a.plot and rows:
        n = len(rows)
        dof = 4*n
        low, high = chi2.ppf([0.025,0.975], dof)/n
        curves = {key:np.mean([r[key]['nees_series'] for r in rows],axis=0)
                  for key in ('EKF','UKF')}
        outcsv = a.output_prefix + '.csv'
        with open(outcsv,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f)
            w.writerow(['time_s','EKF_ANEES','UKF_ANEES','lower_95','upper_95'])
            for i in range(STEPS):
                w.writerow([round((i+1)*DT,6),curves['EKF'][i],curves['UKF'][i],low,high])
        fig,ax=plt.subplots(figsize=(10,5))
        times=np.arange(1,STEPS+1)*DT
        ax.plot(times,curves['EKF'],label='EKF ANEES')
        ax.plot(times,curves['UKF'],label='UKF ANEES')
        ax.axhline(4,linestyle=':',color='black',label='Expected NEES = 4')
        ax.axhspan(low,high,alpha=0.15,label='95% pointwise bounds')
        ax.set(xlabel='Time (s)',ylabel='Average NEES',
               title=f'EKF vs UKF consistency ({n} successful runs)')
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        outpng=a.output_prefix+'.png'
        fig.savefig(outpng,dpi=160)
        plt.close(fig)
        print(f'95% pointwise ANEES bounds for {n} runs: [{low:.3f}, {high:.3f}]')
        print(f'Saved {outcsv} and {outpng}')
    for key in ('EKF','UKF'):
        vals=np.array([r[key]['rmse_m'] for r in rows])
        ns=np.array([r[key]['mean_nees'] for r in rows])
        if len(vals):
            print(f'{key}: RMSE mean={vals.mean():.4f} m, sample SD={vals.std(ddof=1) if len(vals)>1 else 0:.4f} m; mean NEES={ns.mean():.3f}')
            for j,r in enumerate(rows):
                print(f'  seed={successful_seeds[j]}: RMSE={r[key]["rmse_m"]:.4f} m, NEES={r[key]["mean_nees"]:.3f}, first |bearing innovation|>pi={r[key]["first_bearing_innovation_over_pi"]}')

if __name__=='__main__': main()
