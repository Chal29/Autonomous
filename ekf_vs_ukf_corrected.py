"""Reproducible reconstructed Week 4 Assignment 3 experiment; not instructor source.
Usage: python ekf_vs_ukf_corrected.py --runs 3 --seed 42 --plot
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
DT, STEPS = .1, 200
A=np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]],float)
PROCESS=np.diag([.001,.001,.002,.002]); Q=np.diag([.5**2,np.deg2rad(4)**2])
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def h(x): return np.array([np.hypot(x[0],x[1]),np.arctan2(x[1],x[0])])
def H_jac(x):
    px,py=x[:2]; r2=max(px*px+py*py,1e-10); r=np.sqrt(r2)
    return np.array([[px/r,py/r,0,0],[-py/r2,px/r2,0,0]])
def sigma_points(mu,P,alpha):
    n=len(mu); lam=alpha*alpha*n-n
    L=np.linalg.cholesky((n+lam)*(.5*(P+P.T)))
    X=np.vstack((mu,mu+L.T,mu-L.T))
    wm=np.full(2*n+1,1/(2*(n+lam))); wm[0]=lam/(n+lam)
    wc=wm.copy(); wc[0]+=1-alpha*alpha+2
    return X,wm,wc
def ekf_correct(mu,P,z,nowrap):
    H=H_jac(mu); S=H@P@H.T+Q; K=np.linalg.solve(S,H@P).T
    innovation=z-h(mu)
    if not nowrap: innovation[1]=wrap(innovation[1])
    m=mu+K@innovation; B=np.eye(4)-K@H
    cov=B@P@B.T+K@Q@K.T
    return m,.5*(cov+cov.T),innovation
def ukf_correct(mu,P,z,alpha,nowrap):
    X,wm,wc=sigma_points(mu,P,alpha); Z=np.array([h(x) for x in X])
    if nowrap:
        zhat=wm@Z
    else:
        # Central-reference angle mean avoids +/-pi branch cut in sigma points.
        ref=Z[0,1]; zhat=np.array([wm@Z[:,0],wrap(ref+wm@wrap(Z[:,1]-ref))])
    dZ=Z-zhat
    if not nowrap: dZ[:,1]=wrap(dZ[:,1])
    dX=X-mu
    S=np.einsum('i,ij,ik->jk',wc,dZ,dZ)+Q
    S=.5*(S+S.T)
    if np.linalg.eigvalsh(S).min()<=0: raise ValueError('UKF innovation covariance S is not positive definite')
    C=np.einsum('i,ij,ik->jk',wc,dX,dZ)
    K=np.linalg.solve(S,C.T).T
    innovation=z-zhat
    if not nowrap: innovation[1]=wrap(innovation[1])
    m=mu+K@innovation; cov=P-K@S@K.T; cov=.5*(cov+cov.T)
    if np.linalg.eigvalsh(cov).min()<=0: raise ValueError('UKF posterior covariance is not positive definite')
    return m,cov,innovation
def simulate(seed,sigma0,alpha,nowrap):
    rng=np.random.default_rng(seed)
    truth=np.array([-10.,.5,1.,0.]); prior=np.array([-10.2,.7,1.,0.])
    P0=np.diag([9.,9.,1.,1.])*sigma0
    means={'EKF':prior.copy(),'UKF':prior.copy()}; covs={k:P0.copy() for k in means}
    rec={k:{'sqerr':[],'nees':[],'near_sqerr':[],'near_nees':[],'first_raw_over_pi':None,'first_error_over_1m':None} for k in means}
    for t in range(STEPS):
        truth=A@truth+rng.multivariate_normal(np.zeros(4),PROCESS)
        z=h(truth)+rng.multivariate_normal(np.zeros(2),Q); z[1]=wrap(z[1])
        for k in means:
            mp=A@means[k]; Pp=A@covs[k]@A.T+PROCESS
            try:
                if k=='EKF': m,P,innov=ekf_correct(mp,Pp,z,nowrap)
                else: m,P,innov=ukf_correct(mp,Pp,z,alpha,nowrap)
                e=m-truth; err=float(np.linalg.norm(e[:2]))
                nv=float(e@np.linalg.solve(P,e))
                if not np.isfinite(nv): raise ValueError('non-finite NEES')
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                raise RuntimeError('%s at step %d: %s'%(k,t+1,exc))
            if rec[k]['first_raw_over_pi'] is None and abs(innov[1])>np.pi: rec[k]['first_raw_over_pi']=(t+1,float(innov[1]))
            if rec[k]['first_error_over_1m'] is None and err>1.: rec[k]['first_error_over_1m']=(t+1,err,float(innov[1]))
            means[k],covs[k]=m,P
            rec[k]['sqerr'].append(err*err); rec[k]['nees'].append(nv)
            if 9<= (t+1)*DT <=11:
                rec[k]['near_sqerr'].append(err*err); rec[k]['near_nees'].append(nv)
    for k in rec:
        r=rec[k]; r['rmse']=float(np.sqrt(np.mean(r['sqerr']))); r['mean_nees']=float(np.mean(r['nees']))
        r['near_rmse']=float(np.sqrt(np.mean(r['near_sqerr']))); r['near_mean_nees']=float(np.mean(r['near_nees']))
    return rec
def main():
    p=argparse.ArgumentParser(); p.add_argument('--runs',type=int,default=3); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sigma0',type=float,default=1); p.add_argument('--alpha',type=float,default=.001)
    p.add_argument('--nowrap',action='store_true'); p.add_argument('--plot',action='store_true'); p.add_argument('--output-prefix',default='corrected')
    a=p.parse_args(); rows=[]; failed=[]
    for seed in range(a.seed,a.seed+a.runs):
        try: rows.append((seed,simulate(seed,a.sigma0,a.alpha,a.nowrap)))
        except RuntimeError as e: failed.append((seed,str(e)))
    print('Requested %d; successful %d; failed %d'%(a.runs,len(rows),len(failed)))
    for seed,reason in failed: print('FAILED seed=%d %s'%(seed,reason))
    for k in ('EKF','UKF'):
        if not rows: continue
        for metric in ('rmse','mean_nees','near_rmse','near_mean_nees'):
            vals=np.array([r[k][metric] for _,r in rows]); print('%s %s: %.5f (SD %.5f)'%(k,metric,vals.mean(),vals.std(ddof=1) if len(vals)>1 else 0))
        for seed,r in rows: print('%s seed=%d first_raw_over_pi=%s first_error_over_1m=%s'%(k,seed,r[k]['first_raw_over_pi'],r[k]['first_error_over_1m']))
    if a.plot and rows:
        n=len(rows); low,high=chi2.ppf([.025,.975],4*n)/n
        curves={k:np.mean([r[k]['nees'] for _,r in rows],axis=0) for k in ('EKF','UKF')}
        with open(a.output_prefix+'.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['t_s','EKF_ANEES','UKF_ANEES','low_95','high_95'])
            for i in range(STEPS): w.writerow([(i+1)*DT,curves['EKF'][i],curves['UKF'][i],low,high])
        fig,ax=plt.subplots(figsize=(9,4)); t=np.arange(1,STEPS+1)*DT
        for k in curves: ax.plot(t,curves[k],label=k)
        ax.axhline(4,ls=':',color='black',label='Expected NEES = 4'); ax.axhspan(low,high,alpha=.15,label='95% pointwise bounds')
        ax.set(xlabel='Time (s)',ylabel='ANEES',title='EKF / UKF consistency (%d successful runs)'%n); ax.legend(); fig.tight_layout(); fig.savefig(a.output_prefix+'.png',dpi=160); plt.close(fig)
        print('95%% pointwise bounds [%.3f, %.3f]'%(low,high))
if __name__=='__main__': main()
