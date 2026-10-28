"""Week 5, Assignment 4: reproducible experiments Q1-Q6.

Install: pip install numpy matplotlib
Examples: python particle_filter_assignment.py --experiment q1
          python particle_filter_assignment.py --experiment q4 --seed 0
          python particle_filter_assignment.py --M 2000 --sigr 0.05 --lm 2

Baseline matches the supplied Python example: measurement-only weights each step.
--weight-mode recursive for a sequential importance filter when resampling is skipped.
All figures and CSVs are written to --outdir (default: week05_outputs).
"""
import argparse
import csv
from pathlib import Path
import numpy as np

LM_ALL = np.array([[2., 2.], [8., 3.], [5., 9.], [1., 8.]])
SIG_MOT = np.array([0.08, 0.08, 0.04])


def motion_sample(P, rng, noise_scale=1.):
    n = len(P)
    out = P.copy()
    th = P[:, 2]
    out[:, 0] += .5*np.cos(th) + rng.normal(0, .08*noise_scale, n)
    out[:, 1] += .5*np.sin(th) + rng.normal(0, .08*noise_scale, n)
    out[:, 2] = (th + .12 + rng.normal(0, .04*noise_scale, n) + np.pi) % (2*np.pi)-np.pi
    return out


def likelihood(P, z, landmarks, sigma):
    d = np.linalg.norm(P[:, None, :2] - landmarks[None, :, :], axis=2)
    return -.5*np.square((d-z)/sigma).sum(axis=1)


def normalize_logweights(logw):
    w = np.exp(logw-logw.max())
    return w/w.sum()


def resample_indices(w, rng, sampler):
    m = len(w)
    cdf = np.cumsum(w)
    if sampler == 'low':
        draws = (rng.random()+np.arange(m))/m
    else:
        draws = rng.random(m)
    return np.minimum(np.searchsorted(cdf, draws), m-1)


def simulate(M=2000, seed=0, sigr=.35, lm=4, resample='neff', sampler='low',
             kidnap=0, inject=0., weight_mode='baseline', temper=1.,
             motion_noise_scale=1., steps=25, snapshots=()):
    rng = np.random.default_rng(seed)
    landmarks = LM_ALL[:lm]
    P = np.column_stack((rng.uniform(0,10,M), rng.uniform(0,10,M), rng.uniform(-np.pi,np.pi,M)))
    truth = np.array([3.,3.,.3])
    w = np.full(M, 1/M)
    records, frames = [], {}
    for t in range(1, steps+1):
        if kidnap and t == kidnap:
            truth = np.array([7.5, 7., -2.])
        truth = motion_sample(truth[None,:],rng)[0]
        z = np.linalg.norm(truth[:2]-landmarks,axis=1)+rng.normal(0,sigr,lm)
        P = motion_sample(P,rng,motion_noise_scale)
        if inject:
            k = min(M, max(1, round(inject*M)))
            ix = rng.choice(M,k,replace=False)
            P[ix] = np.column_stack((rng.uniform(0,10,k),rng.uniform(0,10,k),rng.uniform(-np.pi,np.pi,k)))
            # Randomly injected particles are given a prior mass rather than inheriting arbitrary old weights.
            w[ix] = 1/M
            w /= w.sum()
        logw = likelihood(P,z,landmarks,sigr)/temper
        if weight_mode == 'recursive':
            logw += np.log(np.maximum(w,1e-300))
        w = normalize_logweights(logw)
        ess = 1/np.sum(w*w)
        did = resample == 'always' or (resample == 'neff' and ess < M/2)
        unique = M
        if did:
            idx = resample_indices(w,rng,sampler)
            unique = len(np.unique(idx))
            P = P[idx]
            w = np.full(M,1/M)
        est = w@P[:,:2]
        err = float(np.linalg.norm(est-truth[:2]))
        spread = float(P[:,:2].std(axis=0).mean())
        records.append(dict(step=t,error_m=err,ess=float(ess),spread_m=spread,
                            resampled=int(did),unique_ancestors=unique,
                            truth_x=float(truth[0]),truth_y=float(truth[1]),
                            est_x=float(est[0]),est_y=float(est[1])))
        if t in snapshots:
            frames[t] = (P[:,:2].copy(),truth[:2].copy(),est.copy(),landmarks.copy())
    return records, frames


def save_csv(rows,path):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plot_lines(data, title, ylabel, path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for name, records in data.items():
        ax.plot([r['step'] for r in records],[r[ylabel] for r in records],label=name)
    ax.set(xlabel='Step',ylabel=ylabel.replace('_',' '),title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=170); plt.close(fig)


def plot_snapshots(frames, path, title):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(frames),figsize=(4*len(frames),4),squeeze=False)
    for ax,(t,(P,truth,est,lm)) in zip(axes[0],frames.items()):
        ax.scatter(P[:,0],P[:,1],s=2,alpha=.17,label='Particles')
        ax.scatter(lm[:,0],lm[:,1],marker='^',s=65,label='Landmarks')
        ax.scatter(*truth,marker='*',s=110,label='Truth')
        ax.scatter(*est,marker='x',s=80,label='Mean')
        ax.set(xlim=(-1,12),ylim=(-1,12),xlabel='x (m)',ylabel='y (m)',title=f'Step {t}')
        ax.set_aspect('equal')
    axes[0,0].legend(fontsize=7);fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def experiment(args):
    out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    q=args.experiment
    if q=='q1':
        rows=[]
        for m in (100,500,2000,10000):
            for seed in range(args.runs):
                rec,_=simulate(M=m,seed=seed,weight_mode=args.weight_mode)
                rows.append(dict(M=m,seed=seed,final_error_m=rec[-1]['error_m'],
                                 success_0_5m=int(rec[-1]['error_m']<=.5),
                                 resampling_steps=sum(r['resampled'] for r in rec)))
        save_csv(rows,out/'q1_particle_sweep.csv')
        print('Q1: success defined as final error <= 0.5 m')
        for m in (100,500,2000,10000):
            a=[r for r in rows if r['M']==m]
            print(f'M={m:5d}: {sum(r["success_0_5m"] for r in a)}/{len(a)} = '
                  f'{100*np.mean([r["success_0_5m"] for r in a]):.1f}%, '
                  f'mean error={np.mean([r["final_error_m"] for r in a]):.3f} m')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4));ax.bar([str(m) for m in (100,500,2000,10000)],
              [100*np.mean([r['success_0_5m'] for r in rows if r['M']==m]) for m in (100,500,2000,10000)])
        ax.set(xlabel='Particles (M)',ylabel='Success rate (%)',ylim=(0,105),title='Q1: Global localization success')
        fig.tight_layout();fig.savefig(out/'q1_success.png',dpi=170);plt.close(fig)
    elif q=='q2':
        traces={}
        for policy in ('neff','always'):
            rec,_=simulate(M=args.M,seed=args.seed,resample=policy,weight_mode=args.weight_mode)
            traces[policy]=rec
            save_csv(rec,out/f'q2_{policy}.csv')
            print(f'{policy}: final error={rec[-1]["error_m"]:.3f} m, resampling steps={sum(r["resampled"] for r in rec)}')
        plot_lines(traces,'Q2: Particle spread by resampling policy','spread_m',out/'q2_spread.png')
        plot_lines(traces,'Q2: Localization error','error_m',out/'q2_error.png')
    elif q=='q3':
        configs={'baseline_sigma_035':dict(sigr=.35),'precise_sigma_005':dict(sigr=.05),
                 'tempered_sigma_005':dict(sigr=.05,temper=5.),
                 'more_motion_noise_sigma_005':dict(sigr=.05,motion_noise_scale=2.)}
        traces={}
        for name,opts in configs.items():
            rec,frames=simulate(M=args.M,seed=args.seed,snapshots=(1,5,15,25),
                                weight_mode=args.weight_mode,**opts)
            traces[name]=rec;save_csv(rec,out/f'q3_{name}.csv')
            plot_snapshots(frames,out/f'q3_{name}_particles.png',name)
            print(f'{name}: final error={rec[-1]["error_m"]:.3f} m')
        plot_lines(traces,'Q3: Precise sensor paradox','error_m',out/'q3_error.png')
    elif q=='q4':
        rec,frames=simulate(M=args.M,seed=args.seed,lm=2,snapshots=(1,4,8,15,25),weight_mode=args.weight_mode)
        save_csv(rec,out/'q4_two_landmarks.csv')
        plot_snapshots(frames,out/'q4_ambiguity.png','Q4: Two-landmark hypotheses')
        print(f'Q4 final error={rec[-1]["error_m"]:.3f} m')
    elif q=='q5':
        rows=[]
        for sampler in ('low','multinomial'):
            for seed in range(args.repeats):
                rec,_=simulate(M=args.M,seed=seed,sampler=sampler,weight_mode=args.weight_mode)
                rows.append(dict(sampler=sampler,seed=seed,final_error_m=rec[-1]['error_m'],
                                 final_est_x=rec[-1]['est_x'],final_est_y=rec[-1]['est_y'],
                                 resampling_steps=sum(r['resampled'] for r in rec)))
        save_csv(rows,out/'q5_resampling_comparison.csv')
        for sampler in ('low','multinomial'):
            a=[r for r in rows if r['sampler']==sampler]
            print(f'{sampler}: mean error={np.mean([r["final_error_m"] for r in a]):.3f} m, '
                  f'variance of final error={np.var([r["final_error_m"] for r in a],ddof=1):.4f} m²')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        ax.boxplot([[r['final_error_m'] for r in rows if r['sampler']==s] for s in ('low','multinomial')],
                   tick_labels=['Low variance','Multinomial'])
        ax.set(ylabel='Final localization error (m)',title=f'Q5: {args.repeats} repetitions')
        fig.tight_layout();fig.savefig(out/'q5_error_boxplot.png',dpi=170);plt.close(fig)
    elif q=='q6':
        traces={}
        for name,inj in (('without_injection',0.),('with_injection',.02)):
            rec,frames=simulate(M=args.M,seed=args.seed,kidnap=15,inject=inj,
                                snapshots=(14,15,18,25),weight_mode=args.weight_mode)
            traces[name]=rec;save_csv(rec,out/f'q6_{name}.csv')
            plot_snapshots(frames,out/f'q6_{name}_particles.png',name)
            print(f'{name}: final error={rec[-1]["error_m"]:.3f} m')
        plot_lines(traces,'Q6: Kidnapped robot at step 15','error_m',out/'q6_error.png')
    elif q=='single':
        rec,frames=simulate(M=args.M,seed=args.seed,sigr=args.sigr,lm=args.lm,
             resample=args.resample,sampler=args.sampler,kidnap=args.kidnap,
             inject=args.inject,weight_mode=args.weight_mode,snapshots=(1,5,15,25))
        save_csv(rec,out/'single_run.csv');plot_snapshots(frames,out/'single_particles.png','Particle filter')
        print(f'Final error: {rec[-1]["error_m"]:.3f} m')
    else: raise ValueError(q)
    print('Saved output in',out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment',choices=('q1','q2','q3','q4','q5','q6','single'),default='single')
    p.add_argument('--M',type=int,default=2000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--sigr',type=float,default=.35)
    p.add_argument('--lm',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--resample',choices=('neff','always','never'),default='neff')
    p.add_argument('--sampler',choices=('low','multinomial'),default='low')
    p.add_argument('--kidnap',type=int,default=0)
    p.add_argument('--inject',type=float,default=0.)
    p.add_argument('--weight-mode',choices=('baseline','recursive'),default='baseline')
    p.add_argument('--runs',type=int,default=20)
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--outdir',default='week05_outputs')
    a=p.parse_args()
    if a.M<=0 or a.runs<=0 or a.repeats<=1 or a.sigr<=0 or a.inject<0 or a.inject>1: p.error('Invalid parameter range')
    experiment(a)
