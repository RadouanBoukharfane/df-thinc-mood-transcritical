import json,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path('/mnt/data/dmp_study'); OUT=ROOT/'dmp_results';OUT.mkdir(exist_ok=True)
old=json.loads(Path('/mnt/data/step9_refine/refinement_results/refinement_metrics.json').read_text())
oldmap={r['N']:r for r in old}

def measure(path,N):
 a=np.load(path);rho=a['rho'];Y=a['Y'];r0=a['rho0'];Y0=a['Y0'];x=a['x'];y=a['y'];dx=float(x[1]-x[0]);dy=float(y[1]-y[0]);X,Yg=np.meshgrid(x,y)
 exmax=float(r0.max());exmin=float(r0.min());jump=exmax-exmin;over=max(0.,float(rho.max()-exmax));under=max(0.,float(exmin-rho.min()))
 mixed=(Y>.1)&(Y<.9);inside=Y>=.5;inside0=Y0>=.5;j=N//2
 w=Y;sw=w.sum();xc=(w*X).sum()/sw;yc=(w*Yg).sum()/sw;vx=(w*(X-xc)**2).sum()/sw;vy=(w*(Yg-yc)**2).sum()/sw
 return dict(N=N,method='DMP-FCT r=1e-3',rho_L1=float(np.mean(abs(rho-r0))),Y_L1=float(np.mean(abs(Y-Y0))),rho_max=float(rho.max()),rho_min=float(rho.min()),overshoot_abs=over,overshoot_pct_jump=100*over/jump,undershoot_abs=under,mixed_cell_fraction=float(mixed.mean()),centerline_mixed_width_m=float(mixed[j].sum()*dx),threshold_area_rel_error=float(abs(inside.sum()-inside0.sum())/inside0.sum()),anisotropy=float(abs(vx-vy)/max(vx,vy)),mass_rel=float(abs(rho.sum()-r0.sum())/r0.sum()),fuel_mass_rel=float(abs((rho*Y).sum()-(r0*Y0).sum())/(r0*Y0).sum()))

rows=[]
for N in [25,50,100]:
 o=oldmap[N].copy();o['method']='sans DMP'; rows.append(o)
 p=ROOT/'results'/('N100_r0.001_full.npz' if N==100 else f'N{N}_fct_r0.001.npz')
 rows.append(measure(p,N))
# rates within each method
for method in ['sans DMP','DMP-FCT r=1e-3']:
 rr=sorted([r for r in rows if r['method']==method],key=lambda z:z['N'])
 for i,r in enumerate(rr):
  if i==0: continue
  prev=rr[i-1]
  for k in ['rho_L1','Y_L1','centerline_mixed_width_m','overshoot_abs']:
   if r[k]>0 and prev[k]>0:r[k+'_rate']=float(np.log(prev[k]/r[k])/np.log(r['N']/prev['N']))
(OUT/'dmp_refinement_metrics.json').write_text(json.dumps(rows,indent=2))
keys=sorted(set().union(*(r.keys() for r in rows)))
with open(OUT/'dmp_refinement_metrics.csv','w',newline='') as f:
 w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
# plots
fig,axs=plt.subplots(2,2,figsize=(10,7.5))
for method,marker in [('sans DMP','o'),('DMP-FCT r=1e-3','s')]:
 rr=sorted([r for r in rows if r['method']==method],key=lambda z:z['N']);Ns=[r['N'] for r in rr]
 for ax,key,label in zip(axs.ravel(),['overshoot_pct_jump','rho_L1','Y_L1','centerline_mixed_width_m'],['Overshoot densité (% du saut)',r'$L^1(\rho)$ [kg/m³]',r'$L^1(Y_f)$','Largeur mixte centrale [m]']):
  ax.plot(Ns,[r[key] for r in rr],marker=marker,label=method)
  ax.set_xlabel('N par direction');ax.set_ylabel(label);ax.grid(alpha=.3)
for ax in axs.ravel():ax.legend(fontsize=8)
for ax in axs.ravel()[1:]:ax.set_yscale('log')
fig.tight_layout();fig.savefig(OUT/'dmp_refinement_summary.png',dpi=180);plt.close(fig)
# profiles
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for N in [25,50,100]:
 a=np.load(ROOT/'results'/('N100_r0.001_full.npz' if N==100 else f'N{N}_fct_r0.001.npz'));j=N//2
 axs[0].plot(1e3*a['x'],a['rho'][j],label=f'DMP N={N}');axs[1].plot(1e3*a['x'],a['Y'][j],label=f'DMP N={N}')
a=np.load(ROOT/'results/N100_r0.001_full.npz');axs[0].plot(1e3*a['x'],a['rho0'][50],'k--',label='Exact');axs[1].plot(1e3*a['x'],a['Y0'][50],'k--',label='Exact')
for ax in axs:ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_xlabel('x [mm]')
axs[0].set_ylabel(r'$\rho$ [kg/m³]');axs[1].set_ylabel(r'$Y_f$');fig.tight_layout();fig.savefig(OUT/'dmp_centerline_refinement.png',dpi=180);plt.close(fig)
# N100 before/after zoom
old100=np.load('/mnt/data/step9_refine/refinement_results/solution_N100.npz');new100=np.load(ROOT/'results/N100_r0.001_full.npz')
fig,axs=plt.subplots(1,2,figsize=(10,4.2));j=50;x=1e3*new100['x']
axs[0].plot(x,old100['rho'][j],label='sans DMP');axs[0].plot(x,new100['rho'][j],label='DMP-FCT');axs[0].plot(x,new100['rho0'][j],'k--',label='Exact')
axs[1].plot(x,old100['Y'][j],label='sans DMP');axs[1].plot(x,new100['Y'][j],label='DMP-FCT');axs[1].plot(x,new100['Y0'][j],'k--',label='Exact')
for ax in axs:ax.set_xlim(.18,.82);ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_xlabel('x [mm]')
axs[0].set_ylabel(r'$\rho$ [kg/m³]');axs[1].set_ylabel(r'$Y_f$');fig.tight_layout();fig.savefig(OUT/'dmp_N100_before_after.png',dpi=180);plt.close(fig)
print(json.dumps(rows,indent=2))
