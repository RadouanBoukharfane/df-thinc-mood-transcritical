import json,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path('/mnt/data/dmp_study');OUT=ROOT/'dmp_final';OUT.mkdir(exist_ok=True)
old=json.loads(Path('/mnt/data/step9_refine/refinement_results/refinement_metrics.json').read_text()); old={r['N']:r for r in old}

def measure(path,N,label):
 a=np.load(path);rho,Y,r0,Y0,x,y=[a[k] for k in ['rho','Y','rho0','Y0','x','y']];dx=float(x[1]-x[0]);dy=float(y[1]-y[0]);X,Yg=np.meshgrid(x,y)
 mx,mn=float(r0.max()),float(r0.min());jump=mx-mn; over=max(0.,float(rho.max()-mx));under=max(0.,float(mn-rho.min())); mixed=(Y>.1)&(Y<.9);j=N//2
 inside=Y>=.5;inside0=Y0>=.5;w=Y;sw=w.sum();xc=(w*X).sum()/sw;yc=(w*Yg).sum()/sw;vx=(w*(X-xc)**2).sum()/sw;vy=(w*(Yg-yc)**2).sum()/sw
 return dict(N=N,method=label,rho_L1=float(np.mean(abs(rho-r0))),Y_L1=float(np.mean(abs(Y-Y0))),rho_max=float(rho.max()),rho_min=float(rho.min()),overshoot_abs=over,overshoot_pct_jump=100*over/jump,undershoot_abs=under,mixed_cell_fraction=float(mixed.mean()),centerline_mixed_width_m=float(mixed[j].sum()*dx),threshold_area_rel_error=float(abs(inside.sum()-inside0.sum())/inside0.sum()),anisotropy=float(abs(vx-vy)/max(vx,vy)),mass_rel=float(abs(rho.sum()-r0.sum())/r0.sum()),fuel_mass_rel=float(abs((rho*Y).sum()-(r0*Y0).sum())/(r0*Y0).sum()))
rows=[]
for N in [25,50,100]:
 o=old[N].copy();o['method']='sans DMP';rows.append(o)
 strict=ROOT/'results'/('N100_r0_full.npz' if N==100 else f'N{N}_fct_r0.npz');rows.append(measure(strict,N,'DMP strict FCT'))
 relaxed=ROOT/'results'/('N100_r0.001_full.npz' if N==100 else f'N{N}_fct_r0.001.npz');rows.append(measure(relaxed,N,'DMP relaxé 1e-3'))
for method in sorted(set(r['method'] for r in rows)):
 rr=sorted([r for r in rows if r['method']==method],key=lambda z:z['N'])
 for i,r in enumerate(rr):
  if i==0:continue
  p=rr[i-1]
  for k in ['rho_L1','Y_L1','centerline_mixed_width_m','overshoot_abs']:
   if r.get(k,0)>0 and p.get(k,0)>0:r[k+'_rate']=float(np.log(p[k]/r[k])/np.log(r['N']/p['N']))
(OUT/'dmp_final_metrics.json').write_text(json.dumps(rows,indent=2));keys=sorted(set().union(*(r.keys() for r in rows)))
with open(OUT/'dmp_final_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
# summary plot
fig,axs=plt.subplots(2,2,figsize=(10,7.5));styles={'sans DMP':'o-','DMP strict FCT':'s-','DMP relaxé 1e-3':'^-'}
for method,style in styles.items():
 rr=sorted([r for r in rows if r['method']==method],key=lambda z:z['N']);Ns=[r['N'] for r in rr]
 for ax,key,label in zip(axs.ravel(),['overshoot_pct_jump','rho_L1','Y_L1','centerline_mixed_width_m'],['Overshoot densité (% du saut)',r'$L^1(\rho)$ [kg/m³]',r'$L^1(Y_f)$','Largeur mixte centrale [m]']):ax.plot(Ns,[r[key] for r in rr],style,label=method);ax.set_xlabel('N par direction');ax.set_ylabel(label);ax.grid(alpha=.3)
for ax in axs.ravel():ax.legend(fontsize=8)
for ax in axs.ravel()[1:]:ax.set_yscale('log')
fig.tight_layout();fig.savefig(OUT/'dmp_final_summary.png',dpi=180);plt.close(fig)
# N100 profiles
base=np.load('/mnt/data/step9_refine/refinement_results/solution_N100.npz');strict=np.load(ROOT/'results/N100_r0_full.npz');rel=np.load(ROOT/'results/N100_r0.001_full.npz');j=50;x=1e3*strict['x']
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for a,l in [(base,'sans DMP'),(rel,'DMP relaxé'),(strict,'DMP strict')]:axs[0].plot(x,a['rho'][j],label=l);axs[1].plot(x,a['Y'][j],label=l)
axs[0].plot(x,strict['rho0'][j],'k--',label='Exact');axs[1].plot(x,strict['Y0'][j],'k--',label='Exact')
for ax in axs:ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_xlabel('x [mm]')
axs[0].set_ylabel(r'$\rho$ [kg/m³]');axs[1].set_ylabel(r'$Y_f$');fig.tight_layout();fig.savefig(OUT/'dmp_N100_profiles.png',dpi=180);plt.close(fig)
print(json.dumps(rows,indent=2))
