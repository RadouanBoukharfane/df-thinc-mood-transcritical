import sys,json,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,'/mnt/data/step9_refine')
import dodecane_drop_2d as d
OUT=Path('/mnt/data/step9_refine/refinement_results');OUT.mkdir(exist_ok=True)
# materialize N100 from chunk state
st=np.load('/mnt/data/step9_refine/chunk_states/N100.npz'); s=d.setup_figure13(100)
rho=st['rho'];Z=st['Z'];Y=Z/np.maximum(rho,1.0e-14)
np.savez_compressed(OUT/'solution_N100.npz',rho=rho,Y=Y,rho_exact=s['rho'],Y_exact=s['Yfuel'],x=s['x'],y=s['y'])

def groups_width(x,y,lo=.1,hi=.9):
 m=(y>lo)&(y<hi)
 if not m.any():return 0.
 ids=np.where(m)[0]; groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1);dx=x[1]-x[0]
 return float(sum(len(g)*dx for g in groups))
rows=[]
for N in (25,50,100):
 a=np.load(OUT/f'solution_N{N}.npz');rho=a['rho'];Y=a['Y'];r0=a['rho_exact'];Y0=a['Y_exact'];x=a['x'];y=a['y'];dx=x[1]-x[0];dy=y[1]-y[0]
 X,YY=np.meshgrid(x,y); exmin=float(r0.min());exmax=float(r0.max());jump=exmax-exmin
 over=max(0,float(rho.max()-exmax));under=max(0,float(exmin-rho.min()))
 mixed=(Y>.1)&(Y<.9); inside=Y>=.5; inside0=Y0>=.5
 w=Y;sw=w.sum();xc=float((w*X).sum()/sw);yc=float((w*YY).sum()/sw);vx=float((w*(X-xc)**2).sum()/sw);vy=float((w*(YY-yc)**2).sum()/sw)
 rows.append(dict(N=N,dx_m=float(dx),rho_L1=float(np.mean(abs(rho-r0))),Y_L1=float(np.mean(abs(Y-Y0))),rho_max=float(rho.max()),rho_exact_max=exmax,
  overshoot_abs=over,overshoot_pct_jump=100*over/jump,overshoot_pct_plateau=100*over/exmax,undershoot_abs=under,
  mixed_cell_fraction=float(mixed.mean()),mixed_area_m2=float(mixed.sum()*dx*dy),centerline_mixed_width_m=groups_width(x,Y[N//2]),
  threshold_area_rel_error=float(abs(inside.sum()-inside0.sum())/inside0.sum()),anisotropy=float(abs(vx-vy)/max(vx,vy)),mass_proxy_rel=float(abs(rho.sum()-r0.sum())/r0.sum()),fuel_mass_proxy_rel=float(abs((rho*Y).sum()-(r0*Y0).sum())/(r0*Y0).sum())))
# rates from successive doubling
for i,row in enumerate(rows):
 if i==0:
  row['rho_L1_rate']=None;row['Y_L1_rate']=None;row['mixed_width_rate']=None;row['overshoot_rate']=None
 else:
  prev=rows[i-1]
  for key,out in [('rho_L1','rho_L1_rate'),('Y_L1','Y_L1_rate'),('centerline_mixed_width_m','mixed_width_rate'),('overshoot_abs','overshoot_rate')]:
   row[out]=float(np.log(prev[key]/row[key])/np.log(row['N']/prev['N'])) if row[key]>0 and prev[key]>0 else None
(OUT/'refinement_metrics.json').write_text(json.dumps(rows,indent=2))
keys=list(rows[0].keys())
with open(OUT/'refinement_metrics.csv','w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
# combined plot
fig,axs=plt.subplots(2,2,figsize=(10,7.5))
axs=axs.ravel();
for ax,key,label in zip(axs,['overshoot_pct_jump','rho_L1','Y_L1','centerline_mixed_width_m'],['Overshoot (% du saut)',r'$L^1(\rho)$ [kg/m³]',r'$L^1(Y_f)$','Largeur mixte centrale [m]']):
 ax.plot([r['N'] for r in rows],[r[key] for r in rows],'-o');ax.set_xlabel('N par direction');ax.set_ylabel(label);ax.grid(alpha=.3)
 if key!='overshoot_pct_jump':ax.set_yscale('log')
fig.tight_layout();fig.savefig(OUT/'refinement_summary.png',dpi=180);plt.close(fig)
# profiles
fig,axs=plt.subplots(1,2,figsize=(10,4.2))
for r in rows:
 a=np.load(OUT/f"solution_N{r['N']}.npz");j=r['N']//2
 axs[0].plot(1e3*a['x'],a['rho'][j],label=f"N={r['N']}");axs[1].plot(1e3*a['x'],a['Y'][j],label=f"N={r['N']}")
a=np.load(OUT/'solution_N100.npz');axs[0].plot(1e3*a['x'],a['rho_exact'][50],'k--',label='Exact');axs[1].plot(1e3*a['x'],a['Y_exact'][50],'k--',label='Exact')
for ax in axs:ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_xlabel('x [mm]')
axs[0].set_ylabel(r'$\rho$ [kg/m³]');axs[1].set_ylabel(r'$Y_f$');fig.tight_layout();fig.savefig(OUT/'centerline_refinement.png',dpi=180);plt.close(fig)
print(json.dumps(rows,indent=2))
