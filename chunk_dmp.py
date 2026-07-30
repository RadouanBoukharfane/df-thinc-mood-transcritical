import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data/dmp_study')
import dodecane_drop_2d as d
N=int(sys.argv[1]); chunk=int(sys.argv[2]); rel=float(sys.argv[3]); tag=f'{rel:g}'
root=Path('/mnt/data/dmp_study/chunks');root.mkdir(exist_ok=True);f=root/f'N{N}_r{tag}_half.npz'
if not f.exists():
 s=d.setup_figure13(N);rho=s['rho'][:N//2].copy();Z=rho*s['Yfuel'][:N//2]
 c=d.sound_speed_from_T_rho_Y(s['temperature'],s['rho'],s['Yfuel']);dt_nom=.8/(np.max((abs(s['u0'])+c)/s['dx'])+np.max(c/s['dy']))
 tf=s['L']/s['u0'];n=int(np.ceil(tf/dt_nom));dt=tf/n;k=0;maxth=maxdis=0.
 np.savez(f,rho=rho,Z=Z,k=k,n=n,dt=dt,maxth=maxth,maxdis=maxdis)
a=np.load(f);rho=a['rho'];Z=a['Z'];k=int(a['k']);n=int(a['n']);dt=float(a['dt']);maxth=float(a['maxth']);maxdis=float(a['maxdis']);s=d.setup_figure13(N)
end=min(n,k+chunk)
for kk in range(k,end):
 rho,Z,th,di=d._rk3(rho,Z,dt,s['u0'],s['dx'],'final',.2,1.6,False,dmp_rel=rel,dmp_abs=1e-10,dmp_radius=1);maxth=max(maxth,th);maxdis=max(maxdis,di)
np.savez(f,rho=rho,Z=Z,k=end,n=n,dt=dt,maxth=maxth,maxdis=maxdis)
print(json.dumps({'N':N,'rel':rel,'k':end,'n':n,'done':end==n}),flush=True)
