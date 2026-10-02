import sys,pickle,time; sys.path.insert(0,'.'); sys.path.insert(0,'../v5chain/reports/research'); sys.argv=['x']
import numpy as np
from src.cells7.systems import di
from src.cells7.cell import Cell
from src.cells7.spore_v import SporeV
from v7_faces_di import tstar_box
t0=time.time(); lg=lambda *a: print(*a,round(time.time()-t0),flush=True); S=di(); cells=[Cell(S,k,c,r,t) for k,c,r,t in pickle.load(open('/tmp/claude-1000/di_cells_filled.pkl','rb'))]
V=SporeV(S,cells,lambda Y:(np.abs(Y[:,0])<=.025)&(np.abs(Y[:,1])<=.025),hs=.01,ht=.01,blend=True); V.build(); V.solve(it=8000); lg('iters',V.iters,'конечных',(V.Vn<500).mean().round(3))
rng=np.random.default_rng(0); Qa=rng.uniform(-.375,.375,(200,2)); pa=Qa*4; Tsa=tstar_box(pa[:,0],pa[:,1],.1); va=V.value(Qa); o=(va<500)&(Tsa>.05); r=va[o]/Tsa[o]; lg('blend V*/T*: n',o.sum(),'mean',r.mean().round(3),'med',np.median(r).round(3),'доля <1',(r<1).mean().round(3))
Q=Qa[:60]; ph=Q*4; Ts=tstar_box(ph[:,0],ph[:,1],.1)
for dt in (.06,.02):
    T,sw=V.rollout(Q,dt=dt); ok=np.isfinite(T)&(Ts>.05); r=T[ok]/Ts[ok]; lg('dt',dt,'дошли',np.isfinite(T).mean().round(3),'T/T* mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'sw',np.median(sw))
