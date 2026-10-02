import sys,pickle,time; sys.path.insert(0,'.'); sys.path.insert(0,'../v5chain/reports/research'); sys.argv=['x']
import numpy as np
from src.cells7.systems import di
from src.cells7.cell import Cell
from src.cells7.spore_v import SporeV
from v7_faces_di import tstar_box
t0=time.time(); S=di(); cells=[Cell(S,k,c,r,t) for k,c,r,t in pickle.load(open('/tmp/claude-1000/di_cells_filled.pkl','rb'))]
V=SporeV(S,cells,lambda Y:(np.abs(Y[:,0])<=.025)&(np.abs(Y[:,1])<=.025),hs=.01,ht=.01); V.Vn=np.load('/tmp/claude-1000/di_V_filled.npy'); print('init',round(time.time()-t0),'median Tn',np.median(V.dtc),flush=True)
rng=np.random.default_rng(0); Q=rng.uniform(-.375,.375,(200,2))[:60]; ph=Q*4; Ts=tstar_box(ph[:,0],ph[:,1],.1)
for dt in (.06,.02,.0075):
    T,sw=V.rollout(Q,dt=dt); ok=np.isfinite(T)&(Ts>.05); r=T[ok]/Ts[ok]; print('dt',dt,'sub4 дошли',np.isfinite(T).mean().round(3),'T/T* mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'sw',np.median(sw),round(time.time()-t0),flush=True)
# где V*<T*: по расстоянию до цели
Qa=rng.uniform(-.375,.375,(3000,2)); pa=Qa*4; Tsa=tstar_box(pa[:,0],pa[:,1],.1); va=V.value(Qa); lo=(va<Tsa-1e-3)&(va<500)
print('V*<T*: доля',lo.mean().round(3),'| медиана T* у таких',np.median(Tsa[lo]).round(3) if lo.any() else '-','| у остальных',np.median(Tsa[~lo]).round(3),'| V/T* у таких med',np.median(va[lo]/np.maximum(Tsa[lo],1e-9)).round(3))
