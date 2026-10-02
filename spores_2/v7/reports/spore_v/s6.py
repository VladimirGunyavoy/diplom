import sys,pickle,os,time; sys.path.insert(0,'.'); sys.path.insert(0,'../v5chain/reports/research'); sys.argv=['x']
import numpy as np
from src.cells7.systems import di
from src.cells7.cell import Cell
from src.cells7.spore_v import SporeV
from v7_faces_di import tstar_box
S=di(); t0=time.time(); lg=lambda *a: print(*a,round(time.time()-t0),flush=True)
cells=[Cell(S,k,c,r,t) for k,c,r,t in pickle.load(open('/tmp/claude-1000/di_cells_m40.pkl','rb'))]
goal=lambda Y:(np.abs(Y[:,0])<=.025)&(np.abs(Y[:,1])<=.025)
rng=np.random.default_rng(5)
for rnd in range(3):                                           # заполнение дыр: точки вне ядра+гало ВСЕХ клеток → клетка слоя 0 или 1 (по очереди), точки внутри новой клетки выбывают
    Vt=SporeV(S,cells,goal,hs=.05,ht=.05); Qa=rng.uniform(-1,1,(30000,2)); qi=Vt._pairs(Qa)[0]; unc=np.nonzero(np.bincount(qi,minlength=len(Qa))==0)[0]; lg('раунд',rnd,'клеток',len(cells),'дыр',len(unc),'из',len(Qa))
    if len(unc)<30: break
    add=0
    while len(unc):
        p=Qa[unc[0]]; c=Cell(S,add%2,p); cells.append(c); add+=1; ins=c.locate(Qa[unc],1.1)[2]; unc=unc[~ins]; unc=unc[1:] if False else unc
    lg('добавлено',add)
pickle.dump([(c.k,c.c,c.r,c.tau) for c in cells],open('/tmp/claude-1000/di_cells_filled.pkl','wb')); lg('клеток итого',len(cells))
V=SporeV(S,cells,goal,hs=.01,ht=.01); lg('nodes',V.N); V.build(); V.solve(it=8000); lg('iters',V.iters,'конечных',(V.Vn<500).mean().round(3)); np.save('/tmp/claude-1000/di_V_filled.npy',V.Vn)
rng=np.random.default_rng(0); Q=rng.uniform(-.375,.375,(200,2)); ph=Q*4; v0=V.value(Q); Ts0=tstar_box(ph[:,0],ph[:,1],.1); o=(v0<500)&(Ts0>.05)
lg('V*/T*: n',o.sum(),'mean',(v0[o]/Ts0[o]).mean().round(3),'med',np.median(v0[o]/Ts0[o]).round(3),'max',(v0[o]/Ts0[o]).max().round(2))
Q=Q[:60]; ph=Q*4
for eps in (0.0,0.003):
    T,sw=V.rollout(Q,dt=.06,eps=eps); Ts=tstar_box(ph[:,0],ph[:,1],.1); ok=np.isfinite(T)&(Ts>.05); r=T[ok]/Ts[ok]
    lg('eps',eps,'дошли',np.isfinite(T).mean().round(3),'T/T* mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'switch med',np.median(sw))
