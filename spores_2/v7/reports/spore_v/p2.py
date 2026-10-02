import sys,time,pickle,os; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('tests/run_faces_pend.py').read().split("S = pend_top()")[0])
from src.cells7.cell import Cell
from src.cells7.spore_v import SporeV, fill_gaps
S=pend_top(); t0=time.time(); lg=lambda *a: print(*a,round(time.time()-t0),flush=True)
cells=[Cell(S,k,c,r,t) for k,c,r,t in pickle.load(open('/tmp/claude-1000/pend_cells_m40.pkl','rb'))]; lg('клеток',len(cells))
gx,gv=.02,.02        # цель (физ.: |φ|≤.063 рад, |ω|≤.08)
goal=lambda Y:(np.abs(S.wrap(Y)[:,0])<=gx)&(np.abs(Y[:,1])<=gv)
cells=fill_gaps(S,cells,goal,Cell,lo=-1,hi=1,log=lg); lg('после заполнения',len(cells))
pickle.dump([(c.k,c.c,c.r,c.tau) for c in cells],open('/tmp/claude-1000/pend_cells_filled.pkl','wb'))
V=SporeV(S,cells,goal,hs=.02,ht=.02); lg('узлов',V.N); V.build(); V.solve(it=8000); lg('iters',V.iters,'конечных',(V.Vn<500).mean().round(3)); np.save('/tmp/claude-1000/pend_V.npy',V.Vn)
rng=np.random.default_rng(0); Q=np.stack([rng.uniform(-1,1,60),rng.uniform(-.8,.8,60)],1)
from atlas6_ref import ref_T   # эталон v6 PendAtlas
Ta=ref_T(Q)
for eps in (0.0,0.003):
    T,sw=V.rollout(Q,dt=.06,eps=eps); ok=np.isfinite(T)&(Ta<500)&(Ta>1); r=T[ok]/Ta[ok]
    lg('eps',eps,'дошли',np.isfinite(T).mean().round(3),'T/Ta mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'switch med',np.median(sw))
