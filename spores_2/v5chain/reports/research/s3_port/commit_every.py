import os, sys, time
os.environ.setdefault('M','3'); sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.algo import solve8 as S
g = S.load_growN()
atlas,_ = S.build_atlas([1.,-1.,0.],0,par=False); G = S.make_graph(g, atlas,'N'); G.set_calc('q'); G.solve(verbose=False)
X0 = S.starts(g,200,1); Tr,_=S.t_ref(X0); tm=3*Tr+2; mx=S.path_extent(X0,G.rho); inw=mx<=float(g.XLV[0])
for ev in (1,3,6,12):
    t0=time.perf_counter(); T,sw=G.run_commit(X0,tm,every=ev); dt=time.perf_counter()-t0; ok=np.isfinite(T)&inw; r=T[ok]/Tr[ok]
    print('commit every',ev,'%.2f s'%dt,'в поле дошли %.3f'%(np.isfinite(T)[inw].mean()),'T/Tref мед %.3f p90 %.3f max %.2f'%(np.median(r),np.percentile(r,90),r.max()),'sw %.1f'%sw[inw].mean(),flush=True)
