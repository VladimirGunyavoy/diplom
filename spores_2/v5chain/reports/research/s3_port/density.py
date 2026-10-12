import os, sys, time
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.algo import solve8 as S
g = S.load_growN()
t0=time.perf_counter(); atlas,_ = S.build_atlas([1.,-1.,0.],0,par=False); tb=time.perf_counter()-t0
G = S.make_graph(g, atlas,'N'); t0=time.perf_counter(); G.solve(verbose=False); ts=time.perf_counter()-t0
X0 = S.starts(g,200,1); Tr,_=S.t_ref(X0); tm=3*Tr+2; mx=S.path_extent(X0,G.rho); inw=mx<=float(g.XLV[0])
t0=time.perf_counter(); Vs=G.Vstar(X0)[0]; tq=(time.perf_counter()-t0)*1e3
m=np.isfinite(Vs)&inw; vr=Vs[m]/Tr[m]
print('RMAX',os.environ.get('RMAX'),'M',g.M,'cells',[len(atlas[u]) for u in G.us],'nodes',G.n,'build %.1f s solve %.2f s (проходов %d) Vstar200 %.0f ms'%(tb,ts,G.passes,tq),'V*/Tref мед %.3f p90 %.3f'%(np.median(vr),np.percentile(vr,90)),flush=True)
def ag(tag,**kw):
    t0=time.perf_counter(); T,sw=G.run_agent(X0,tm,**kw); dt=time.perf_counter()-t0; ok=np.isfinite(T); r=T[ok&inw]/Tr[ok&inw]
    print(tag,'в поле дошли %.3f'%ok[inw].mean(),'T/Tref мед %.3f p90 %.3f'%(np.median(r),np.percentile(r,90)),'sw %.1f'%sw[inw].mean(),'%.1f s'%dt,flush=True)
ag('look',look=True)
G.set_calc('q'); t0=time.perf_counter(); G.solve(verbose=False); print('solveQ %.2f s'%(time.perf_counter()-t0))
ag('commit',commit=True)
