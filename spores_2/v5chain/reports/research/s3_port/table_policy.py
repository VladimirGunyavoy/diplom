import os, sys, time
os.environ.setdefault('M','3'); sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.algo import solve8 as S
g = S.load_growN(); BIG=1e3
atlas,_ = S.build_atlas([1.,-1.,0.],0,par=False); G = S.make_graph(g, atlas,'N'); G.set_calc('q'); G.solve(verbose=False)
X0 = S.starts(g,300,1); Tr,_=S.t_ref(X0); tm=3*Tr+2; mx=S.path_extent(X0,G.rho); inw=mx<=float(g.XLV[0])
def rep(tag,f):
    t0=time.perf_counter(); T,sw=f(); dt=time.perf_counter()-t0; ok=np.isfinite(T); r=T[ok&inw]/Tr[ok&inw]
    print(tag,'в поле дошли %.3f'%ok[inw].mean(),'T/Tref мед %.3f p90 %.3f max %.2f'%(np.median(r),np.percentile(r,90),r.max()),'sw %.1f'%sw[inw].mean(),'%.2f s'%dt,flush=True)
EV=3
rep('exact commit every3',lambda:G.run_commit(X0,tm,every=EV))
orig=G.Vlayers
for h in (.02,.01,.005):
    xs=np.arange(-3.5,3.5+h/2,h); vs=np.arange(-3.0,3.0+h/2,h); XX,VV=np.meshgrid(xs,vs,indexing='ij'); P=np.c_[XX.ravel(),VV.ravel()]
    t0=time.perf_counter(); Vl=orig(P); tt=time.perf_counter()-t0; L=Vl.shape[1]
    tab=np.where(np.isfinite(Vl),Vl,BIG).reshape(len(xs),len(vs),L)
    def VL(X,tab=tab,xs=xs,vs=vs,h=h):
        fx=(X[:,0]-xs[0])/h; fv=(X[:,1]-vs[0])/h; ins=(fx>=0)&(fx<len(xs)-1)&(fv>=0)&(fv<len(vs)-1); fx=np.clip(fx,0,len(xs)-1.001); fv=np.clip(fv,0,len(vs)-1.001)
        i=fx.astype(int); j=fv.astype(int); a=(fx-i)[:,None]; b=(fv-j)[:,None]
        o=(1-a)*(1-b)*tab[i,j]+a*(1-b)*tab[i+1,j]+(1-a)*b*tab[i,j+1]+a*b*tab[i+1,j+1]
        o=np.where((o<BIG/2)&ins[:,None],o,np.inf); inb=np.linalg.norm(X,axis=1)<=G.rho; o[inb]=0.; return o
    G.Vlayers=VL; print('grid h',h,tab.shape[:2],'построение таблицы %.1f s'%tt,'память %.0f МБ'%(tab.nbytes/1e6),flush=True)
    rep('  таблица h=%g commit every3'%h,lambda:G.run_commit(X0,tm,every=EV))
    rep('  таблица h=%g commit every1'%h,lambda:G.run_commit(X0,tm,every=1))
