import os, sys, time
os.environ.setdefault('M','3'); sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.algo import solve8 as S
g = S.load_growN(); HI = g.HexIdx
def freeze(s):
    ks = list(s.bins); fk = np.array(ks, np.int64); o = np.argsort(fk); fk = fk[o]; L = [np.concatenate(s.bins[ks[i]]) for i in o]
    ln = np.array([len(l) for l in L], np.int64); s.fk = fk; s.fst = np.r_[0, np.cumsum(ln)]; s.fv = np.concatenate(L) if L else np.zeros(0, np.int64); s.frozen = True
def query_csr(s, Y):
    z = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, g.N)))
    if s.n == 0 or not len(Y): return z
    kk = s._key(np.floor(Y / g.QB).astype(np.int64)); pos = np.minimum(np.searchsorted(s.fk, kk), len(s.fk) - 1); pts = np.flatnonzero(s.fk[pos] == kk)
    if not len(pts): return z
    st = s.fst[pos[pts]]; cn = s.fst[pos[pts] + 1] - st; tot = int(cn.sum()); pi = np.repeat(pts, cn); ix = np.repeat(st, cn) + np.arange(tot) - np.repeat(np.cumsum(cn) - cn, cn)
    out = []; CH = 1 << 20
    for a in range(0, tot, CH): out.append(s._test(Y, pi[a:a+CH], s.fv[ix[a:a+CH]]))
    return [np.concatenate(a) for a in zip(*out)]
atlas,_ = S.build_atlas([1.,-1.,0.],0,par=False); G = S.make_graph(g, atlas,'N'); G.solve(verbose=False)
X0 = S.starts(g,200,1); Tr,_=S.t_ref(X0); tm=3*Tr+2
def run(tag):
    for pol in ('base','look'):
        t0=time.perf_counter(); T,sw=G.run_agent(X0,tm,**S.POLICIES[pol]); print(tag,pol,'%.2f s'%(time.perf_counter()-t0),'reach %.2f'%np.isfinite(T).mean(),'T sum',np.nansum(np.where(np.isfinite(T),T,0)).round(4),flush=True)
    t0=time.perf_counter(); T,sw=G.run_agent(X0[:1],tm[:1],**S.POLICIES['base']); print(tag,'1 start base %.0f ms'%((time.perf_counter()-t0)*1e3),flush=True)
run('orig')
Y=X0.copy(); pairs0=[HI.query(G.idx[u],Y) for u in G.us]
for u in G.us: freeze(G.idx[u])
orig=HI.query; HI.query=lambda s,Y: query_csr(s,Y)
pairs1=[HI.query(G.idx[u],Y) for u in G.us]
for a,b in zip(pairs0,pairs1):
    k0=np.lexsort((a[1],a[0])); k1=np.lexsort((b[1],b[0])); print('pairs',len(a[0]),len(b[0]),np.array_equal(a[0][k0],b[0][k1]) and np.array_equal(a[1][k0],b[1][k1]))
run('csr')
G.set_calc('q'); G.solve(verbose=False)
t0=time.perf_counter(); T,sw=G.run_agent(X0,tm,commit=True); print('csr commit %.2f s'%(time.perf_counter()-t0), 'reach %.2f'%np.isfinite(T).mean())
