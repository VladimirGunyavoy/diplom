"""d5 Г5.1 (описательно): общее почти-оптимальное u у вершин симплекса. Запуск как tvg_lp: (из ~/spore_v5/d4/spores_2/w39n, те же env, что fullv.sh, но TAUS/TAU по среде) python3 g51.py di4 100000 8 0
Граф строится кодом tvg_lp.AgentLP без изменений (vit_dl обёрнут только для перехвата J, LAM, OKd, GL). Печатает ПАСПОРТ и таблицу по δ ∈ {0,.01,.03,.1}."""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/spores_2/w39n'))
import numpy as np
import tvg_lp as TL
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
T = TL.T; CAP = {}
_vd = TL.vit_dl
def vd(J, LAM, OKd, GL, ING, nP, TAUV, *a, **k):
    CAP.update(J=J, LAM=LAM, OKd=OKd, GL=GL, nP=nP); return _vd(J, LAM, OKd, GL, ING, nP, TAUV, *a, **k)
TL.vit_dl = vd
def main():
    ts = T.TS; S = T.mkS(ts); d = T.d
    rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(T.env.period > 0, -T.env.period / 2, T.LO); hi_ = np.where(T.env.period > 0, T.env.period / 2, T.HI)
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
    back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = TL.AgentLP(S, back); W = ag.W; n = len(ag.P); L2 = len(T.TAUV); L = int(T.SIDX.max()) + 1; TAUV = T.TAUV; SIDX = T.SIDX
    print('ПАСПОРТ g51: env=%s NB=%d n=%d L2=%d L=%d TAUV=%s KLP=%d VCAP=%s seed=%d' % (T.ENV, T.NB, n, L2, L, np.unique(TAUV), TL.KLP, os.environ.get('VCAP'), ts), flush=True)
    J = CAP['J'] % CAP['nP']; LAM = CAP['LAM']; OKd = CAP['OKd']; GL = CAP['GL']; K = J.shape[1]
    fin = W < 1e2; print('узлов с конечной W: %.1f%%' % (100 * fin.mean()))
    Jn = J.reshape(n, L2, K); Ln = LAM.reshape(n, L2, K)
    c = np.where(OKd, np.where(GL > 0, GL, TAUV[None] + (Ln * W[Jn]).sum(-1)), np.inf); best = c.min(1); ok_n = fin & np.isfinite(best) & (best < 1e2)
    # симплексы: рёбра (i,a) с конечными вершинами, не в цель
    simp = (OKd & (GL <= 0)); ii, aa = np.nonzero(simp); V = Jn[ii, aa]; LM = Ln[ii, aa]; allfin = fin[V].all(1); ii, aa, V, LM = ii[allfin], aa[allfin], V[allfin], LM[allfin]
    print('симплексов (рёбер, все вершины конечны): %d' % len(ii), flush=True)
    out = dict(env=T.ENV, NB=T.NB, n=n, nsimp=int(len(ii)), rows=[])
    for dl in (0, .01, .03, .10):
        M = np.zeros((n, L), bool)
        for a in range(L2): M[:, SIDX[a]] |= (c[:, a] <= best + dl * TAUV[a] + 1e-12)
        M &= ok_n[:, None]
        common = M[V[:, 0]].copy()
        for k in range(1, K): common &= M[V[:, k]]
        fs = common.any(1).mean(); nopt = M.sum(1)[ok_n].mean()
        # жадное покрытие областями с общим u
        order = np.argsort(-M.sum(0)); assigned = np.zeros(n, bool); nreg = 0
        for u in order:
            Su = M[:, u] & ~assigned
            if not Su.any(): continue
            sel = np.nonzero(common[:, u] & Su[V].all(1))[0]
            idx = np.nonzero(Su)[0]; mp = -np.ones(n, np.int64); mp[idx] = np.arange(len(idx))
            if len(sel): r = np.repeat(mp[V[sel, 0]], K - 1); cc = mp[V[sel, 1:]].ravel(); g = coo_matrix((np.ones(len(r)), (r, cc)), shape=(len(idx), len(idx))); ncomp = connected_components(g, directed=False)[0]
            else: ncomp = len(idx)
            nreg += ncomp; assigned |= Su
        nass = int(assigned.sum()); row = dict(delta=dl, frac_simplex_common_u=float(fs), mean_nearopt_u_per_node=float(nopt), nodes_covered=nass, regions=int(nreg), compression=float(nass / max(nreg, 1)))
        out['rows'].append(row); print('δ=%.2f: симплексов с общим u %.1f%%, почти-опт. u на узел %.2f, узлов покрыто %d, областей %d, сжатие ×%.2f' % (dl, 100 * fs, nopt, nass, nreg, row['compression']), flush=True)
    os.makedirs(os.path.expanduser('~/spore_v5/d4/out'), exist_ok=True); json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/g51_%s_%d_%d.json' % (T.ENV, T.NB, ts)), 'w'))
if __name__ == '__main__': main()
