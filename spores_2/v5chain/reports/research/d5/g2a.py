"""d5 Г2а (описательно, ДИ 4D 100k): признаки ошибки V̂ в узле — (а) невязка локальной линейной подгонки W по K=20 ближайшим соседям (масштаб S), (б) невязка Беллмана |W − min_a c(i,a)|.
Истина: верхние 20% узлов по |W − V*|/V* (V* точный, max по осям, di4_ref_exact.tstar), только узлы с W<VCAP. AUC (через ранги) и при пометке верхних 30% по признаку: доля помеченных от истины (recall) и точность. |W_f−W_b| НЕ мерено (нужна обратная раздача c12/fb — отдельно).
Запуск как g51 (di4 100000 8 0), FBPROC=8."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import g51
from g51 import TL, CAP, T
from multiprocessing import Pool
import di4_ref_exact as R
def _vs(Q): return np.array([R.tstar(q) for q in Q])
def auc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score); npos = label.sum(); nneg = len(label) - npos; return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))
def main():
    ts = T.TS; S = T.mkS(ts); d = T.d; VC = float(os.environ.get('VCAP', 40))
    rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(T.env.period > 0, -T.env.period / 2, T.LO); hi_ = np.where(T.env.period > 0, T.env.period / 2, T.HI)
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]; back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = TL.AgentLP(S, back); W = ag.W; P = ag.P; n = len(P); L2 = len(T.TAUV); K = d + 1
    print('ПАСПОРТ g2a: env=%s NB=%d n=%d VCAP=%g' % (T.ENV, T.NB, n, VC), flush=True)
    with Pool(int(os.environ.get('FBPROC', 8))) as p: Vs = np.concatenate(p.map(_vs, np.array_split(P, 64))).max(1)
    J = CAP['J'] % CAP['nP']; OKd = CAP['OKd']; GL = CAP['GL']; Jn = J.reshape(n, L2, K); Ln = CAP['LAM'].reshape(n, L2, K)
    c = np.where(OKd, np.where(GL > 0, GL, T.TAUV[None] + (Ln * W[Jn]).sum(-1)), np.inf); bell = np.abs(W - c.min(1))
    X = T._scaled(S, T._images(S, P)) if False else ag.tree.data[:n]; dd, nb = ag.tree.query(X, k=21, workers=8); nb = nb % n
    A = np.concatenate([X[nb[:, 1:]] - X[:, None], np.ones((n, 20, 1))], 2); y = W[nb[:, 1:]]
    coef = np.linalg.lstsq(A.reshape(-1, d + 1)[:0], y.reshape(-1)[:0], rcond=None) if False else None
    res = np.empty(n)
    for a in range(0, n, 20000):
        Aa = A[a:a + 20000]; ya = y[a:a + 20000]; G = np.einsum('nki,nkj->nij', Aa, Aa) + 1e-9 * np.eye(d + 1); bq = np.einsum('nki,nk->ni', Aa, ya); th = np.linalg.solve(G, bq[..., None])[..., 0]
        res[a:a + 20000] = np.sqrt(((np.einsum('nki,ni->nk', Aa, th) - ya) ** 2).mean(1))
    ok = (W < VC - 1e-9) & np.isfinite(Vs) & (Vs > 1e-3); err = np.abs(W - Vs) / Vs; thr = np.percentile(err[ok], 80); lab = (err >= thr) & ok
    out = dict(env=T.ENV, NB=T.NB, n_ok=int(ok.sum()), err_thr80=float(thr), rows=[])
    for name, sc in (('лин.подгонка', res), ('Беллман', bell)):
        s = sc[ok]; l = lab[ok]; a = auc(s, l); fl = s >= np.percentile(s, 70); rec = float((fl & l).sum() / l.sum()); pr = float((fl & l).sum() / fl.sum())
        out['rows'].append(dict(feature=name, auc=a, flagged30_recall=rec, flagged30_precision=pr)); print('%s: AUC %.3f; пометка верхних 30%%: recall %.2f, точность %.2f (база %.2f)' % (name, a, rec, pr, l.mean()), flush=True)
    json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/g2a_%s_%d_%d.json' % (T.ENV, T.NB, ts)), 'w'))
if __name__ == '__main__': main()
