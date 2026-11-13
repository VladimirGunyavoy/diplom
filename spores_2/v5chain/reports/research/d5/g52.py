"""d5 Г5.2 (описательно): «поток не гнёт симплекс». Симплекс = опорные вершины V (K=d+1) ребра (i,a) графа tvg_lp (как в g51). Для каждого (u,τ)=b:
доля симплексов, где |Φ_b(mean V) − mean Φ_b(V)| < 5% диаметра симплекса (в масштабе S.scale, углы — разность в (−per/2,per/2]). Φ_b(вершин) = T.edges по узлам, Φ_b(центра) = T.edges по центрам.
Выборка NS симплексов (по умолчанию 20000, rng 5). Порог Г4.1: доля ≥ 90% (по каждому b и усреднённо). Запуск как g51."""
import os, sys, json
import numpy as np
import g51
from g51 import TL, CAP, T
def main():
    ts = T.TS; S = T.mkS(ts); d = T.d; NSM = int(os.environ.get('NSIMP', 20000))
    rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(T.env.period > 0, -T.env.period / 2, T.LO); hi_ = np.where(T.env.period > 0, T.env.period / 2, T.HI)
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]; back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = TL.AgentLP(S, back); W = ag.W; n = len(ag.P); L2 = len(T.TAUV); K = d + 1; P = ag.P
    print('ПАСПОРТ g52: env=%s NB=%d n=%d L2=%d NSIMP=%d порог 5%% диаметра' % (T.ENV, T.NB, n, L2, NSM), flush=True)
    J = CAP['J'] % CAP['nP']; OKd = CAP['OKd']; GL = CAP['GL']; Jn = J.reshape(n, L2, K); fin = W < 1e2
    ii, aa = np.nonzero(OKd & (GL <= 0)); V = Jn[ii, aa]; g = fin[V].all(1); V = np.unique(np.sort(V[g], 1), axis=0)
    rng = np.random.default_rng(5); V = V[rng.choice(len(V), min(NSM, len(V)), replace=False)]; per = np.asarray(T.env.period, float); sc = np.asarray(S.scale, float)
    def dif(A, B): D = A - B; return np.where(per > 0, (D + per / 2) % np.where(per > 0, per, 1.) - per / 2, D) / sc
    # вершины симплекса относительно первой (углы разворачиваем), центр = v0 + mean(dif(v_j, v0))
    X0 = P[V[:, 0]]; rel = np.stack([dif(P[V[:, k]], X0) * sc for k in range(K)], 1); C = X0 + rel.mean(1)
    diam = max_pair = np.max([np.linalg.norm((rel[:, i] - rel[:, j]) / sc, axis=1) for i in range(K) for j in range(i + 1, K)], axis=0)
    Ec, OKc, _ = T.edges(C, S); Ev, OKv, _ = T.edges(P, S)
    res = []; allf = []
    for b in range(L2):
        Ei = Ev[V, b]; ref = Ei[:, 0:1]; relE = np.stack([dif(Ei[:, k], Ei[:, 0]) * sc for k in range(K)], 1) if False else np.stack([dif(Ei[:, k], Ei[:, 0]) * sc for k in range(K)], 1)
        cen = Ei[:, 0] + relE.mean(1); dev = np.linalg.norm(dif(Ec[:, b], cen), axis=1); ok = OKc[:, b] & OKv[V, b].all(1)
        fr = float((dev[ok] < .05 * diam[ok]).mean()) if ok.any() else float('nan'); res.append(dict(b=int(b), u=int(T.SIDX[b]), tau=float(T.TAUV[b]), frac=fr, n_ok=int(ok.sum()), med_dev_over_diam=float(np.median(dev[ok] / diam[ok])) if ok.any() else None)); allf.append((dev < .05 * diam) | ~ok)
        print('b=%d u=%d τ=%.3f: доля %.1f%% (допустимых %d), мед |откл|/диам %.4f' % (b, T.SIDX[b], T.TAUV[b], 100 * fr, ok.sum(), res[-1]['med_dev_over_diam'] or -1), flush=True)
    allok = np.stack(allf, 1).all(1).mean(); print('по всем (u,τ) одновременно: %.1f%%; среднее по b: %.1f%%' % (100 * allok, 100 * np.nanmean([r['frac'] for r in res])), flush=True)
    json.dump(dict(env=T.ENV, NB=T.NB, rows=res, all_b=float(allok)), open(os.path.expanduser('~/spore_v5/d4/out/g52_%s_%d_%d.json' % (T.ENV, T.NB, ts)), 'w'))
if __name__ == '__main__': main()
