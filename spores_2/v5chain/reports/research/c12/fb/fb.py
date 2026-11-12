"""c12 / research-39: ВПЕРЁД-СБОР против НАЗАД-РАЗДАЧИ на ДИ4D 100k (RNDN, LP K20, VCAP40, TAUS .075/.15/.3). НЕ ЗАПУСКАЛСЯ (машины возвращены 12.11).
Запуск (когда разрешат), из ~/spore_v5/c12fb (там tvg_vt.py, tvg_dl.py, tvg_lp.py r37-копии, di4_ref_exact.py):
  NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8 KDW=8 NPROC=8 RNDN=1 KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAU=.15 KLP=20 VCAP=40 TAUS=.075,.15,.3 \
  STARTS=~/spore_v5/r37/di4_ref_300_rho35_s7.npy python3 fb.py di4 100000 8 0
Шаги: (1) V* в узлах (tbox по осям, max); (2) e_f = W_lp − V*, e_I = I_LP[V*](y) − V*(y) в точках прилёта; (3) назад-раздача W_b: облако прообразов
z_j = Φ_{−τ}^u(x_j) на каждый из 27 классов, LP-веса узла x_i по облаку, vit_dl тем же кодом; (4) детектор D=|W_lp−W_b| против |e_f| (Спирмен, топ-10%)
и AUC для узлов вблизи излома V* (порог senior-2: AUC ≥ .8); (5) агент с W_lp / W_b / среднее.
Память ~5 ГБ."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.expanduser('~/spore_v5/c12fb'))
import numpy as np
import tvg_lp as L
T = L.T; env, d = T.env, T.d
from tvg_dl import vit_dl, run_agent
from multiprocessing import Pool
from scipy.stats import spearmanr
import di4_ref_exact as R

NP = int(os.environ.get('FBPROC', 16)); VCAP = float(os.environ.get('VCAP', 40)); NIT = int(os.environ.get('NITDL', 3000))

def _vstar(Q): return np.array([R.tstar(q) for q in Q])   # (m,2): T по осям
def vstar(Q):
    Q = np.asarray(Q, float)
    with Pool(NP) as p: return np.concatenate(p.map(_vstar, np.array_split(Q, NP * 8)))   # (n,2)

def stats(x):
    x = np.asarray(x); return dict(med=float(np.median(x)), p90=float(np.percentile(np.abs(x), 90)), mean=float(x.mean()), pos=float((x > 0).mean()))

def auc(score, label):
    """AUC: P(score_pos > score_neg), через ранги"""
    from scipy.stats import rankdata
    r = rankdata(score); npos = label.sum(); nneg = len(label) - npos
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else float('nan')

def main():
    ts = T.TS; S = T.mkS(ts); out = {}
    rng_ = np.random.default_rng(1000 + ts); lo_, hi_ = T.LO, T.HI
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
    back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = L.AgentLP(S, back); Wlp = ag.W.copy(); P = ag.P; n = len(P); L2 = len(T.TAUV); K = d + 1
    ING = T.in_goal(P)
    # (1) V*
    t0 = time.perf_counter(); Va = vstar(P); Vs = Va.max(1); Vs[ING] = 0.; AX = Va.argmax(1); print('V* %.0f с' % (time.perf_counter() - t0), flush=True)
    fin = (Wlp < VCAP - 1e-9) & ~ING
    ef = (Wlp - Vs)[fin]; out['e_f'] = stats(ef); out['e_f']['frac_gt10pct'] = float((np.abs(ef) > .1 * Vs[fin]).mean()); print('e_f', out['e_f'], flush=True)
    # (2) ошибка интерполяции в точках прилёта
    E, OK, GL = T.edges(P, S); rng = np.random.default_rng(5)
    ii, aa = np.nonzero(OK & (GL <= 0)); sel = rng.choice(len(ii), min(60000, len(ii)), replace=False); Y = E[ii[sel], aa[sel]]; del E
    keep = ~T.in_goal(Y); Y = Y[keep]; Jj, Ll, ok = ag.ip.weights(Y); Vy = vstar(Y).max(1)
    I = (Ll.astype(float) * Vs[Jj]).sum(1); eI = (I - Vy)[ok]; Yk = Y[ok]; sp = (Vs[Jj].max(1) - Vs[Jj].min(1))[ok]
    out['e_I'] = stats(eI); out['e_I']['frac_gt10pct'] = float((np.abs(eI) > .1 * Vy[ok]).mean())
    out['e_I_spread_spearman'] = float(spearmanr(np.abs(eI), sp)[0])
    rin = np.abs(Yk).max(1); out['e_I_by_radius'] = {('%.1f' % a): float(np.percentile(np.abs(eI[(rin >= a) & (rin < a + .5)]), 90)) for a in np.arange(0, 3.5, .5) if ((rin >= a) & (rin < a + .5)).sum() > 50}
    print('e_I', out['e_I'], out['e_I_by_radius'], flush=True)
    # (3) назад-раздача: облако прообразов на класс, веса узла x_i по облаку
    Jb = np.empty((n * L2, K), np.int32); Lb = np.empty((n * L2, K), np.float32); insb = np.empty((n, L2), bool); t0 = time.perf_counter()
    for a in range(L2):
        z = env.step(P, np.broadcast_to(T.ACT[T.SIDX[a]], P.shape[:-1] + (env.m,)), -T.TAUV[a])
        ipb = L.InterpLP(S, T.cKDTree(S.wrap(z) / S.scale), n)
        J, Lm, okb = ipb.weights(P); Jb[a::L2] = J; Lb[a::L2] = Lm; insb[:, a] = okb
    print('назад-веса %.0f с' % (time.perf_counter() - t0), flush=True)
    OKb = (OK & insb) | (OK & (GL > 0))
    Wb, it, res = vit_dl(Jb, Lb, OKb, GL, ING, n, T.TAUV, NIT, 1e3, VCAP); print('W_b: it %d res %.1e, конечных %.1f%%' % (it, res, 100 * (Wb < 1e2).mean()), flush=True)
    # (4) детектор
    fb = fin & (Wb < VCAP - 1e-9); D = np.abs(Wlp - Wb)[fb]; ea = np.abs(Wlp - Vs)[fb]
    out['detector'] = dict(spearman=float(spearmanr(D, ea)[0]), top10=float(np.isin(np.argsort(-D)[:len(D) // 10], np.argsort(-ea)[:len(D) // 10]).mean()))
    # излом V*: узел «у излома», если среди 10 ближайших узлов (один шаг узлов) сменилась ось-максимум V*; AUC детектора D по этой метке (порог senior-2: ≥ .8)
    _, nn = ag.tree.query(T._scaled(S, P), k=11, workers=int(os.environ.get('KDW', -1))); nn = nn[:, 1:] % n
    kink = (AX[nn] != AX[:, None]).any(1)[fb]; out['detector']['kink_frac'] = float(kink.mean()); out['detector']['auc_kink'] = auc(D, kink)
    out['detector']['auc_kink_ea'] = auc(ea, kink)   # потолок: если бы D = точной ошибке
    print('детектор', out['detector'], flush=True)
    # (5) агент
    ns = int(os.environ.get('NSTART', len(env.X0))); kvd = T.KV; ag.mode = 'ip'
    for name, W in (('lp', Wlp), ('b', Wb), ('avg', .5 * (Wlp + Wb))):
        ag.W = W; Ts, s = run_agent(ag, ns); out['agent_' + name] = dict(T=Ts, **s); print('агент %s: %s' % (name, s), flush=True)
    json.dump(out, open(os.path.expanduser('~/spore_v5/c12fb/fb_%s_%d_%d.json' % (T.ENV, T.NB, ts)), 'w'), default=float)
    np.savez_compressed(os.path.expanduser('~/spore_v5/c12fb/fb_nodes_%d_%d.npz' % (T.NB, ts)), P=P, Wlp=Wlp, Wb=Wb, Vs=Vs)

if __name__ == '__main__': main()
