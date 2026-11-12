"""research-37 (graph_to_cells.md шаг 1, задача senior-2): прореживание графа.
Узел i убираем, если W_i воспроизводится LP-интерполяцией по KLP соседям БЕЗ него с ошибкой ≤ EPS (и он не в цели, W < VCAP).
Раунд: кандидаты по возрастанию ошибки, жадно независимое множество (соседи удалённого в этом раунде не удаляются), не больше FRAC·n;
затем пересборка LP-весов и VI (VCAP), замер агента. Контроль — равномерные узлы того же числа (тот же агент).
Источник узлов: THINSRC=npz (P) или равномерные N. Запуск: ... THINSRC=~/spore_v5/r35/refP_0.npz EPS=.02 FRAC=.15 ROUNDS=6 python3 tvg_thin.py pend 0 8 SEED"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
import tvg_vt as T
from tvg_lp import AgentLP, lp_batch, KLP
from tvg_dl import run_agent
env, d = T.env, T.d
if os.environ.get("S100") == "1":   # как tvg_ref.py research-36: 100 стартов атласа маятника
    _rq = np.random.default_rng(0); env.X0 = np.stack([_rq.uniform(-np.pi, np.pi, 100), _rq.uniform(-2, 2, 100)], 1)
    _rf = np.load("/home/random/spore_v5/r30/spores_2/v7/src/cells7/pend_ref_T.npy"); env.Tref = _rf[:, 0] if _rf.ndim == 2 else _rf; env.keep = np.isfinite(env.Tref) & (env.Tref > .05)
EPS = float(os.environ.get('EPS', .02)); FRAC = float(os.environ.get('FRAC', .15)); ROUNDS = int(os.environ.get('ROUNDS', 6)); VCAP = float(os.environ.get('VCAP', 40))

def box(rng, n):
    lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
    P = rng.uniform(lo_, hi_, (3 * n, d)); P = P[T.okv(P)][:n]; return P

def mkag(S, P):
    back = (P, np.zeros(len(P)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P))))
    ag = AgentLP(S, back); ag.mode = 'ip'; return ag

def loo_err(S, ag):
    """ошибка интерполяции W_i по соседям без самого узла (leave-one-out)"""
    n = ag.nP; Qs = S.wrap(ag.P) / S.scale; dd, jj = ag.tree.query(Qs, k=KLP + 1, workers=int(os.environ.get('KDW', -1)))
    self_ = (jj % n) == np.arange(n)[:, None]; order = np.argsort(self_, axis=1, kind='stable')[:, :KLP]   # выкинуть себя (и свои образы)
    jj = np.take_along_axis(jj, order, 1); NB_ = ag.tree.data[jj]; Jl, L, ok = lp_batch(np.ascontiguousarray(NB_), np.ascontiguousarray(Qs))
    J = np.take_along_axis(jj, Jl, 1) % n; Wc = np.minimum(ag.W, VCAP); w = (L * Wc[J]).sum(1)
    err = np.where(ok, np.abs(w - Wc), np.inf); return err, jj % n

def thin_round(S, ag, rng):
    n = ag.nP; err, nbr = loo_err(S, ag); cand = np.flatnonzero((err <= EPS) & (ag.W < VCAP - 1) & ~T.in_goal(ag.P))
    cand = cand[np.argsort(err[cand], kind='stable')]; lock = np.zeros(n, bool); rm = []; lim = int(FRAC * n)
    for i in cand:
        if lock[i]: continue
        rm.append(i); lock[i] = True; lock[nbr[i]] = True
        if len(rm) >= lim: break
    keep = np.ones(n, bool); keep[rm] = False
    return ag.P[keep], len(cand), float(np.median(err[np.isfinite(err)]))

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts); rng = np.random.default_rng(3000 + ts); t0 = time.perf_counter()
    src = os.environ.get('THINSRC')
    P = np.load(os.path.expanduser(src))['P'] if src else box(np.random.default_rng(1000 + ts), T.NB)
    out = dict(src=src, eps=EPS, frac=FRAC, rounds=[])
    for r in range(ROUNDS + 1):
        ag = mkag(S, P); Ts, s = run_agent(ag, ns); rec = dict(round=r, nodes=len(P), t=time.perf_counter() - t0, T=Ts, **s)
        if os.environ.get('CTRL', '1') == '1' and r > 0:
            agu = mkag(S, box(np.random.default_rng(2000 + 7 * r + ts), len(P))); Tu, su = run_agent(agu, ns); rec['uni'] = dict(T=Tu, **su)
        out['rounds'].append(rec)
        print('раунд %d: узлов %d, t %.0f с | прорежен: дошли %d/%d мед %.3f p90 %.3f max %.3f | равномерные: %s' % (
            r, len(P), rec['t'], s['reach'], s['n'], s['med'], s['p90'], s['max'],
            ('%d/%d мед %.3f p90 %.3f max %.3f' % (su['reach'], su['n'], su['med'], su['p90'], su['max'])) if 'uni' in rec else '—'), flush=True)
        if r == ROUNDS: break
        P2, nc, em = thin_round(S, ag, rng); print('   кандидатов (ошибка ≤ %.3f) %d, мед ошибка LOO %.4f, убрано %d' % (EPS, nc, em, len(P) - len(P2)), flush=True)
        if len(P2) == len(P): break
        P = P2
    json.dump(out, open(os.path.expanduser('~/spore_v5/r37/thin_%s_%s_%g_%d%s.json' % (T.ENV, 'src' if src else str(T.NB), EPS, ts, os.environ.get('TAG', ''))), 'w'), default=float)

if __name__ == '__main__': main()
