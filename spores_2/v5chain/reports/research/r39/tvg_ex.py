"""research-39: точные рёбра узел→сосед (идея пользователя 05:4x «вперёд-назад, переходы между соседями») против LP-интерполяции — ДИ nD.
Граф на ТЕХ ЖЕ узлах, что LP-схема (RNDN, rng(1000+ts)): ребро i→j, j ∈ kNN вокруг точек баллистического прогноза P_i + (v_i·t, 0), t ∈ TEX;
вес — ТОЧНОЕ минимальное общее время T_ij перехода (оси независимы, |u_k| ≤ 1). Множество допустимых времён оси (x0,v0)→(xf,vf):
  T ≥ |vf−v0|,  Dmin(T) ≤ dx ≤ Dmax(T),  Dmax = ((a+T)/2)² − s,  Dmin = s − ((a−T)/2)²,  a = v0+vf, s = (v0²+vf²)/2
  ⇒ [|Δv|, ∞) минус открытые интервалы (−a ± 2√(dx+s)) и (a ± 2√(s−dx)) (если подкоренные > 0). Синхронизация: T = min общий допустимый ≥ max T_k*.
W_ex = кратчайший путь до узлов цели (Dijkstra по обращённому графу). Агент — как в LP-схеме (V̂ = LP-интерполяция W), меняется только W.
Коробка на траектории ребра не проверяется (рёбра короткие; концы в поле) — оговорка.
Запуск (aida ~/spore_v5/r37): KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAU=.3 RB=.35 KLP=20 VCAP=40 TAUS=.075,.15,.3 RNDN=1 SKIPBASE=1 NPROC=6
  STARTS=~/spore_v5/r37/di4_ref_300_rho35_s7.npy KEX=32 TEX=0,.15,.3,.6 python3 tvg_ex.py di4 100000 8 0"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np, numba as nb
import tvg_vt as T
import tvg_lp as LPm
from tvg_dl import run_agent
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
env, d = T.env, T.d; h = d // 2
KEX = int(os.environ.get('KEX', 32)); TEX = [float(v) for v in os.environ.get('TEX', '0,.15,.3,.6').split(',')]

@nb.njit(cache=True)
def _first(L, lo, a1, b1, a2, b2):
    """min T ≥ max(L, lo), не лежащий в открытых (a1,b1), (a2,b2)"""
    t = max(L, lo)
    for _ in range(4):
        if a1 < t < b1: t = b1
        if a2 < t < b2: t = b2
    return t

@nb.njit(cache=True)
def _axis(x0, v0, xf, vf):
    dx = xf - x0; a = v0 + vf; s = .5 * (v0 * v0 + vf * vf); lo = abs(vf - v0)
    a1 = b1 = a2 = b2 = -1.
    if dx + s > 0: r = 2. * np.sqrt(dx + s); a1 = -a - r; b1 = -a + r
    if s - dx > 0: r = 2. * np.sqrt(s - dx); a2 = a - r; b2 = a + r
    return lo, a1, b1, a2, b2

@nb.njit(parallel=True, cache=True)
def tsync(P, I, J, hh):
    n = len(I); out = np.empty(n)
    for e in nb.prange(n):
        p = P[I[e]]; q = P[J[e]]; ax = np.empty((hh, 5))
        for k in range(hh):
            lo, a1, b1, a2, b2 = _axis(p[k], p[hh + k], q[k], q[hh + k]); ax[k, 0] = lo; ax[k, 1] = a1; ax[k, 2] = b1; ax[k, 3] = a2; ax[k, 4] = b2
        t = 0.
        for k in range(hh): t = max(t, _first(0., ax[k, 0], ax[k, 1], ax[k, 2], ax[k, 3], ax[k, 4]))
        for _ in range(20):
            t0 = t
            for k in range(hh): t = max(t, _first(t, ax[k, 0], ax[k, 1], ax[k, 2], ax[k, 3], ax[k, 4]))
            if t == t0: break
        out[e] = t
    return out

def selftest():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import importlib.util
    spec = importlib.util.spec_from_file_location('ref', os.path.expanduser('~/spore_v5/r37/di4_ref_exact.py')); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
    rng = np.random.default_rng(5); A = rng.uniform(-2, 2, (20000, 4)); t1 = np.array([_first(0., *_axis(*a)) for a in A]); t2 = R.tpp(A[:, 0], A[:, 1], A[:, 2], A[:, 3])
    print('selftest ось: |T_мой − tpp| max %.2e' % np.abs(t1 - t2).max())
    # синхронизация: проверка достижимости ровно за T обеими осями через Dmin ≤ dx ≤ Dmax
    P = rng.uniform(-2, 2, (20000, 2 * 2)); Q = rng.uniform(-2, 2, (20000, 4)); Pc = np.vstack([P, Q]); I = np.arange(20000); Jj = I + 20000; ts = tsync(Pc, I, Jj, 2)
    bad = 0
    for k in range(2):
        x0, v0, xf, vf = P[:, k], P[:, 2 + k], Q[:, k], Q[:, 2 + k]; a = v0 + vf; s = .5 * (v0 ** 2 + vf ** 2); dx = xf - x0
        ok = (ts >= np.abs(vf - v0) - 1e-9) & (s - ((a - ts) / 2) ** 2 <= dx + 1e-9) & (dx <= ((a + ts) / 2) ** 2 - s + 1e-9); bad += (~ok).sum()
    print('selftest синхр.: недопустимых %d из 20000' % bad)

class AgentEX(LPm.AgentLP):
    def exact_W(self):
        t0 = time.perf_counter(); P = np.ascontiguousarray(self.P); n = self.nP; S = self.S; Is, Js = [], []
        for tt in TEX:
            Q = P.copy(); Q[:, :h] += P[:, h:] * tt
            _, jj = self.tree.query(S.wrap(Q) / S.scale, k=KEX + 1, workers=int(os.environ.get('KDW', -1))); jj = jj % n
            Is.append(np.repeat(np.arange(n), KEX + 1)); Js.append(jj.ravel())
        I = np.concatenate(Is); J = np.concatenate(Js); m = I != J; I, J = I[m], J[m]
        key = I.astype(np.int64) * n + J; key = np.unique(key); I = (key // n).astype(np.int64); J = (key % n).astype(np.int64)
        okJ = T.okv(P[J]); I, J = I[okJ], J[okJ]
        w = tsync(P, I, J, h); t1 = time.perf_counter()
        G = csr_matrix((w + 1e-12, (J, I)), shape=(n, n))   # обращённый: J→I, путь i→…→цель
        ING = np.flatnonzero(T.in_goal(P)); D = dijkstra(G, directed=True, indices=ING, min_only=True); t2 = time.perf_counter()
        D[~np.isfinite(D)] = 1e3; vcap = float(os.environ.get('VCAP', 0))
        if vcap > 0: D = np.minimum(D, vcap)
        fin = D < min(1e2, vcap if vcap > 0 else 1e9) - 1e-9
        print('  ТОЧНЫЕ рёбра: %d (%.1f/узел), веса %.1f с, Dijkstra %.1f с, узлов цели %d; W_ex конечных %.1f%%, мед %.2f; T_ребра мед %.3f p90 %.3f' % (
            len(w), len(w) / n, t1 - t0, t2 - t1, len(ING), 100 * fin.mean(), np.median(D[fin]), np.median(w), np.percentile(w, 90)), flush=True)
        return D, t2 - t0

def W0(ag, W):
    J, lam, ok = ag.ip.weights(env.X0); w = (lam * W[J]).sum(1); w[~ok] = np.nan; return w

def main():
    if os.environ.get('SELFTEST') == '1': selftest(); return
    ts = T.TS; ns = int(os.environ.get('NSTART', len(env.X0))); S = T.mkS(ts)
    rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
    back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = AgentEX(S, back); Wlp, kvlp = ag.W, T.KV; Wex, tex = ag.exact_W(); out = dict(nodes=len(P_), tlp=ag.tdl, tex=tex, KEX=KEX, TEX=TEX)
    k = env.keep[:ns]; tr = env.Tref[:ns]; a, b = W0(ag, Wlp)[:ns] / tr, W0(ag, Wex)[:ns] / tr
    for nm, r in (('W_lp', a), ('W_ex', b)):
        rr = r[k & np.isfinite(r)]; print('  %s(x0)/T_ref по графу (keep, в оболочке %d): мед %.3f p10 %.3f p90 %.3f max %.3f' % (nm, len(rr), np.median(rr), np.percentile(rr, 10), np.percentile(rr, 90), rr.max()), flush=True)
    out['W0lp'] = a.tolist(); out['W0ex'] = b.tolist()
    ag.mode = 'ip'
    if os.environ.get('SKIPLP') != '1':
        ag.W, T.KV = Wlp, kvlp; Ts, s = run_agent(ag, ns); out['lp_ip'] = dict(T=Ts, **s); print('W_lp, V̂ LP-интерп: %s' % s, flush=True)
    ag.W = Wex; T.KV = T.kv_auto(S, ag.tree, ag.P, Wex); Ts, s = run_agent(ag, ns); out['ex_ip'] = dict(T=Ts, **s); print('W_ex (точные рёбра), V̂ LP-интерп: %s' % s, flush=True)
    json.dump(out, open(os.path.expanduser('~/spore_v5/r37/ex_%s_%d_K%d_%d%s.json' % (T.ENV, T.NB, KEX, ts, os.environ.get('TAG', ''))), 'w'), default=float)

if __name__ == '__main__': main()
