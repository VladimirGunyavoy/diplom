"""w39: рецепт «обратное дерево + V (итерация ценностей по узлам) + жадный агент» на стендах bench/common (di4, pend, pend15).
Образец — w38_car/car_val_w38.py. Запуск (aida, ~/spore_v5/a39/spores_2/w39): python tvg.py ENV NB K TREESEED
ENV: TAU (.2), RB (.3), KV (.05), SC ('1,1,1,1'), NIT (300), NU3 (di4: 1 — сетка {-1,0,1}², 0 — только углы), NSTART (все)."""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'bench')); sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
import numpy as np, numba as nb, common
from src.atlas6.adaptive_nd import SysN, build_back, _images, _scaled
from scipy.spatial import cKDTree
ENV, NB, KN, TS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
TAU = float(os.environ.get('TAU', .2)); RB = float(os.environ.get('RB', .3)); KV = float(os.environ.get('KV', .05)); HYST = float(os.environ.get('HYST', 0)); DEP = int(os.environ.get('DEP', 1)); TIEP = float(os.environ.get('TIEP', 0)); DEPB = int(os.environ.get('DEPB', 1))
env = common.make(ENV); d = env.n
SC = tuple(float(v) for v in os.environ.get('SC', ','.join(['1'] * d)).split(','))
if d == 4 and os.environ.get('NU3', '1') == '1': ACT = np.array([[a, b] for a in (-1., 0., 1.) for b in (-1., 0., 1.)])
elif d == 4: ACT = np.array([[a, b] for a in (-1., 1.) for b in (-1., 1.)])
else: ACT = np.array([[-env.umax[0]], [0.], [env.umax[0]]]) if os.environ.get('NU3', '1') == '1' else np.array([[-env.umax[0]], [env.umax[0]]])
L = len(ACT); per = tuple(env.period)

def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); n = max(1, int(np.ceil(abs(t) / (.02 if ENV != 'di4' else 10.)))); h = t / n; Z = P
    u = np.broadcast_to(ACT[s], Z.shape[:-1] + (env.m,))
    for _ in range(n): Z = env.step(Z, u, h)
    return Z
GL_HALF = env.half - .02 if env.ball is None else None
def in_goal(P):
    P = np.asarray(P); return (np.abs(common.wrap(P, env.period)) <= env.half - .02).all(-1)
LO, HI = env.lo.copy(), env.hi.copy()
if d == 4: LO, HI = env.fl_lo - .25, env.fl_hi + .25
def okv(P): P = np.asarray(P); return ~(((P < LO) | (P > HI)) & (env.period == 0)).any(-1)
def mkS(seed):
    rng = np.random.default_rng(seed); n = 300; hw = env.half - .02
    seeds = rng.uniform(-hw, hw, (n, d)); return SysN(flow, L, SC, in_goal, seeds, per=per, ok=okv, blocked=None)

def edges(P, S):
    n = len(P); E = np.zeros((n, L, d)); OK = np.zeros((n, L), bool); GL = np.zeros((n, L)); NS = 5; h = TAU / NS
    for s in range(L):
        Z = P.copy(); u = np.broadcast_to(ACT[s], Z.shape[:-1] + (env.m,)); ok = np.ones(n, bool); hit = np.zeros(n)
        for k in range(NS):
            Zs = Z
            for _ in range(max(1, int(np.ceil(h / (.02 if ENV != 'di4' else 10.))))): Zs = env.step(Zs, u, h / max(1, int(np.ceil(h / (.02 if ENV != 'di4' else 10.)))))
            Z = Zs; ok &= okv(Z); g = in_goal(Z) & (hit == 0); hit[g] = (k + 1) * h
        E[:, s] = Z; OK[:, s] = ok; GL[:, s] = hit
    return E, OK, GL

@nb.njit(parallel=True)
def vit(J, D, OK, GL, ING, nP, tau, kv, nit, big):
    W = np.full(nP, big); K_ = J.shape[1]; L_ = OK.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(L_):
                if not OK[i, a]: continue
                if GL[i, a] > 0: c = GL[i, a]
                else:
                    m = 1e30
                    for k in range(K_):
                        v = W[J[i * L_ + a, k]] + kv * D[i * L_ + a, k]
                        if v < m: m = v
                    c = tau + m
                if c < best: best = c
            Wn[i] = best
        dd = np.max(np.abs(Wn - W)); W = Wn
        if dd < 1e-4: break
    return W, it, dd


def kv_auto(S, tree, P, W, k=9, q=.9):
    """квантиль q локальной Липшица V по рёбрам графа узлов (kNN в метрике S.scale): |W_i-W_j|/d_ij"""
    dd, jj = tree.query(S.wrap(P) / S.scale, k=k, workers=-1); jj = jj % len(P); i = np.arange(len(P))[:, None]
    m = (dd > 1e-9) & (W[:, None] < 1e2) & (W[jj] < 1e2); r = np.abs(W[:, None] - W[jj])[m] / dd[m]; return float(np.quantile(r, q))

class Agent:
    def __init__(self, S, back):
        global KV
        self.S, self.tree, self.P = S, back[5], np.asarray(back[0]); self.nP = len(self.P)
        UNI = float(os.environ.get('UNI', 0))
        if UNI > 0:   # равномерные узлы вне дерева: доля UNI от узлов дерева, случайно по полю; граф/VI общие
            rng = np.random.default_rng(1000 + TS); n = int(UNI * self.nP); lo = np.where(env.period > 0, -env.period / 2, LO); hi = np.where(env.period > 0, env.period / 2, HI)
            U = rng.uniform(lo, hi, (n, d)); U = U[okv(U)]; self.P = np.vstack([self.P, U]); self.nP = len(self.P); self.tree = cKDTree(_scaled(S, _images(S, self.P)))
        t0 = time.perf_counter(); E, OK, GL = edges(self.P, S); t1 = time.perf_counter()
        Ew = S.wrap(E.reshape(-1, d)); dd, j = self.tree.query(Ew / S.scale, k=min(KN * 3, len(self.tree.data)), workers=-1)
        J = (j % self.nP).astype(np.int32); D = dd.astype(np.float64); t2 = time.perf_counter()
        self.W, it, res = vit(J, D, OK, GL, in_goal(self.P), self.nP, TAU, KV, int(os.environ.get('NIT', 300)), 1e3); t3 = time.perf_counter(); self.tvi = t3 - t0
        if os.environ.get('KVAUTO') == '1':
            for _ in range(10):
                kn_ = kv_auto(S, self.tree, self.P, self.W); conv = abs(kn_ - KV) < .05 * kn_; KV = kn_
                if conv: break
                self.W, it, res = vit(J, D, OK, GL, in_goal(self.P), self.nP, TAU, KV, int(os.environ.get('NIT', 300)), 1e3)
            self.tvi = time.perf_counter() - t0
        fin = self.W < 1e2; self.kva = kv_auto(S, self.tree, self.P, self.W); print('  KV_auto (q.9 Липшица V по kNN-рёбрам) = %.3f при KV=%g' % (self.kva, KV), flush=True)
        print('  узлов %d: edges %.1f, knn %.1f, vi %.1f с (it %d res %.1e); W конечных %.1f%%, мед %.2f макс %.2f' % (self.nP, t1 - t0, t2 - t1, t3 - t2, it, res, 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max()), flush=True)
    def _w(self, x):
        d_, j_ = self.tree.query(self.S.wrap(np.asarray(x, float)[None]) / self.S.scale, k=min(KN * 3, self.nP)); return float((self.W[j_ % self.nP] + KV * d_).min())
    def _cost(self, X, dep):
        """X (n,d) -> (n,): min_a [GL>0 ? GL : TAU + (dep==1 ? W^(E_a) : cost(E_a,dep-1))]"""
        E2, OK2, GL2 = edges(X, self.S); n = len(X)
        if dep == 1:
            d2, j2 = self.tree.query(self.S.wrap(E2.reshape(-1, d)) / self.S.scale, k=min(KN * 3, len(self.tree.data))); j2 = j2 % self.nP
            sub = (self.W[j2] + KV * d2).min(1).reshape(n, L)
        else:
            sub = np.full((n, L), 1e9); m = OK2 & (GL2 <= 0)
            if m.any(): sub[m] = self._cost(E2[m], dep - 1)
        leaf = np.where(GL2 > 0, GL2, TAU + sub); return np.where(OK2, leaf, 1e9).min(1)
    def act(self, x0):
        E, OK, GL = edges(np.asarray(x0, float)[None], self.S); E, OK, GL = E[0], OK[0], GL[0]
        sc = np.full(L, 1e9)
        for s in range(L):
            if OK[s] and GL[s] > 0: sc[s] = GL[s]
        if (sc < 1e9).any(): return int(np.argmin(sc)), True
        if not OK.any(): return 0, False
        dd, j = self.tree.query(self.S.wrap(E) / self.S.scale, k=min(KN * 3, len(self.tree.data))); j = j % self.nP
        Wq = TAU + (self.W[j] + KV * dd).min(1); c = np.where(OK, Wq, 1e9)
        dep = DEP
        if TIEP > 0:   # c9 TIEP: lookahead DEP только если ни одно действие не даёт прогресса: min_a W^(конец) >= W^(x) - TIEP*tau
            wx = self._w(x0); wend = np.where(OK, Wq - TAU, 1e9).min(); self.nst = getattr(self, 'nst', 0) + 1
            if wend >= wx - TIEP * TAU: self.ntie = getattr(self, 'ntie', 0) + 1
            else: dep = DEPB
        if dep >= 2:   # поиск вперёд на dep шагов: c[a] = TAU + cost(E_a, dep-1)
            idx = np.flatnonzero(OK); c = np.full(L, 1e9); c[idx] = TAU + self._cost(E[idx], dep - 1)
            c[idx] = np.minimum(c[idx], Wq[idx] + TAU)
        k = int(np.argmin(c)); pv = getattr(self, 'prev', None)
        if HYST > 0 and pv is not None and OK[pv] and c[pv] <= c[k] + HYST: k = pv
        self.prev = k; return k, False

def episode(ag, x0, tmax=40.):
    ag.prev = None; x = np.array(x0, float); us = []; dts = []; tms = []; t = 0.
    while t < tmax:
        t1 = time.perf_counter(); s, fin = ag.act(x); tms.append(time.perf_counter() - t1); us.append(ACT[s]); dts.append(TAU)
        x = env.step(x[None], ACT[s][None], TAU)[0] if ENV == 'di4' else flow(x, s, TAU)
        t += TAU
        if fin or in_goal(x): break
    Tv, why = common.verify(env, x0, us, dts)
    return Tv, why, float(np.mean(tms))

if __name__ == '__main__':
    ns = int(os.environ.get('NSTART', len(env.X0))); out = {}
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0; print('ts %d: build %.1f с, узлов %d' % (ts, tb, len(back[0])), flush=True)
        ag = Agent(S, back); Ts = []; ms = []; te = time.perf_counter()
        for i in range(ns):
            if not env.keep[i]: Ts.append(None); ms.append(0); continue
            T, why, m = episode(ag, env.X0[i]); Ts.append(T); ms.append(m)
        tq = (time.perf_counter() - te) / max(1, env.keep[:ns].sum())
        sm = common.summarize(env, Ts + [None] * (len(env.X0) - ns)) if ns == len(env.X0) else None
        k = env.keep[:ns]; T = np.array([np.nan if t is None else t for t in Ts]); ok = np.isfinite(T) & (T <= 3 * env.Tref[:ns]) & k; r = T[ok] / env.Tref[:ns][ok]
        print('%s NB=%d K=%d TAU=%g RB=%g KV=%g SC=%s ts=%d: дошли %d/%d, T/T_ref мед %.3f mean %.3f p90 %.3f max %.3f, мс/шаг %.2f, с/старт %.2f, build %.1f vi %.1f с' % (
            ENV, NB, KN, TAU, RB, KV, SC, ts, ok.sum(), k.sum(), np.median(r), r.mean(), np.percentile(r, 90), r.max(), 1e3 * np.mean(ms), tq, tb, ag.tvi), flush=True)
        dn, _j = ag.tree.query(S.wrap(env.X0[:ns]) / S.scale, k=1); wi = np.argsort(-(T / env.Tref[:ns]))[:4] if False else np.argsort(-np.nan_to_num(T / env.Tref[:ns], nan=9))[:4]
        print('   d_near: мед %.4f, у 4 худших стартов %s (T/Tref %s)' % (np.median(dn[k]), np.round(dn[wi], 3), np.round((T / env.Tref[:ns])[wi], 2)), flush=True)
        print('   TIEP=%g DEP=%d DEPB=%d: доля шагов с lookahead %.3f (%d/%d)' % (TIEP, DEP, DEPB, getattr(ag, 'ntie', 0) / max(1, getattr(ag, 'nst', 0)), getattr(ag, 'ntie', 0), getattr(ag, 'nst', 0)), flush=True)
        out[ts] = dict(T=Ts, build=tb, nodes=len(back[0]))
    json.dump(out, open('tvg_%s_%d_%g_%g_%g_%d_dep%d_tiep%g_%d.json' % (ENV, NB, TAU, RB, KV, TS, DEP, TIEP, DEPB), 'w'), default=float)
