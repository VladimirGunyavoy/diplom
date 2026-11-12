"""research-36: В-шов в любой размерности — выпуклые веса по k ближайшим через малый LP:
min Σλ_j |x_j−q|²  s.t. Σλ_j x_j = q, Σλ_j = 1, λ ≥ 0  (по поднятию на параболоид = симплекс Делоне среди k соседей).
Пакетно в numba (Big-M симплекс, правило Бланда). q вне оболочки соседей → ребра нет (VI) / фолбэк nn (агент).
Запуск: RNDN=1 KVAUTO=1 RB=.03 TAU=.2 KLP=12 python3 tvg_lp.py pend 100000 8 0   (ENV di4 — 4D)"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/c12/mem/spores_2/w39'))
import numpy as np, numba as nb
import tvg_vt as T
from tvg_dl import vit_dl, run_agent
env, d = T.env, T.d
KLP = int(os.environ.get('KLP', 4 * d + 4)); NIT = int(os.environ.get('NITDL', 3000))

@nb.njit(cache=True)
def lp1(X, c, out_j, out_l):
    k, dd = X.shape; m = dd + 1; nv = k + m; Tb = np.zeros((m + 1, nv + 1)); basis = np.empty(m, np.int64); M = 1e4
    for i in range(dd):
        for j in range(k): Tb[i, j] = X[j, i]
    for j in range(k): Tb[dd, j] = 1.
    Tb[dd, nv] = 1.
    for i in range(m): Tb[i, k + i] = 1.; basis[i] = k + i
    for j in range(k): Tb[m, j] = c[j]
    for i in range(m): Tb[m, k + i] = M
    for i in range(m):
        for j in range(nv + 1): Tb[m, j] -= M * Tb[i, j]
    for it in range(200):
        e = -1
        for j in range(nv):
            if Tb[m, j] < -1e-10: e = j; break
        if e < 0: break
        r = -1; best = 1e300
        for i in range(m):
            if Tb[i, e] > 1e-12:
                q = Tb[i, nv] / Tb[i, e]
                if q < best - 1e-14 or (abs(q - best) <= 1e-14 and basis[i] < basis[r]): best = q; r = i
        if r < 0: return False
        p = Tb[r, e]
        for j in range(nv + 1): Tb[r, j] /= p
        for i in range(m + 1):
            if i != r and Tb[i, e] != 0.:
                f = Tb[i, e]
                for j in range(nv + 1): Tb[i, j] -= f * Tb[r, j]
        basis[r] = e
    s = 0.
    for i in range(m):
        if basis[i] >= k:
            if Tb[i, nv] > 1e-9: return False
            out_j[i] = 0; out_l[i] = 0.
        else: out_j[i] = basis[i]; out_l[i] = max(Tb[i, nv], 0.); s += out_l[i]
    if s <= 0: return False
    for i in range(m): out_l[i] /= s
    return True

@nb.njit(parallel=True, cache=True)
def lp_batch(NB_, Qs):
    n, k, dd = NB_.shape; m = dd + 1; J = np.zeros((n, m), np.int64); L = np.zeros((n, m)); OK = np.zeros(n, np.bool_)
    for t in nb.prange(n):
        X = NB_[t] - Qs[t]; c = np.empty(k)
        cm = 0.
        for j in range(k):
            s = 0.
            for i in range(dd): s += X[j, i] * X[j, i]
            c[j] = s; cm = max(cm, s)
        for j in range(k): c[j] /= (cm + 1e-300)
        jj = np.zeros(m, np.int64); ll = np.zeros(m); OK[t] = lp1(X, c, jj, ll); J[t] = jj; L[t] = ll
    return J, L, OK

class InterpLP:
    def __init__(self, S, tree, nP): self.S, self.tree, self.nP = S, tree, nP
    def weights(self, Q, ch=None, J_out=None, L_out=None):
        """порциями по ch точек (по умолчанию из LPCH_MB: ~ MB на порцию), J int32, lam float32; пик ≈ хранимое + одна порция"""
        Q = np.asarray(Q, float); n = len(Q); K = d + 1
        if ch is None: ch = max(1000, int(float(os.environ.get('LPCH_MB', 300)) * 1e6 / (KLP * d * 8 * 3)))
        J = np.empty((n, K), np.int32) if J_out is None else J_out; L = np.empty((n, K), np.float32 if os.environ.get('LAM32') == '1' else np.float64) if L_out is None else L_out; ok = np.empty(n, bool)
        for a in range(0, n, ch):
            b = min(n, a + ch); Qs = self.S.wrap(Q[a:b]) / self.S.scale; dd, jj = self.tree.query(Qs, k=KLP, workers=int(os.environ.get('KDW', -1)))
            Jl, Lc, okc = lp_batch(np.ascontiguousarray(self.tree.data[jj]), np.ascontiguousarray(Qs))
            J[a:b] = np.take_along_axis(jj, Jl, 1) % self.nP; L[a:b] = Lc; ok[a:b] = okc; del dd, jj, Jl, Lc
        return J, L, ok

class AgentLP(T.Agent):
    def __init__(self, S, back):
        T.Agent.__init__(self, S, back); self.Wnn = self.W.copy(); self.kvnn = T.KV
        t0 = time.perf_counter(); self.ip = InterpLP(S, self.tree, self.nP); n = len(self.P); L2 = len(T.TAUV); K = d + 1
        J = np.empty((n * L2, K), np.int32); LAM = np.empty((n * L2, K), np.float32 if os.environ.get('LAM32') == '1' else np.float64); OK = np.empty((n, L2), bool); GL = np.empty((n, L2)); ins = np.empty((n, L2), bool)
        nch = max(1, int(float(os.environ.get('EDGCH', 5000))))   # узлов в порции рёбер
        for a in range(0, n, nch):
            b = min(n, a + nch); E, OK[a:b], GL[a:b] = T.edges(self.P[a:b], S)
            _, _, okc = self.ip.weights(E.reshape(-1, d), J_out=J[a * L2:b * L2], L_out=LAM[a * L2:b * L2]); ins[a:b] = okc.reshape(-1, L2); del E
        OKd = (OK & ins) | (OK & (GL > 0)); t1 = time.perf_counter()
        print('  LP-веса (k=%d): %.1f с, рёбер вне оболочки соседей %.3f%%' % (KLP, t1 - t0, 100 * (OK & ~ins & (GL <= 0)).sum() / max(1, (OK & (GL <= 0)).sum())), flush=True)
        self.W, it, res = vit_dl(J, LAM, OKd, GL, T.in_goal(self.P), self.nP, T.TAUV, NIT, 1e3, float(os.environ.get("VCAP", 0))); self.tdl = time.perf_counter() - t0
        fin = self.W < 1e2; print('  VI-LP: %.1f с, it %d res %.1e; [у потолка %.1f%%] W конечных %.1f%%, мед %.2f макс %.2f' % (self.tdl, it, res, 100 * (self.W >= float(os.environ.get('VCAP', 1e9)) - 1e-9).mean(), 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max()), flush=True)
        T.KV = T.kv_auto(S, self.tree, self.P, self.W); self.mode = 'nn'
    def vh(self, Q):
        if self.mode == 'nn': return T.Agent.vh(self, Q)
        Q = np.asarray(Q, float); J, lam, ok = self.ip.weights(Q); w = (lam * self.W[J]).sum(1)
        bad = ~ok | (self.W[J] >= 1e2).any(1)
        if bad.any(): w[bad] = T.Agent.vh(self, Q[bad])
        return w

def main():
    ts = T.TS; ns = int(os.environ.get('NSTART', len(env.X0))); S = T.mkS(ts); t0 = time.perf_counter()
    if os.environ.get('RNDN') == '1':
        rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
        P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
        back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    else: back = T.build_back(S, T.TAU, T.NB, T.RB)
    print('build %.1f с, узлов %d' % (time.perf_counter() - t0, len(back[0])), flush=True)
    ag = AgentLP(S, back); out = dict(nodes=len(back[0]), tlp=ag.tdl); Wd, kvd = ag.W, T.KV
    if os.environ.get('SKIPBASE') != '1':
        ag.W, T.KV, ag.mode = ag.Wnn, ag.kvnn, 'nn'; Ts, s = run_agent(ag, ns); out['base'] = dict(T=Ts, **s); print('БАЗА nn+KV: %s' % s, flush=True)
    ag.W, T.KV, ag.mode = Wd, kvd, 'ip'; Ts, s = run_agent(ag, ns); out['lp_ip'] = dict(T=Ts, **s); print('W_lp, V̂ LP-интерп: %s' % s, flush=True)
    import resource; out['rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6; np.save(os.path.expanduser('~/spore_v5/c12/memout/W_%s_%d_%s.npy' % (T.ENV, T.NB, os.environ.get('TAG', ''))), ag.W)
    json.dump(out, open(os.path.expanduser('~/spore_v5/c12/memout/lp_%s_%d_%g_%s_%d' + os.environ.get('TAG', '') + '.json') % (T.ENV, T.NB, T.TAU, 'rnd' if os.environ.get('RNDN') == '1' else 'tree', ts), 'w'), default=float)

if __name__ == '__main__': main()
