"""research-36: В-шов (senior-2, vi_seam_interp.md): VI с выпуклой интерполяцией W по Делоне вместо «ближайший сосед + KV·d».
Запуск: cd ~/spore_v5/d3/out && RNDN=1 KVAUTO=1 RB=.03 TAU=.2 python3 tvg_dl.py pend 100000 8 0
Базовая строка (nn+KV) считается в том же прогоне (tvg_vt.Agent), затем W_dl; агент: (a) V̂ = min(W_j + KV_auto(W_dl)·d), (b) V̂ = барицентрическая W_dl."""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/d3'))
import numpy as np, numba as nb
import tvg_vt as T
from scipy.spatial import Delaunay
env, d = T.env, T.d
if os.environ.get("S100") == "1":   # research-36: 100 стартов атласа (grow_cells2d.starts_ref: rng(0)), эталон pend_ref_T.npy как у строки атласа big.md
    _rq = np.random.default_rng(0); env.X0 = np.stack([_rq.uniform(-np.pi, np.pi, 100), _rq.uniform(-2, 2, 100)], 1)
    _rf = np.load("/home/random/spore_v5/r30/spores_2/v7/src/cells7/pend_ref_T.npy"); env.Tref = _rf[:, 0] if _rf.ndim == 2 else _rf; env.keep = np.isfinite(env.Tref) & (env.Tref > .05)
NIT = int(os.environ.get('NITDL', 3000)); MARG = float(os.environ.get('MARG', .5))

def periodic_pts(P):
    """P + образы по периодическим осям в полосе MARG у края; idx -> исходный узел"""
    X, I = [P], [np.arange(len(P))]
    for k in np.nonzero(env.period)[0]:
        Xn, In = [], []
        for Z, J in zip(X, I):
            per = env.period[k]
            m = Z[:, k] > per / 2 - MARG; Y = Z[m].copy(); Y[:, k] -= per; Xn.append(Y); In.append(J[m])
            m = Z[:, k] < -per / 2 + MARG; Y = Z[m].copy(); Y[:, k] += per; Xn.append(Y); In.append(J[m])
        X += Xn; I += In
    return np.vstack(X), np.concatenate(I)

class Interp:
    def __init__(self, S, P):
        self.S = S; X, self.I = periodic_pts(np.asarray(P)); self.tri = Delaunay(X / S.scale)
    def weights(self, Q):
        Qs = self.S.wrap(np.asarray(Q, float)) / self.S.scale; s = self.tri.find_simplex(Qs); ok = s >= 0; s0 = np.where(ok, s, 0)
        Tm = self.tri.transform[s0]; b = np.einsum('nij,nj->ni', Tm[:, :d], Qs - Tm[:, d]); lam = np.c_[b, 1 - b.sum(1)]
        lam = np.clip(lam, 0, None); lam /= lam.sum(1, keepdims=True)
        J = self.I[self.tri.simplices[s0]]; return J.astype(np.int64), lam, ok

@nb.njit(parallel=True)
def vit_dl(J, LAM, OK, GL, ING, nP, tau, nit, big, vcap=0.):
    W = np.full(nP, vcap if vcap > 0. else big); L_ = OK.shape[1]; K_ = J.shape[1]
    for i in range(nP):
        if ING[i]: W[i] = 0.
    dd = 0.; it = 0
    for it in range(nit):
        Wn = np.empty(nP)
        for i in nb.prange(nP):
            if ING[i]: Wn[i] = 0.; continue
            best = big
            for a in range(L_):
                if not OK[i, a]: continue
                if GL[i, a] > 0: c = GL[i, a]
                else:
                    r = i * L_ + a; s = 0.; bad = False
                    for k in range(K_):
                        w = W[J[r, k]]
                        if vcap > 0. and w > vcap: w = vcap
                        if w >= big and LAM[r, k] > 1e-12: bad = True; break
                        s += LAM[r, k] * w
                    if bad: continue
                    c = tau[a] + s
                if c < best: best = c
            Wn[i] = best
        dd = np.max(np.abs(Wn - W)); W = Wn
        if dd < 1e-5: break
    return W, it, dd

class AgentDL(T.Agent):
    def __init__(self, S, back):
        T.Agent.__init__(self, S, back); self.Wnn = self.W.copy(); self.kvnn = T.KV
        t0 = time.perf_counter(); self.ip = Interp(S, self.P); E, OK, GL = T.edges(self.P, S); L2 = OK.shape[1]
        J, LAM, ins = self.ip.weights(E.reshape(-1, d)); OKd = OK & ins.reshape(-1, L2) | (OK & (GL > 0)); t1 = time.perf_counter()
        print('  Делоне: %.1f с, рёбер вне оболочки %.4f%%' % (t1 - t0, 100 * (OK & ~ins.reshape(-1, L2) & (GL <= 0)).mean()), flush=True)
        self.W, it, res = vit_dl(J, LAM, OKd, GL, T.in_goal(self.P), self.nP, T.TAUV, NIT, 1e3); self.tdl = time.perf_counter() - t0
        fin = self.W < 1e2; print('  VI-Делоне: %.1f с, it %d res %.1e; W конечных %.1f%%, мед %.2f макс %.2f' % (self.tdl, it, res, 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max()), flush=True)
        T.KV = T.kv_auto(S, self.tree, self.P, self.W); print('  KV_auto(W_dl) = %.3f (было %.3f)' % (T.KV, self.kvnn), flush=True)
        self.mode = 'nn'
    def vh(self, Q):
        if self.mode == 'nn': return T.Agent.vh(self, Q)
        Q = np.asarray(Q, float); J, lam, ok = self.ip.weights(Q); w = (lam * self.W[J]).sum(1)
        bad = ~ok | (self.W[J] >= 1e2).any(1)
        if bad.any(): w[bad] = T.Agent.vh(self, Q[bad])
        return w
    def W_at(self, x0, W, kv):
        dq, jq = self.tree.query(self.S.wrap(x0[None]) / self.S.scale, k=min(T.KN * 3, self.nP)); jq = jq[0] % self.nP
        nn = float((W[jq] + kv * dq[0]).min()); J, lam, ok = self.ip.weights(x0[None]); it = float((lam[0] * W[J[0]]).sum()) if ok[0] else np.nan
        return nn, it

_AG = None
def _ep1(i):
    if not env.keep[i]: return (None, 0)
    Tv, why, m = T.episode(_AG, env.X0[i]); return (Tv, m)

def run_agent(ag, ns):
    Ts = []; ms = []; NPR = int(os.environ.get("NPROC", 1))
    if NPR > 1:   # research-38: старты параллельно, fork после VI (W/дерево общие copy-on-write)
        import multiprocessing as mp
        global _AG; _AG = ag
        with mp.get_context("fork").Pool(NPR) as pool: rr = pool.map(_ep1, range(ns), chunksize=1)
        Ts = [r[0] for r in rr]; ms = [r[1] for r in rr]
    else:
      for i in range(ns):
        if not env.keep[i]: Ts.append(None); ms.append(0); continue
        Tv, why, m = T.episode(ag, env.X0[i]); Ts.append(Tv); ms.append(m)
    Tt = np.array([np.nan if t is None else t for t in Ts]); k = env.keep[:ns]; ok = np.isfinite(Tt) & (Tt <= 3 * env.Tref[:ns]) & k; r = Tt[ok] / env.Tref[:ns][ok]
    return Ts, dict(reach=int(ok.sum()), n=int(k.sum()), med=float(np.median(r)), mean=float(r.mean()), p90=float(np.percentile(r, 90)), max=float(r.max()), ms=1e3 * float(np.mean(ms)))

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts)
    rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
    P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
    back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
    ag = AgentDL(S, back); out = dict(nodes=len(P_), tdl=ag.tdl)
    # агент базовый (W_nn, KV_nn) — контроль, что совпадает с vt_.2_1.log
    Wd, kvd = ag.W, T.KV
    ag.W, T.KV, ag.mode = ag.Wnn, ag.kvnn, 'nn'; Ts, s = run_agent(ag, ns); out['base'] = dict(T=Ts, **s); print('БАЗА nn+KV: %s' % s, flush=True)
    ag.W, T.KV = Wd, kvd
    ag.mode = 'nn'; Ts, s = run_agent(ag, ns); out['dl_nn'] = dict(T=Ts, **s); print('W_dl, V̂ nn+KV_auto: %s' % s, flush=True)
    ag.mode = 'ip'; Ts, s = run_agent(ag, ns); out['dl_ip'] = dict(T=Ts, **s); print('W_dl, V̂ барицентр: %s' % s, flush=True)
    diag = []
    for i in range(ns):
        x0 = env.X0[i]; a, b = ag.W_at(x0, ag.Wnn, ag.kvnn); c, e = ag.W_at(x0, Wd, kvd); tr = float(env.Tref[i])
        diag.append(dict(i=i, Tref=tr, W0nn_nn=a / tr, W0nn_ip=b / tr, W0dl_nn=c / tr, W0dl_ip=e / tr, T_base=out['base']['T'][i], T_ip=out['dl_ip']['T'][i]))
        print('  старт %2d Tref %.2f | W0/Tref: nn-VI %.3f (интерп %.3f) | Делоне-VI %.3f (интерп %.3f) | T/Tref база %s Делоне-ип %s' % (i, tr, a / tr, b / tr, c / tr, e / tr,
              None if out['base']['T'][i] is None else round(out['base']['T'][i] / tr, 3), None if out['dl_ip']['T'][i] is None else round(out['dl_ip']['T'][i] / tr, 3)), flush=True)
    out['diag'] = diag; json.dump(out, open(os.path.expanduser(('~/spore_v5/d3/out/dl_%s_%d_%s_%d' + ('_s100' if os.environ.get('S100')=='1' else '') + '.json') % (T.ENV, T.NB, os.environ.get('TAUS', str(T.TAU)), ts)), 'w'), default=float)

if __name__ == '__main__': main()
