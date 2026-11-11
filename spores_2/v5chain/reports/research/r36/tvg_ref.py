"""research-36 A3 (слово пользователя 03:05: линейная интерполяция ломается на сепаратрисах, адаптивная сетка — чтобы найти сепаратрису):
Делоне-шов + адаптивное дробление по невязке Беллмана в центроидах симплексов (как у Грюне, adaptive SL-HJB):
e(c) = |interp W(c) − min_a[τ_a + interp W(flow(c,a,τ_a))]|; каждый раунд добавить ADD·n центроидов с наибольшей e, пересобрать Делоне и VI.
Запуск: RNDN=1 KVAUTO=1 RB=.03 TAU=.2 [TAUS=...] NB0=70000 ROUNDS=3 ADD=.1 python3 tvg_ref.py pend 100000 8 SEED"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r35'))
import numpy as np
import tvg_vt as T
from tvg_dl import Interp, vit_dl, run_agent
env, d = T.env, T.d
if os.environ.get("S100") == "1":   # research-36: 100 стартов атласа (grow_cells2d.starts_ref: rng(0)), эталон pend_ref_T.npy как у строки атласа big.md
    _rq = np.random.default_rng(0); env.X0 = np.stack([_rq.uniform(-np.pi, np.pi, 100), _rq.uniform(-2, 2, 100)], 1)
    _rf = np.load("/home/random/spore_v5/r30/spores_2/v7/src/cells7/pend_ref_T.npy"); env.Tref = _rf[:, 0] if _rf.ndim == 2 else _rf; env.keep = np.isfinite(env.Tref) & (env.Tref > .05)
NB0 = int(os.environ.get('NB0', 70000)); ROUNDS = int(os.environ.get('ROUNDS', 3)); ADD = float(os.environ.get('ADD', .1)); NIT = 3000

def solve(S, P):
    ip = Interp(S, P); E, OK, GL = T.edges(P, S); L2 = OK.shape[1]; J, LAM, ins = ip.weights(E.reshape(-1, d)); OKd = (OK & ins.reshape(-1, L2)) | (OK & (GL > 0))
    W, it, res = vit_dl(J, LAM, OKd, GL, T.in_goal(P), len(P), T.TAUV, NIT, 1e3); return W, ip

def residual(S, P, W, ip):
    tri = ip.tri; C = (tri.points[tri.simplices]).mean(1) * S.scale; inb = np.ones(len(C), bool)
    for k in np.nonzero(env.period)[0]: inb &= (C[:, k] >= -env.period[k] / 2) & (C[:, k] < env.period[k] / 2)
    C = C[inb]; Wv = W[ip.I[tri.simplices[inb]]]; good = (Wv < 1e2).all(1) & T.okv(C) & ~T.in_goal(C); C, Wv = C[good], Wv[good]
    wi = Wv.mean(1); E, OK, GL = T.edges(C, S); L2 = OK.shape[1]; J, LAM, ins = ip.weights(E.reshape(-1, d))
    sub = (LAM * W[J]).sum(1).reshape(-1, L2); bad = (~ins | (W[J] >= 1e2).any(1)).reshape(-1, L2)
    cand = np.where(GL > 0, GL, T.TAUV[None] + sub); cand[~OK | ((GL <= 0) & bad)] = 1e9; wb = cand.min(1); m = wb < 1e8
    return C[m], np.abs(wi[m] - wb[m]), wi[m] - wb[m]

class AgentR(T.Agent):
    def __init__(self, S, P, W, ip):
        self.S, self.P, self.nP, self.W, self.ip = S, P, len(P), W, ip; self.tree = T.cKDTree(T._scaled(S, T._images(S, P))); T.KV = T.kv_auto(S, self.tree, P, W)
    def vh(self, Q):
        Q = np.asarray(Q, float); J, lam, ok = self.ip.weights(Q); w = (lam * self.W[J]).sum(1); bad = ~ok | (self.W[J] >= 1e2).any(1)
        if bad.any(): w[bad] = T.Agent.vh(self, Q[bad])
        return w

def wat(ag, x0):
    J, lam, ok = ag.ip.weights(x0[None]); return float((lam[0] * ag.W[J[0]]).sum())

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts); rng = np.random.default_rng(1000 + ts)
    lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
    P = rng.uniform(lo_, hi_, (2 * NB0, d)); P = P[T.okv(P)][:NB0]; out = dict(rounds=[]); t0 = time.perf_counter()
    for r in range(ROUNDS + 1):
        W, ip = solve(S, P); C, e, es = residual(S, P, W, ip)
        if os.environ.get("NOEVAL") == "1" and r < ROUNDS:
            k = int(ADD * len(P)); top = np.argpartition(-e, k)[:k]; P = np.vstack([P, S.wrap(C[top])]); print("раунд %d: %d узлов, t %.1f с" % (r, len(P), time.perf_counter() - t0), flush=True); continue
        ag = AgentR(S, P, W, ip); Ts, s = run_agent(ag, ns); w0 = [wat(ag, env.X0[i]) / env.Tref[i] for i in range(ns)]
        rec = dict(round=r, nodes=len(P), t=time.perf_counter() - t0, e_med=float(np.median(e)), e_p99=float(np.quantile(e, .99)), frac_over=float((es > 0).mean()), W0=w0, T=Ts, **s)
        out['rounds'].append(rec)
        print('раунд %d: узлов %d, t %.1f с, невязка мед %.4f p99 %.4f (доля interp>bellman %.2f) | дошли %d/%d мед %.3f p90 %.3f max %.3f | W0/Tref 5/10/12/6 %s, все %.3f–%.3f' % (
            r, len(P), rec['t'], rec['e_med'], rec['e_p99'], rec['frac_over'], s['reach'], s['n'], s['med'], s['p90'], s['max'], np.round([w0[i] for i in (5, 10, 12, 6)], 3), min(w0), max(w0)), flush=True)
        if r == ROUNDS: break
        k = int(ADD * len(P)); top = np.argpartition(-e, k)[:k]; P = np.vstack([P, S.wrap(C[top])])
    np.savez_compressed(os.path.expanduser("~/spore_v5/r35/refP_%d.npz" % ts), P=P, W=W, n0=NB0) if os.environ.get("DUMP") else None
    json.dump(out, open(os.path.expanduser(('~/spore_v5/r35/ref_%s_%d_%s_%d' + ('_s100' if os.environ.get('S100')=='1' else '') + '.json') % (T.ENV, NB0, os.environ.get('TAUS', str(T.TAU)), ts)), 'w'), default=float)

if __name__ == '__main__': main()
