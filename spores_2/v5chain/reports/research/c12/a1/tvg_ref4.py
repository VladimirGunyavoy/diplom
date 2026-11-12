"""research-37 (задача senior-2 12.11): дробление по невязке Беллмана в nD без Делоне.
Невязка в случайных пробных точках c: e(c) = |Σλ W(c) − min_a[τ_a + Σλ W(flow(c, a, τ_a))]| (LP-веса по KLP соседям, VCAP).
Раунд: M = PROBE·n пробных точек, добавить ADD·n с наибольшей e как узлы, пересобрать LP-веса и VI. Парный контроль в том же прогоне:
равномерные случайные узлы того же итогового числа (другое зерно), тот же агент (LP-интерп, переменный τ).
Запуск: KVAUTO=1 KV=.05 DEP=2 DEPPEN=1 TAU=.3 RB=.35 KLP=20 VCAP=40 TAUS=.075,.15,.3 STARTS=... NFIN=30000 ROUNDS=5 ADD=.1 python3 tvg_ref4.py di4 30000 8 SEED"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
import tvg_vt as T
from tvg_lp import AgentLP
from tvg_dl import run_agent
env, d = T.env, T.d
NFIN = int(os.environ.get('NFIN', 30000)); ROUNDS = int(os.environ.get('ROUNDS', 5)); ADD = float(os.environ.get('ADD', .1)); PROBE = float(os.environ.get('PROBE', 1.))
VCAP = float(os.environ.get('VCAP', 40))

def box(rng, n):
    lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
    P = rng.uniform(lo_, hi_, (3 * n, d)); P = P[T.okv(P)][:n]; assert len(P) == n; return P

def mkag(S, P):
    back = (P, np.zeros(len(P)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P))))
    ag = AgentLP(S, back); ag.mode = 'ip'; return ag

def residual(S, ag, C):
    Wc = np.minimum(ag.W, VCAP)
    J, lam, ok = ag.ip.weights(C); wi = (lam * Wc[J]).sum(1)
    good = ok & (wi < VCAP - 1) & ~T.in_goal(C); C, wi = C[good], wi[good]
    E, OK, GL = T.edges(C, S); L2 = OK.shape[1]; J, lam, ins = ag.ip.weights(E.reshape(-1, d))
    sub = (lam * Wc[J]).sum(1).reshape(-1, L2); bad = ~ins.reshape(-1, L2)
    cand = np.where(GL > 0, GL, T.TAUV[None] + sub); cand[~OK | ((GL <= 0) & bad)] = 1e9; wb = cand.min(1); m = wb < VCAP - 1
    return C[m], np.abs(wi[m] - wb[m]), wi[m] - wb[m]

def evalag(ag, ns):
    Ts, s = run_agent(ag, ns); return Ts, s

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts); rng = np.random.default_rng(1000 + ts)
    n0 = int(round(NFIN / (1 + ADD) ** ROUNDS)); P = box(rng, n0); t0 = time.perf_counter(); log = []
    for r in range(ROUNDS + 1):
        ag = mkag(S, P)
        if r == ROUNDS: break
        k = int(round(n0 * (1 + ADD) ** (r + 1))) - len(P) if r < ROUNDS - 1 else NFIN - len(P)
        C = box(rng, int(PROBE * len(P)) + k); C, e, es = residual(S, ag, C); top = np.argpartition(-e, k)[:k]
        log.append(dict(round=r, nodes=len(P), e_med=float(np.median(e)), e_p99=float(np.quantile(e, .99)), e_sel_min=float(e[top].min()), frac_over=float((es > 0).mean())))
        print('раунд %d: узлов %d → +%d, t %.1f с, невязка мед %.4f p99 %.4f, порог отбора %.4f, доля interp>bellman %.2f' % (
            r, len(P), k, time.perf_counter() - t0, log[-1]['e_med'], log[-1]['e_p99'], log[-1]['e_sel_min'], log[-1]['frac_over']), flush=True)
        P = np.vstack([P, C[top]])
    tref = time.perf_counter() - t0; Ts, s = evalag(ag, ns)
    print('ДРОБЛЕНИЕ: узлов %d (старт %d), сборка %.1f с | %s' % (len(P), n0, tref, s), flush=True)
    out = dict(nodes=len(P), n0=n0, t_build=tref, rounds=log, ref=dict(T=Ts, **s))
    if os.environ.get('DUMP'): np.savez_compressed(os.path.expanduser('~/spore_v5/r37/ref4P_%d_%d.npz' % (NFIN, ts)), P=P, W=ag.W, n0=n0)
    t1 = time.perf_counter(); agu = mkag(S, box(np.random.default_rng(2000 + ts), len(P))); tu = time.perf_counter() - t1; Tu, su = evalag(agu, ns)
    print('РАВНОМЕРНЫЕ: узлов %d, сборка %.1f с | %s' % (len(P), tu, su), flush=True)
    out['uni'] = dict(T=Tu, t_build=tu, **su)
    json.dump(out, open(os.path.expanduser('~/spore_v5/r37/ref4_%s_%d_%s_%d%s.json' % (T.ENV, NFIN, os.environ.get('TAUS', str(T.TAU)), ts, os.environ.get('TAG', ''))), 'w'), default=float)

if __name__ == '__main__': main()
