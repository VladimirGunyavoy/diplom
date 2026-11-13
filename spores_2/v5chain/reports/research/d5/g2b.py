"""d5 Г2б: ДИ 4D, 100k итоговых узлов: ARM=unif (равномерные 100k) vs ARM=adapt (70k равномерных + 2 пульса досева по невязке Беллмана → 85k → 100k), парно по зерну.
Невязка Беллмана узла = |W − min_a c(i,a)| (как g2a; W<VCAP). Досев: M=15k точек, узел выбирается с вероятностью ∝ невязка (только W<VCAP), новая точка = узел + .5·r_nn·N(0,I)·scale, обрезка по коробке и T.okv. Остальное — tvg_lp без изменений.
Запуск: как tvg_lp, из ~/spore_v5/d4/spores_2/w39n: python3 g2b.py di4 100000 8 <зерно>; STARTS=…di4_ref_300_rho35_s7.npy (keep → 253). Пишет out/g2b_di4_<seed>.json (оба рукава)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import g51
from g51 import TL, CAP, T
from tvg_dl import run_agent
def build(S, P):
    back = (P, np.zeros(len(P)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P)))); t0 = time.perf_counter(); ag = TL.AgentLP(S, back); return ag, time.perf_counter() - t0
def resid(ag, n):
    L2 = len(T.TAUV); K = T.d + 1; W = ag.W; J = CAP['J'] % CAP['nP']; Jn = J.reshape(n, L2, K); Ln = CAP['LAM'].reshape(n, L2, K); GL = CAP['GL']; OKd = CAP['OKd']
    c = np.where(OKd, np.where(GL > 0, GL, T.TAUV[None] + (Ln * W[Jn]).sum(-1)), np.inf); r = np.abs(W - c.min(1)); r[~np.isfinite(r) | (W >= float(os.environ.get('VCAP', 40)) - 1e-9)] = 0; return r
def evalag(ag):
    T._MG[0] = T.WMARG; ag.mode = 'ip'; ns = len(T.env.X0); Ts, s = run_agent(ag, ns); return s
def main():
    ts = T.TS; S = T.mkS(ts); d = T.d; NB = T.NB; out = dict(seed=ts, NB=NB); lo = np.where(T.env.period > 0, -T.env.period / 2, T.LO); hi = np.where(T.env.period > 0, T.env.period / 2, T.HI)
    print('ПАСПОРТ g2b: env=%s NB=%d seed=%d KLP=%d' % (T.ENV, NB, ts, TL.KLP), flush=True)
    rng = np.random.default_rng(1000 + ts); P_ = rng.uniform(lo, hi, (2 * NB, d)); P_ = P_[T.okv(P_)]
    # рукав unif
    Pu = P_[:NB]; KV0 = T.KV; ag, tb = build(S, Pu); su = evalag(ag); su['build_s'] = tb; out['unif'] = su; print('unif seed %d: %s' % (ts, su), flush=True); del ag
    # рукав adapt
    T.KV = KV0; n0 = int(.7 * NB); P = P_[:n0].copy(); tt = 0.; npulse = 2; M = (NB - n0) // npulse; rng2 = np.random.default_rng(77 + ts); log = []
    for p in range(npulse + 1):
        ag, tb = build(S, P); tt += tb
        if p == npulse: break
        r = resid(ag, len(P)); pr = r / r.sum(); idx = rng2.choice(len(P), M, p=pr); dd, _ = ag.tree.query(ag.tree.data[idx], k=2); rn = dd[:, 1]
        new = P[idx] + .5 * rn[:, None] * rng2.standard_normal((M, d)) * np.asarray(S.scale, float); new = np.clip(new, lo, hi); new = new[T.okv(new)]
        log.append(dict(pulse=p + 1, n=len(P), added=len(new), top_resid_share=float(np.sort(r)[::-1][:len(P) // 10].sum() / r.sum()))); print(log[-1], flush=True); P = np.vstack([P, new]); del ag
    out['n_final'] = len(P); sa = evalag(ag); sa['build_s'] = tt; out['adapt'] = sa; out['pulses'] = log; print('adapt seed %d (узлов %d): %s' % (ts, len(P), sa), flush=True)
    os.makedirs(os.path.expanduser('~/spore_v5/d4/out'), exist_ok=True); json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/g2b_%s_%d.json' % (T.ENV, ts)), 'w'), default=float)
if __name__ == '__main__': main()
