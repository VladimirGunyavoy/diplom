"""Г4.2 рука (б): LP-граф (tvg_lp, c13/g42-копия с regoal) — карта и LP-веса ОДИН раз без цели, на каждую цель: попадания дуг в цель + VI (regoal) + агент.
Контроль (правило 1): цель в нуле на 253 свежих стартах r37 (ожидание: 100k граф ДИ4D 1.020–1.023 / 1.065–1.076 / 1.51–1.58)."""
import os, sys, time, json, hashlib
for k, v in dict(NUMBA_THREADING_LAYER='workqueue', OMP_NUM_THREADS='1', NUMBA_NUM_THREADS=os.environ.get('NT', '8'), KDW=os.environ.get('NT', '8'), NPROC=os.environ.get('NPROC', '4'), RNDN='1', KLP='20', VCAP='40', KVAUTO='1', KV='.05', DEP='2', DEPPEN='1',
                 TAUS='.075,.15,.3', TAU='.15', RB='.03', SKIPBASE='1', STARTS=os.path.expanduser('~/spore_v5/r37/di4_ref_300_rho35_s7.npy')).items(): os.environ.setdefault(k, v)
NB = int(os.environ.get('NBN', 100000)); sys.argv = ['tvg_lp.py', 'di4', str(NB), '8', '0']
import numpy as np, tvg_vt as T, tvg_lp as L
from tvg_dl import run_agent
import common, g42_common
env = T.env; d = env.n; ts = T.TS; S = T.mkS(ts)
print('HEADER: рука (б) LP-граф ДИ4D NB=%d KLP=20 TAUS=.075,.15,.3 TAU=.15 RB=.03 DEP=2 DEPPEN=1 VCAP=40; контроль-старты %s md5=%s; проверяльщик common.verify h .01 строгий; окно цели ±.35 (в графе ±.33); коробка verify %s/%s; код tvg_lp md5=%s' % (NB, os.environ['STARTS'], hashlib.md5(open(os.environ['STARTS'], 'rb').read()).hexdigest()[:10], env.lo.tolist(), env.hi.tolist(), hashlib.md5(open('tvg_lp.py', 'rb').read()).hexdigest()[:10]), flush=True)
t0 = time.perf_counter(); rng_ = np.random.default_rng(1000 + ts); lo_, hi_ = T.LO, T.HI; P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_)))); ag = L.AgentLP(S, back); t_build = time.perf_counter() - t0
print('ПОДГОТОВКА (карта без цели: узлы+рёбра+LP-веса + одно VI для цели в нуле): %.1f с, узлов %d' % (t_build, len(back[0])), flush=True)
ag.mode = 'ip'; Ts, s = run_agent(ag, len(env.X0)); print('КОНТРОЛЬ цель в нуле, %d стартов: %s' % (len(env.X0), s), flush=True)
rows = []; allr = []
for gi, (C, Q, ref, keep) in enumerate(g42_common.make()):
    inG = lambda P, C=C: (np.abs(np.asarray(P) - C) <= env.half - .02).all(-1); T.in_goal = inG; env.in_goal = lambda Y, C=C: (np.abs(np.asarray(Y) - C) < env.half).all(-1)
    env.X0, env.Tref, env.keep = Q, ref, keep; tg, it = ag.regoal(inG); t1 = time.perf_counter(); Ts, s = run_agent(ag, len(Q)); tq = time.perf_counter() - t1
    s.update(goal=C[:2].tolist(), t_goal=tg, vi_it=it, t_agent=tq, n_keep=int(keep.sum())); rows.append(s); allr.append(([None if t is None else float(t) for t in Ts], ref.tolist(), keep.tolist())); print('цель %d %s: на цель %.1f с (дуги+VI), агент %.1f с: %s' % (gi, np.round(C[:2], 2).tolist(), tg, tq, s), flush=True)
reach = sum(r['reach'] for r in rows); nk = sum(r['n'] for r in rows); R = np.array([t / r for (Tl, rl, kl) in allr for t, r, k in zip(Tl, rl, kl) if k and t is not None and t <= 3 * r])
print('ИТОГ (б): дошли %d/%d, T/T_ref мед %.3f p90 %.3f max %.3f; нарушения box %d nogoal %d slow %d; подготовка %.1f с (один раз), на цель мед %.1f с, запрос мед %.1f мс/шаг' % (reach, nk, np.median(R), np.percentile(R, 90), R.max(), sum(r['viol_box'] for r in rows), sum(r['viol_nogoal'] for r in rows), sum(r['viol_slow'] for r in rows), t_build, np.median([r['t_goal'] for r in rows]), np.mean([r['ms'] for r in rows])), flush=True)
o = os.path.expanduser('~/spore_v5/c13/out_g42'); os.makedirs(o, exist_ok=True); json.dump(dict(t_build=t_build, rows=rows, per=allr), open(o + '/arm_b.json', 'w'), default=float)
