"""b26: диагностика хвоста TIGHT з.9 ст.28 (11.5 с). Запуск как car_dyn_astar: runpc.sh diag_tight.py 500000 8 test 0 9 10 (ENV TIGHT=1 HYB=0|1 STARTS=…tight COMPACT=…tight_compact; DIAGI=28). Пишет diag_<HYB>.json (по шагам: t, x, a, W0x, Wx, Wq по действиям, ok, мин. зазор до дисков по прогнозу, ближайший диск, tie/esc/A*-флаг)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_dyn_astar as da
cd = da.cd; np = da.np; time = da.time
from car_dyn6 import ACT, TAU, execute_dyn, HR, GOAL
import envs_dyn as D
I = int(os.environ.get('DIAGI', 28)); K = cd.K0; HYBF = da.HYB
S0 = np.load(os.environ['STARTS'])[:, :4]; COMP = json.load(open(os.environ['COMPACT'])); ref = {r['i']: r for r in COMP[str(K)]['ref']}
S = cd.mkS(K); back = cd.build_back(S, TAU, cd.NB, cd.RB); ag = da.AgentDynA(S, back); H = D.Hazards('random', K)
x = np.array(S0[I], float); t = 0.; nsub = int(round(TAU / .002)); Pp = H.pos(0.); ag.plan_acts = []; ag.streak = 0; ag.hp = []; ag.prev = None; log = []; nesc0 = 0
while t < 60.:
    P0 = H.pos(t); Wv = (P0 - Pp) / TAU if t > 0 else np.zeros((3, 2)); Pp = P0
    ne = getattr(ag, 'nesc', 0); nt = getattr(ag, 'ntie', 0)
    s = ag.act(x, P0, Wv); esc = getattr(ag, 'nesc', 0) > ne; tie = getattr(ag, 'ntie', 0) > nt
    dd = np.hypot(*(P0 - x[:2]).T) - HR
    log.append(dict(t=round(t, 2), x=[float(v) for v in x], a=int(s), W0x=float(ag.W0x), Wx=float(ag.Wx), Wq=[float(v) for v in np.minimum(getattr(ag, 'Wq', np.zeros(9)), 99)], okc=[int(v) for v in getattr(ag, 'okc', np.zeros(9))], cl=[float(v) for v in getattr(ag, 'cl', np.zeros(9))], disc_clr=[float(v) for v in dd], disc=[[float(a) for a in p] for p in P0], vel=[[float(a) for a in w] for w in Wv], esc=bool(esc), tie=bool(tie), nflag=int(ag.nflag), rest=bool(abs(x[3]) < .05)))
    PT = H.pos(t + (np.arange(nsub) + 1) * .002); xn, yn, thn, vn, m, hit = execute_dyn(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], .002, nsub, PT); x = np.array([xn, yn, thn, vn])
    if hit >= 0: t += hit; break
    t += TAU
print('HYB=%d: T=%.2f T_ref=%.3f ratio=%.2f шагов %d, tie-шагов %d, esc-шагов %d' % (HYBF, t, ref[I]['T'], t / ref[I]['T'], len(log), sum(l['tie'] for l in log), sum(l['esc'] for l in log)), flush=True)
json.dump(dict(T=t, Tref=ref[I]['T'], log=log, start=S0[I].tolist(), goal=GOAL.tolist()), open('diag_%d.json' % HYBF, 'w'))
