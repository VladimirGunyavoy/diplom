"""CA4: манип 4D, переходы между целями (старт = предыдущая цель, w = 0). MG(goal) — Manip с центром цели (q1g, q2g, 0, 0), ящик ±(.3,.3,.5,.5) с wrap вокруг него.
A «станции»: 4 позы, 40 случайных переходов между ними; B: 20 новых случайных целей (старт — предыдущая). python manip_goals.py ref A|B [--procs 16] → data/mg_{A,B}_ref.json (эталон OCP IPOPT, вне таймера)."""
import os, sys, json, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
PI = np.pi
class MG(common.Manip):
    name = 'mg'
    def __init__(s, goal, start):
        s.goal = np.array([goal[0], goal[1], 0., 0.], float); s.X0 = np.array([[start[0], start[1], 0., 0.]], float); s.Tref = np.array([np.nan]); s.keep = np.ones(1, bool)
    def in_goal(s, Y): return (np.abs(common.wrap(np.asarray(Y) - s.goal, s.period)) <= s.half).all(-1)
    def goal_dist(s, Y): return (np.abs(common.wrap(np.asarray(Y) - s.goal, s.period)) / s.half).max(-1) - 1.
def poses(n, rng): return np.c_[rng.uniform(-PI, PI, n), rng.uniform(-PI, PI, n)]
def transitions(kind):
    """список (start_xy, goal_xy): A — станции; B — последовательные новые цели"""
    rng = np.random.default_rng(7)
    if kind == 'A':
        st = poses(4, rng); seq = [0]
        while len(seq) < 41: seq.append(int(rng.choice([k for k in range(4) if k != seq[-1]])))
        return [(st[a], st[b]) for a, b in zip(seq[:-1], seq[1:])], st
    g = poses(21, rng); return [(g[k], g[k + 1]) for k in range(20)], g
def work(a):
    kind, k = a; tr, _ = transitions(kind); st, g = tr[k]; e = MG(g, st); import ocp_casadi
    t0 = time.perf_counter(); T, ns = ocp_casadi.solve(e, e.X0[0], 100 + k); return dict(k=k, start=list(map(float, st)), goal=list(map(float, g)), Tref=T, t=time.perf_counter() - t0)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('cmd'); p.add_argument('kind'); p.add_argument('--procs', type=int, default=16); a = p.parse_args()
    if a.cmd == 'ref':
        from multiprocessing import Pool
        n = len(transitions(a.kind)[0])
        with Pool(a.procs) as pool: R = pool.map(work, [(a.kind, k) for k in range(n)])
        json.dump(R, open(os.path.join(common.DATA, f'mg_{a.kind}_ref.json'), 'w'), indent=1, default=float); print(sum(r['Tref'] is not None for r in R), '/', n, 'эталонов')
