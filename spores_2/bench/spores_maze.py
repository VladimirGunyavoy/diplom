"""CA-MAZE: споры growN SYS=dd в лабиринте: GF = gfree лабиринта (подмена G.GF из бенча, v7 не правим). Запуск (aida, bench/): MAXC=3000 python3 spores_maze.py  (слои → solve → 50 запросов, как spores_manip full)"""
import os, sys, time, json, pickle
ENV = dict(SYS='dd', XL='6', RMAX='.3', TMAX='3', RHO='.05', KF='21', M='3', FRAC='.5', OVH='.5', MINROWS='3', GLIM='0', NOLATCH='1', FINGRID='2', VF='2', PESS='1', GS='50', DELTA='.3', MAXC='3000', TQDM_MI='1000', OMP_NUM_THREADS='2', FORB2='.5')
for k, v in ENV.items(): os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.expanduser(os.environ.get('SRC_DIR', '~/spore_v5/c1/manip_src')); sys.path.insert(0, HERE); sys.path.insert(0, SRC)
import numpy as np, common, maze
import growN as G, finish_gen as FG
G.GF = lambda y: M.gfree(y)
M = maze.Maze(); G.GC = G.GC if hasattr(G, 'GC') else None
import spores_manip as SM          # rollout_rec и finish; в нём growN подхватится из того же sys.path (тот же модуль)
assert SM.G is G
if __name__ == '__main__':
    e = common.make('maze'); t0 = time.perf_counter()
    if os.environ.get('LOADL'): d_ = pickle.load(open(os.path.expanduser(os.environ['LOADL']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); t_grow = float(os.environ.get('GROWT', 0))   # слои с диска (GROWT — время роста из лога)
    else: A = G.Atlas(); t_grow = time.perf_counter() - t0
    print('слои', [len(l) for l in A.layers], 'узлов', A.N, round(t_grow), 'с', flush=True)
    if os.environ.get('DUMPL'): pickle.dump(dict(layers=A.layers), open(os.path.expanduser(os.environ['DUMPL']), 'wb'))
    t0 = time.perf_counter(); A.solve(); t_solve = time.perf_counter() - t0; print('solve', round(t_solve), 'с', flush=True)
    res = []; Ts = []; tq = []
    for i in range(len(e.X0)):
        t1 = time.perf_counter(); T, seg, nodur = SM.rollout_rec(A, e.X0[i:i + 1]); dtq = time.perf_counter() - t1; tq.append(dtq); sg = seg[0]
        Tv, why = common.verify(e, e.X0[i], [list(u) for u, d in sg], [d for u, d in sg]); Ts.append(Tv)
        res.append(dict(i=i, T_agent=float(T[0]) if np.isfinite(T[0]) else None, T_verify=Tv, why=why, t_query=dtq, nseg=len(sg))); print(i, res[-1], flush=True)
    s = common.summarize(e, Ts); s.update(env='maze', method='spores growN dd + gfree лабиринта (full%s)' % os.environ.get('TAG', ''), t_grow=t_grow, t_solve=t_solve, t_build=t_grow + t_solve, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), t_query_sum=float(np.sum(tq)),
                                      cells=[len(l) for l in A.layers], nodes=int(A.N), hw='CPU', per_start=res, Tref=e.Tref.tolist()); s['weight'] = common.deep_weight(A)
    json.dump(s, open(os.path.join(common.DATA, 'maze_spores%s.json' % os.environ.get('TAG', '')), 'w'), indent=1, default=float); print({k: v for k, v in s.items() if k not in ('per_start', 'Tref')})
