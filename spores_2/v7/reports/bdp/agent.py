"""Агент «мини-дерево на каждом шаге» (research-10, FORCEPLAN=1) на атласе эллипса; из spores_2/v7: WIN=1.0 TR=1 FORCEPLAN=1 NPROC=2 python3 reports/bdp/agent.py <npz>"""
import sys, time, json; sys.path.insert(0, '.')
import numpy as np
import src.cells7.butterfly_dp as M
A = M.load(sys.argv[1]); t0 = time.time(); T, arcs, wm = M.rollout_edges(A, 81, 0.)
print(json.dumps(dict(atlas=sys.argv[1].split('/')[-1], FORCEPLAN=M.FORCEPLAN, T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs, wmax=round(float(wm), 2), plans=getattr(A, 'nplan', None), sec=round(time.time() - t0))), flush=True)
