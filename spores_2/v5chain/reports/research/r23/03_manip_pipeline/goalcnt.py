"""research-23: узлы в цели / у цели (goal_dist < .05, .2) в атласах manip — c3000 (GS 300) против d3c3000 (GS 0)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
for p in sys.argv[1:]:
    A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(os.path.expanduser(p), 'rb'))['layers']; A.finish(); d = G.goal_dist(A.P); out = ~A.goal
    print(os.path.basename(p), 'узлов', A.N, '| в цели', int(A.goal.sum()), '| снаружи у цели <.05:', int((out & (d < .05)).sum()), '<.2:', int((out & (d < .2)).sum()), flush=True)
