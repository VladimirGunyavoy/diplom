import sys, os, pickle, cProfile, pstats
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import grow_cells2d as G
A = pickle.load(open(sys.argv[1], 'rb')); G.LOOK = int(sys.argv[2]); Q, ref = G.starts_ref(); ok = ref > .05; Q = Q[ok][:int(sys.argv[3])]
pr = cProfile.Profile(); pr.enable()
for q in Q: A.rollout(q[None])
pr.disable(); pstats.Stats(pr).sort_stats('cumtime').print_stats(14)
