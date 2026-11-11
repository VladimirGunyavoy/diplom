import sys, os, time, cProfile, pstats
sys.argv = ["car_val.py", "500000", "8", "sel", "20"]
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import car_val as CV
from src.atlas6.adaptive_nd import build_back
import numpy as np
S = CV.mkS(20); build_back(S, .2, 20000, 1.5)   # прогрев numba
S = CV.mkS(20)
if os.environ.get("PROF"):
    pr = cProfile.Profile(); pr.enable()
t0 = time.perf_counter(); back = build_back(S, CV.TAU, int(os.environ.get("NBT", 500000)), CV.RB); t = time.perf_counter() - t0
if os.environ.get("PROF"):
    pr.disable(); pstats.Stats(pr).sort_stats("cumtime").print_stats(18)
G = np.asarray(back[1]); print("TREE build %.1f с, узлов %d, Gb мед %.4f макс %.4f сумма %.3f; TAU %g RB %g" % (t, len(back[0]), np.median(G), G.max(), G.sum(), CV.TAU, CV.RB), flush=True)
np.save(os.environ.get("OUT", "tree_out") + ".npy", np.asarray(back[0]))
