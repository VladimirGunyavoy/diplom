import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.expanduser("~/spore_v5/r17/v7/src/cells7")); import grow3 as G; import __main__; __main__.Cell = G.Cell; __main__.HexIdx = G.HexIdx
d = pickle.load(open(sys.argv[1], "rb")); A = G.Atlas.__new__(G.Atlas); A.layers = d["layers"]; A.finish(); A.V = d["V"]
Q, T, ref = d["Q"], d["T"], d["ref"]; bad = np.flatnonzero(~np.isfinite(T)); print("fail", bad.tolist())
Tb, sw, P = A.rollout(Q[bad])
for j, i in enumerate(bad):
    p = P[:, j]; last = p[-1]; mv = np.linalg.norm(np.diff(p[-30:], axis=0), axis=1).sum()
    Y = last[None]; vs = [float(A.vstar(G.step(Y, u))[0]) for u in G.US]; v0 = float(A.vstar(Y)[0])
    n_moving = int((np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-12).sum())
    print(i, "ref %.2f" % ref[i], "steps_moved", n_moving, "last", np.round(last, 3).tolist(), "|last|", round(float(np.linalg.norm(np.r_[last[:2], G.wrap(last[2])])), 3), "V*", round(v0, 3), "V*(step u)", [round(x, 2) for x in vs], "pathlen_last30 %.3f" % mv, "sw", int(sw[j]))
