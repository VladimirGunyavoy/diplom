# research-26: почему агент спор застревает у границы запрещённого (CA5, старты 4,5,7,16; атлас b9 A_forb1)
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.expanduser("~/spore_v5/b9")); import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
R = np.load(os.path.expanduser("~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy")); Q = R[:, :4]
ids = [4, 5, 7, 16]; Q = Q[ids]
A = G.Atlas.__new__(G.Atlas); d = pickle.load(open(os.path.expanduser("~/spore_v5/b9/A_forb1.pkl"), "rb")); A.layers = d["layers"]; A.finish(); A.V = d["V"]
print("DTN", G.DTN, "US", [tuple(np.round(u, 2)) for u in G.US], "WTHR", G.WTHR, "PESS", G.PESS, flush=True)
fin = A.V < G.BIG / 2; Pf = A.P[fin]; print("узлов", A.N, "конечных V", fin.mean().round(3), flush=True)
print("старты: gfree", np.round(G.GF(G.wrapy(Q)), 3).tolist(), "V*(старт)", np.round(A.vstar(Q), 2).tolist(), flush=True)
T, sw, path = A.rollout(Q)
print("T", np.round(T, 2).tolist(), flush=True)
for j, i in enumerate(ids):
    P = path[:, j]; mv = np.flatnonzero(np.abs(np.diff(P, axis=0)).sum(1) > 0); last = (mv[-1] + 1) if len(mv) else 0; Y = P[last][None]
    print("\n== старт %d: шагов сделано %d, конечное y %s, gfree %.3f, V*(y) %.2f, в цели %s" % (i, last, np.round(Y[0], 3).tolist(), G.GF(G.wrapy(Y))[0], A.vstar(Y)[0], bool(G.ingoal(Y)[0])))
    for u in G.US:
        ok = G.edge_ok(Y, u)[0]; yn = G.step(Y, u); vs = A.vstar(yn)[0]; I, IDX, W = A.stencils(yn)
        nst = len(I); kn = [round(float(((W[k] > 1e-6) & (A.V[IDX[k]] < G.BIG / 2)).sum()), 0) for k in range(nst)]
        print("  u=%s edge_ok=%s gfree(y+)=%.3f V*(y+)=%s стенсилов %d (известных вершин по стенсилам %s)" % (tuple(np.round(u, 2)), ok, G.GF(G.wrapy(yn))[0], "BIG" if vs >= G.BIG / 2 else round(float(vs), 2), nst, kn[:6]))
    dd = np.abs(G.wrapy(Pf - Y)).max(1) if hasattr(G, 'wrapy') else np.abs(Pf - Y).max(1); print("  ближайший узел с конечным V: |Δ|∞ = %.3f" % dd.min())
    # траектория V* вдоль пути
    print("  V* вдоль пути:", np.round(A.vstar(P[:last + 1:max(1, last // 8)]), 2).tolist(), " gfree:", np.round(G.GF(G.wrapy(P[:last + 1:max(1, last // 8)])), 3).tolist())
