# research-26: ширина дыры покрытия у границы запрещённого (CA5, атлас b9 A_forb1): доля свободных точек без стенсила / с V*=BIG по корзинам gfree
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.expanduser("~/spore_v5/b9")); import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
A = G.Atlas.__new__(G.Atlas); d = pickle.load(open(os.path.expanduser("~/spore_v5/b9/A_forb1.pkl"), "rb")); A.layers = d["layers"]; A.finish(); A.V = d["V"]
rng = np.random.default_rng(3); Y = np.c_[rng.uniform(-np.pi, np.pi, (60000, 2)), rng.uniform(-1, 1, (60000, 2))]; g = G.GF(G.wrapy(Y)); Y, g = Y[g > 0], g[g > 0]
I, IDX, W = A.stencils(Y); cov = np.zeros(len(Y), bool); cov[I] = True; vs = A.vstar(Y); fin = vs < G.BIG / 2
print("свободных точек", len(Y), "(старты как в bench: q∈[−π,π]², ω∈[−1,1]²)")
for a, b in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 1.2), (1.2, 9)]:
    m = (g >= a) & (g < b); print("gfree [%.2f, %.2f): n %5d  покрыто клетками %.3f  V* конечно %.3f" % (a, b, m.sum(), cov[m].mean(), fin[m].mean()), flush=True)
# какой запрет ближе: самостолкновение (π−FD−|q2|) или стол
q2 = np.abs(G.wrapy(Y)[:, 1]); sc = (np.pi - .5 - q2); tab = g < sc - 1e-9
for nm, mm in [("ближе самостолкновение", ~tab), ("ближе стол", tab)]:
    m = mm & (g < .3); print(nm, "gfree<.3: n %d покрыто %.3f V* конечно %.3f" % (m.sum(), cov[m].mean(), fin[m].mean()))
