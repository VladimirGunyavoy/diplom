import sys, pickle, numpy as np
sys.path.insert(0, "src/cells7"); import grow3 as G; import __main__; __main__.Cell = G.Cell; __main__.HexIdx = G.HexIdx
d = pickle.load(open(sys.argv[1], "rb")); FV = (2 * G.XL) ** 2 * 2 * np.pi
print("OVERLAP", sys.argv[1], [round(sum(4 * c.r1 * c.r2 * (1 + G.HALO) ** 2 * (c.nf + c.nb) * G.DTN for c in L) / FV, 2) for L in d["layers"]])
