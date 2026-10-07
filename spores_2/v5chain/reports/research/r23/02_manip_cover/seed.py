"""research-23: посев слоя manip u0 (снимок growN wb5) — покрытие ядром на 100k случайных точках поля (|w| ≤ 3; и |w| ≤ 1) против числа клеток, время.
Варианты задаются окружением (GS, GLIM, DELTA, RMAX, TMAX, SIDE); прочее как manip c3000 b4 (KF 11 M 3 FRAC .5 OVH .5 MINROWS 3)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
MARKS = [250, 500, 1000, 1500, 2000, 3000, 4000]; r2 = np.random.default_rng(5); Y = np.c_[r2.uniform(-np.pi, np.pi, (100000, 2)), r2.uniform(-3, 3, (100000, 2))]; Y1 = Y[np.abs(Y[:, 2:]).max(1) <= 1]
rng = np.random.default_rng(0)
for _ in range(32): rng.uniform(-1, 1, 4)
Q = np.array([[*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)] for _ in range(20)]); t0 = time.time(); tag = os.environ.get('TAG', '')
def log(u, cells):
    if len(cells) not in MARKS: return
    idx = G.HexIdx()
    for c in cells: idx.add(c)
    vol = sum(8 * np.prod(c.r) * (c.nb + c.nf) * G.DTN * np.linalg.norm(G.f(c.c, c.u)) for c in cells)
    print('[%s] клеток %5d | %5.0f с | покрытие поля %.3f, |w|≤1 %.3f | кратность ~%.2f | старты в ядре %d/20 | узлов %d' % (tag, len(cells), time.time() - t0, idx.covered(Y).mean(), idx.covered(Y1).mean(),
          vol / max(idx.covered(Y).mean() * 4 * np.pi ** 2 * 36, 1e-9), idx.covered(Q).sum(), sum(c.G.reshape(-1, 4).shape[0] for c in cells)), flush=True)
cells, _ = G.build_layer(G.US[0], np.random.default_rng(1), log); print('[%s] конец: клеток %d, %.0f с' % (tag, len(cells), time.time() - t0), flush=True)
