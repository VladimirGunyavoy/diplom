"""research-22, эксп. 14: проверка гипотезы «V ниже LB из-за вогнутости T* + линейной интерполяции на крупной клетке» (эксп. 12г).
Прямой замер смещения интерполяции на ТОЧНОЙ функции: для стенсилов одного u (di4 L1600) bias = Σ w·LB(вершины) − LB(точка приземления). Отрицательное среднее × число шагов до цели ≈ наблюдаемое занижение V."""
import sys, os, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G, sgpu
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from tqdm import tqdm
R = .35; G.HexIdx._test = sgpu.test_gpu
def tbox(x0, v0, K=81):
    s_ = np.linspace(-R, R, K); tx = np.r_[s_, s_, np.full(K, -R), np.full(K, R)]; tv = np.r_[np.full(K, -R), np.full(K, R), s_, s_]; out = np.full(len(x0), np.inf)
    for a in tqdm(range(0, len(x0), 20000), desc='tbox', mininterval=10):
        x = x0[a:a + 20000, None]; v = v0[a:a + 20000, None]; best = np.full(x.shape[0], np.inf)
        for s in (1., -1.):
            q = s * (tx - x) + (v * v + tv * tv) / 2; vm = s * np.sqrt(np.maximum(q, 0)); t1 = s * (vm - v); t2 = s * (vm - tv); best = np.minimum(best, np.where((q >= 0) & (t1 >= -1e-12) & (t2 >= -1e-12), t1 + t2, np.inf).min(1))
        out[a:a + 20000] = best
    out[(np.abs(x0) <= R) & (np.abs(v0) <= R)] = 0.; return out
lb = lambda Y: (lambda a, b: (np.maximum(a, b), np.minimum(a, b)))(tbox(Y[:, 0], Y[:, 2]), tbox(Y[:, 1], Y[:, 3]))
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); P = A.P; LBn, _ = lb(P); Y = G.step(P, G.US[0]); a, b, c = A.stencils(Y)
rng = np.random.default_rng(0); k = rng.choice(len(a), 400000, replace=False); a, b, c = a[k], b[k], c[k]; LBy, LBmin = lb(Y[a]); interp = (c * LBn[b]).sum(1); bias = interp - LBy; dom = LBmin / np.maximum(LBy, 1e-9); spread = LBn[b].max(1) - LBn[b].min(1)
print('пар %d | bias = Σw·LB(вершины) − LB(точка): mean %+.4f мед. %+.4f p5 %+.4f p95 %+.4f | доля bias < 0: %.3f | разброс LB по вершинам клетки мед. %.3f (шаг DTN %.2f)' % (len(a), bias.mean(), np.median(bias), *np.percentile(bias, [5, 95]), (bias < 0).mean(), np.median(spread), G.DTN), flush=True)
for lo, hi in ((0, .2), (.2, .5), (.5, .8), (.8, 1.01)):
    m = (dom >= lo) & (dom < hi) & (LBy > .5); print('  min(T1,T2)/LB [%.1f, %.1f): пар %6d | bias mean %+.4f мед. %+.4f p5 %+.4f | × LB/DTN шагов (оценка накопленного): mean %+.3f' % (lo, hi, m.sum(), bias[m].mean(), np.median(bias[m]), np.percentile(bias[m], 5), (bias[m] * LBy[m] / G.DTN).mean()), flush=True)
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 9)):
    m = (LBy >= lo) & (LBy < hi); print('  LB [%d, %d): пар %6d | bias mean %+.4f мед. %+.4f' % (lo, hi, m.sum(), bias[m].mean(), np.median(bias[m])), flush=True)
