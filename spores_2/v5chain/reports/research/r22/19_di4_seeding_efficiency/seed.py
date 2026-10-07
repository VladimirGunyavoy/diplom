"""research-22, эксп. 19: эффективность посева di4 — покрытие слоя и кратность наложения в зависимости от числа клеток (клетки слоя u0 из Lfull в порядке создания).
Проба: 200k случайных точек поля [−2.5, 2.5]⁴. Покрытие — ядром (HexIdx.covered) и с гало (любая гиперячейка); кратность — гиперячеек на покрытую точку."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G, sgpu
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from tqdm import tqdm
G.HexIdx._test = sgpu.test_gpu; d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); L = d_['layers'][0]; rng = np.random.default_rng(0); Y = rng.uniform(-2.5, 2.5, (200000, 4)); idx = G.HexIdx(); marks = [100, 200, 400, 800, 1200, 1600, 2400, 3200, 4800, len(L)]; prev = (0, 0.); vol = 0.
for i, c in enumerate(tqdm(L, desc='клетки слоя u0', mininterval=10)):
    idx.add(c)
    if i + 1 in marks:
        cov = idx.covered(Y).mean(); pi, hid, sc = idx.query(Y); anyc = len(np.unique(pi)) / len(Y); mult = len(pi) / max(len(np.unique(pi)), 1); nodes = sum(cc.G.reshape(-1, 4).shape[0] for cc in L[:i + 1])
        print('клеток %5d | узлов %8d | покрытие ядром %.3f, с гало %.3f | кратность (гиперячеек на покрытую точку) %.2f | прирост покрытия на 100 клеток с прошлой отметки: %.4f' % (i + 1, nodes, cov, anyc, mult, 100 * (anyc - prev[1]) / (i + 1 - prev[0])), flush=True); prev = (i + 1, anyc)
