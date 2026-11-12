"""research-37 (graph_to_cells.md §Поправка senior-2, шаг 2 — описательно, без агента): «точки → клетки» на готовом графе.
Рост области по kNN-графу от затравки: фронт = неприсвоенные соседи области; принимаются узлы фронта, у которых квадратичная модель
W ≈ q(δx) (δx = wrap(x − x_seed)/scale) даёт |W − q| ≤ EPS·max(W, WFLOOR); затем переподгонка на всей области и проверка той же
границы на всех её узлах (иначе слой не принимается, рост стоп). Пока узлов < NFEAT+2 — модель линейная. Затравка — неприсвоенный узел
с наименьшей W (рост «от цели»). Узлы у потолка (W ≥ VCAP−1) и в цели — вне счёта (доля печатается).
Числа: областей на 95% узлов, размеры, хранимых чисел (области × (d + NFEAT)) против узлов × (d + 1), доля границ на полосах невязки
(для маятника refP: узлы с индексом ≥ n0 — добавлены дроблением, т.е. полосы невязки).
Запуск: SRC=~/spore_v5/r35/refP_0.npz EPS=.01 python3 graph_cells.py pend 0 8 0 | ДИ4D: (SRC нет → строит случайные NB + LP-VI, DUMP) ... di4 100000 8 0"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
import tvg_vt as T
env, d = T.env, T.d
EPS = float(os.environ.get('EPS', .01)); WFLOOR = float(os.environ.get('WFLOOR', .5)); VCAP = float(os.environ.get('VCAP', 40)); KG = int(os.environ.get('KG', 2 * d + 4))
NFEAT = 1 + d + d * (d + 1) // 2; MAXR = int(os.environ.get('MAXR', 200000))

def feats(D, quad):
    cols = [np.ones(len(D))] + [D[:, i] for i in range(d)]
    if quad: cols += [D[:, i] * D[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, 1)

def fit(D, w):
    quad = len(D) >= NFEAT + 2; A = feats(D, quad); c, *_ = np.linalg.lstsq(A, w, rcond=None); return c, quad

def pred(c, quad, D): return feats(D, quad) @ c

def main():
    ts = T.TS; S = T.mkS(ts); src = os.environ.get('SRC')
    if src:
        z = np.load(os.path.expanduser(src)); P, W = z['P'], z['W']; n0 = int(z['n0']) if 'n0' in z else len(P)
    else:
        from tvg_ref4 import mkag, box
        P = box(np.random.default_rng(1000 + ts), T.NB); ag = mkag(S, P); W = ag.W.copy(); n0 = len(P)
        np.savez_compressed(os.path.expanduser('~/spore_v5/r37/graph_%s_%d_%d.npz' % (T.ENV, len(P), ts)), P=P, W=W, n0=n0)
    n = len(P); Xs = T._scaled(S, T._images(S, P)); tree = T.cKDTree(Xs); _, nb = tree.query(S.wrap(P) / S.scale, k=KG + 1); nb = nb[:, 1:] % n
    act = (W < VCAP - 1) & ~T.in_goal(P); lab = np.full(n, -1); lab[~act] = -2
    order = np.argsort(np.where(act, W, np.inf)); t0 = time.perf_counter(); sizes = []; reg = 0; ptr = 0
    while True:
        while ptr < n and lab[order[ptr]] != -1: ptr += 1
        if ptr >= n or not act[order[ptr]]: break
        s = order[ptr]; R = np.array([s]); lab[s] = reg; x0 = P[s]
        while len(R) < MAXR:
            F = np.unique(nb[R].ravel()); F = F[lab[F] == -1]
            if len(F) == 0: break
            DR = S.wrap(P[R] - x0) / S.scale; DF = S.wrap(P[F] - x0) / S.scale
            if len(R) == 1:   # первый шаг: линейная подгонка по затравке и её соседям (постоянная модель не растёт при |∇W|·h > ε·W)
                if len(F) < d + 1: break
                Dl = np.vstack([np.zeros((1, d)), DF]); wl = np.concatenate([[W[s]], W[F]]); cl = np.linalg.lstsq(feats(Dl, False), wl, rcond=None)[0]
                acc = F[np.abs(feats(DF, False) @ cl - W[F]) <= EPS * np.maximum(W[F], WFLOOR)]
                if len(acc) < d + 1: break
            else:
                c, q = fit(DR, W[R]); acc = F[np.abs(pred(c, q, DF) - W[F]) <= EPS * np.maximum(W[F], WFLOOR)]
            if len(acc) == 0: break
            R2 = np.concatenate([R, acc]); c, q = fit(S.wrap(P[R2] - x0) / S.scale, W[R2])
            if (np.abs(pred(c, q, S.wrap(P[R2] - x0) / S.scale) - W[R2]) > EPS * np.maximum(W[R2], WFLOOR)).any(): break
            lab[acc] = reg; R = R2
        sizes.append(len(R)); reg += 1
        if reg % 2000 == 0: print('  областей %d, присвоено %.1f%%, t %.0f с' % (reg, 100 * (lab >= 0).sum() / act.sum(), time.perf_counter() - t0), flush=True)
    sz = np.sort(np.array(sizes))[::-1]; cum = np.cumsum(sz) / act.sum(); k95 = int(np.searchsorted(cum, .95) + 1)
    bnd = act & (lab >= 0) & (lab[nb] != lab[:, None]).any(1); refd = np.arange(n) >= n0
    stored_cells = k95 * (d + NFEAT); stored_nodes = int(.95 * act.sum()) * (d + 1)
    res = dict(env=T.ENV, nodes=n, active=int(act.sum()), eps=EPS, regions=reg, k95=k95, size_med=float(np.median(sz)), size_max=int(sz[0]), singletons=int((sz == 1).sum()),
               stored_cells=stored_cells, stored_nodes=stored_nodes, compress=stored_nodes / stored_cells, frac_boundary=float(bnd.sum() / act.sum()),
               refined_frac=float(refd[act].mean()), refined_frac_on_boundary=float(refd[bnd].mean()) if bnd.any() else None, t=time.perf_counter() - t0)
    print('ИТОГ %s' % res, flush=True)
    json.dump(res, open(os.path.expanduser('~/spore_v5/r37/gc_%s_%d_%g_%d%s.json' % (T.ENV, n, EPS, ts, os.environ.get('TAG', ''))), 'w'), default=float)
    np.save(os.path.expanduser('~/spore_v5/r37/gc_lab_%s_%d_%g_%d.npy' % (T.ENV, n, EPS, ts)), lab)

if __name__ == '__main__': main()
