"""В3 v2 (слово research-30): грубый атлас (2 слоя) → поле u*(узлы) → единый слой В3 со СТОПОМ роста по смене u* на фронте (центр фронта) + ПОЛОСА клеток обоих режимов у поверхности переключения
(посев из узлов, где u* отличается от u* соседей в радиусе BAND; клетки каждого u в полосе, рост — только внутри полосы) → итерация (u* из полученного атласа) B3IT раз. Мерило: узлы и p90 сид-к-сиду против грубого,
плюс доля строк клеток (m ≥ 9) с полиномом степени 2 по s (допуск 1e-3, V и положение узла) — как 2.1% обычных клеток (03_dens/affine_work.py).
Env: DTN MCAP MADAPT BSEED BIGOUT (как sweep_dens), B3IT (итераций, 2), BAND (радиус полосы, .15), STOP (1/0), TAG."""
import os, sys, json, time, inspect; sys.path.insert(0, os.path.join(os.getcwd(), '..', 'v7', 'reports', 'big'))
import big_lib as B, numpy as np
from scipy.spatial import cKDTree
G = B.G; G.US = (-G.UM, G.UM); G.STOPU = None; UA = np.array(G.US)
src = inspect.getsource(G.build_layer)
s1 = src.replace('def build_layer(u, rng, log=None):', 'def build_b3(ufun, rng, log=None):', 1).replace("        p = np.array([wrap(p[0]), p[1]])\n", "        p = np.array([wrap(p[0]), p[1]]); u = float(ufun(p))\n", 1).replace("bar = tqdm(desc='layer u=%+g cells' % u,", "bar = tqdm(desc='B3 single layer cells',", 1)
key = "L += float(np.linalg.norm(sl[M // 2] - prev))\n"; assert key in s1
s1 = s1.replace(key, key + "                    if STOPU is not None and float(STOPU(sl[M // 2])) != u: break\n", 1)
s2 = src.replace('def build_layer(u, rng, log=None):', 'def build_band(u, pts, bt, rho, log=None):', 1)
old = "        p = queue.pop(0) if queue else np.array([rng.uniform(-XL, XL), rng.uniform(-WL, WL)])\n"; assert old in s2
s2 = s2.replace(old, "        p = queue.pop(0) if queue else next(it_, None)\n        if p is None: break\n", 1).replace("        p = np.array([wrap(p[0]), p[1]])\n", "        p = np.array([wrap(p[0]), p[1]])\n        if bt.query(p)[0] > rho: continue\n", 1).replace("seeds = []; sc0", "seeds = []; it_ = iter(pts); sc0", 1).replace("bar = tqdm(desc='layer u=%+g cells' % u,", "bar = tqdm(desc='band u=%+g cells' % u,", 1)
assert 'it_ = iter(pts)' in s2 and 'bt.query' in s2; exec(s1, G.__dict__); exec(s2, G.__dict__)
def tile(P): return np.concatenate([P + np.array([sh, 0.]) for sh in (-2 * np.pi, 0., 2 * np.pi)]), np.tile(np.arange(len(P)), 3)
def ustar_nodes(A):
    P = A.P; n = len(P); tg = A.tgoal3(P); Ys = A.step3(P); J = np.minimum(tg, G.DTN + A.vstar(Ys).reshape(len(G.US), n).T); return J.argmin(1)
def make_fields(A, rho):
    P = A.P.copy(); P[:, 0] = G.wrap(P[:, 0]); k = ustar_nodes(A); PT, ix = tile(P); kt = cKDTree(PT); kk = np.tile(k, 3)
    ufield = lambda p: UA[kk[kt.query(np.array([G.wrap(p[0]), p[1]]))[1]]]
    nb = kt.query_ball_point(P, rho); band = np.array([len(set(kk[j])) > 1 for j in nb]); Pb = P[band]
    return ufield, Pb, k
def poly2(A, tol=1e-3):
    ok_ = tot = 0
    for c in A.cells:
        nt, m = c.G.shape[:2]
        if m < 9: continue
        V = A.V[c.o:c.o + nt * m].reshape(nt, m); s_ = np.linspace(-1, 1, m); Va = np.vander(s_, 3); Vm = np.where(V < G.BIG / 2, V, 0.)
        def res(Z): return np.abs(Va @ np.linalg.lstsq(Va, Z.T, rcond=None)[0] - Z.T).max(0)
        okr = ((res(Vm) <= tol) | (V.max(1) >= G.BIG / 2)) & (np.maximum(res(c.G[..., 0]), res(c.G[..., 1])) <= tol); ok_ += int(okr.sum()); tot += nt
    return dict(rows_poly2=ok_ / max(tot, 1), rows_m_ge9=tot)
seed = int(os.environ.get('BSEED', 0)); RHO = float(os.environ.get('BAND', .15)); NIT = int(os.environ.get('B3IT', 2)); STOP = int(os.environ.get('STOP', 1))
tag = os.environ.get('TAG', 'b3v2_s%d' % seed); t0 = time.perf_counter(); A0 = G.Atlas(seed=seed); tg0 = time.perf_counter() - t0; ts0 = B.solve(A0); s0, _ = B.eval100(A0)
co = dict(cells=len(A0.cells), nodes=int(A0.N), t_grow=tg0, t_solve=ts0, **s0, **poly2(A0)); print('грубый:', co, flush=True); Ap = A0; R_ = []
for it in range(NIT):
    t1 = time.perf_counter(); uf, Pb, kst = make_fields(Ap, RHO); G.STOPU = uf if STOP else None; rng = np.random.default_rng(seed + 100 * (it + 1)); cells, idx = G.build_b3(uf, rng); G.STOPU = None
    sh = np.random.default_rng(seed + 7).permutation(len(Pb)); Pb = Pb[sh]; PbT, _ = tile(Pb); bt = cKDTree(PbT); layers = [cells]
    for u in G.US: layers.append(G.build_band(u, Pb, bt, RHO)[0])
    tg1 = time.perf_counter() - t1; A1 = G.Atlas.__new__(G.Atlas); A1.layers = layers; A1.idx = []; A1.finish(); ts1 = B.solve(A1); s1, _ = B.eval100(A1)
    R_.append(dict(iter=it + 1, band_nodes=len(Pb), cells_main=len(cells), cells_band=[len(l) for l in layers[1:]], cells=len(A1.cells), nodes=int(A1.N), t_grow=tg1, t_solve=ts1, goalh_used=getattr(A1, 'goalh_used', False), V_fin=float((A1.V < G.BIG / 2).mean()), **s1, **poly2(A1))); print('B3v2 iter', it + 1, R_[-1], flush=True); Ap = A1
r = dict(tag=tag, BAND=RHO, STOP=STOP, coarse=co, iters=R_); json.dump(r, open(os.path.join(os.environ['BIGOUT'], 'b3v2_%s.json' % tag), 'w'), indent=1, default=float); print('ГОТОВО', flush=True)
