"""b15: покрытие/V* у дисков на базе c1600 (гипотеза research-30: клетки у дисков мелкие/пессимистичные). Полосы зазора dobs=dist−.7 ∈ [0,.1],[.1,.3],[.3,.6],[.6,1.2]:
(а) случайные (x,y) вне дисков, (θ,v) — с ближайшей точки эталонных траекторий (env_loss.json Yr; берутся точки ≤ 1 м от пути): доля covered, доля V*<BIG, медианный объём покрывающей клетки (∏2r), мед. min-радиус;
(б) точки на эталонных траекториях: V* / (T_ref − t) по полосам. Запуск: M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 PESS=1 GS=300 SYS=car LOAD=~/spore_v5/car/c1600.pkl python envs_band.py"""
import os, sys, json
os.environ.setdefault('SYS', 'car'); os.environ.setdefault('OMP_NUM_THREADS', '4')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np, pickle, inspect, __main__, growCar as G, common
from scipy.spatial import cKDTree
for nm, o in vars(G).items():
    if inspect.isclass(o): setattr(__main__, nm, o)
d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); dob = lambda p: (np.linalg.norm(p[:, None, :] - HZ[None], axis=2) - .7).min(1)
L = json.load(open(os.path.join(common.DATA, 'env_loss.json')))
RP = []; RT = []
for r in L:
    Y = np.array(r['Yr']); T = r['ref']['T']; t = np.linspace(0, T, len(Y)); RP.append(Y); RT.append(T - t)
RP = np.concatenate(RP); RT = np.concatenate(RT); tr = cKDTree(RP[:, :2])
rng = np.random.default_rng(0); P = rng.uniform(-5, 5, (400000, 2)); P = P[dob(P) > 0]; dd, ii = tr.query(P); k = dd <= 1.0; P, ii = P[k], ii[k]
Y = np.c_[P, RP[ii, 2], np.clip(RP[ii, 3], -1, 2)]; band = dob(P)
A.stencils(Y[:10]); R = np.array([c.r for c in A.cells]); vol = np.prod(2 * R, 1); rmin = R.min(1)
def run(Y):
    cov = A.qx.covered(Y) if hasattr(A.qx, 'covered') else None; pi, hid, sc = G._pquery(A.qx, G.wrapy(Y)); cid = np.full(len(Y), -1); cid[pi] = A.qx.I[hid, 0]
    return cov, cid, A.vstar(Y)
cov, cid, vs = run(Y)
print('полоса dobs | n | covered | V*<BIG | мед. объём кл. | мед. min-радиус | мед. V*')
bands = [(0, .1), (.1, .3), (.3, .6), (.6, 1.2), (1.2, 9)]; out = []
for lo, hi in bands:
    m = (band >= lo) & (band < hi); c = cid[m]; ok = c >= 0
    row = [m.sum(), cov[m].mean(), (vs[m] < 500).mean(), np.median(vol[c[ok]]) if ok.any() else np.nan, np.median(rmin[c[ok]]) if ok.any() else np.nan, np.median(vs[m][vs[m] < 500]) if (vs[m] < 500).any() else np.nan]
    out.append(row); print(f'[{lo},{hi}) | {row[0]} | {row[1]:.3f} | {row[2]:.3f} | {row[3]:.2e} | {row[4]:.3f} | {row[5]:.2f}')
Yr = RP.copy(); Yr[:, 3] = np.clip(Yr[:, 3], -1, 2); br = dob(Yr[:, :2]); vr = A.vstar(Yr[:, :4]); m0 = (RT > .5)
print('на эталоне: полоса | n | V*<BIG | мед. V*/(T_ref−t) | p90')
for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, 1.2), (1.2, 9)]:
    m = (br >= lo) & (br < hi) & m0 & (vr < 500)
    n_all = ((br >= lo) & (br < hi) & m0).sum(); q = vr[m] / RT[m] if m.any() else np.array([np.nan])
    print(f'[{lo},{hi}) | {n_all} | {m.sum()/max(n_all,1):.3f} | {np.median(q):.3f} | {np.percentile(q,90):.3f}')
