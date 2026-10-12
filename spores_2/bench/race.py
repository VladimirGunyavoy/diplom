"""RACE: гоночная трасса «Картленд» как система бенча (спецификация knowledge/research/race_track.md; общий gfree — для b9 и c3).
Состояние 5D (x, y, θ, v, κ) в метрах/секундах (пкс × .2 м, ось y вниз как в картинке; ход по часовой), управление u = (s_κ, s_a) ∈ [−1, 1]² (у спор — слои {−1,0,1}²).
ẋ = v cosθ, ẏ = v sinθ, θ̇ = vκ, κ̇ = K s_κ, v̇ = s_a·min(a_max(1 − v/v_max), a_дост) при s_a > 0, s_a·min(b_max, a_дост) при s_a < 0; a_дост = √((μg)² − (v²κ)²).
gfree(Y) ≤ 0 — запрещено: выезд за трек (sdf − запас), срыв v²|κ| > μg, |κ| > κ_max, v ∉ [0, v_max]. Цель — пересечь финиш (полоса 3 пкс за линией y=240, x 410..438, ход вниз).
a_max(v) = a_max(1 − v/v_max) — выбор c3 (в спецификации «газ ≤ a_max(v)» без формулы); меняется здесь одной строкой."""
import os, sys, numpy as np, scipy.ndimage as nd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
PXM = .2; MUG = 12.; VMAX = 15.; AMAX = 4.; BMAX = 8.; KMAX = 1 / 3.; KRATE = 2 * KMAX / .5; MARGIN = .2; FIN_Y = 240 * PXM; FIN_X = (410 * PXM, 438 * PXM); FIN_W = 3 * PXM
SDF = np.load(os.path.join(common.REF, 'r25', 'kartland_sdf_px.npz'))['sdf'].astype(float)

def sdf_m(x, y):
    """знаковое расстояние до края трассы (м); > 0 — на трассе; вне картинки — −большое"""
    x = np.asarray(x, float) / PXM; y = np.asarray(y, float) / PXM; sh = x.shape
    v = nd.map_coordinates(SDF, [y.ravel(), x.ravel()], order=1, mode='nearest') * PXM
    out = (x.ravel() < 0) | (y.ravel() < 0) | (x.ravel() > SDF.shape[1] - 1) | (y.ravel() > SDF.shape[0] - 1)
    return np.where(out, -50., v).reshape(sh)

def centerline():
    """ось трассы: обход хребта sdf от финиша по ходу; (P (n,2) м, s (n,) м накопленная длина, замкнута)"""
    sd = lambda p: nd.map_coordinates(SDF, [[p[1]], [p[0]]], order=1)[0]
    grad = lambda p, e=1.: np.array([(sd(p + [e, 0]) - sd(p - [e, 0])) / (2 * e), (sd(p + [0, e]) - sd(p - [0, e])) / (2 * e)])
    p = np.array([424., 240.]); t = np.array([0., 1.]); pts = [p.copy()]
    for i in range(1000):
        q = p + 3 * t
        for _ in range(6): g = grad(q); n = np.array([-t[1], t[0]]); q = q + np.clip(g @ n, -1, 1) * n * .8
        tn = q - p; tn /= np.linalg.norm(tn); t = .7 * t + .3 * tn; t /= np.linalg.norm(t); p = q; pts.append(p.copy())
        if i > 20 and np.linalg.norm(p - pts[0]) < 4: break
    P = np.array(pts) * PXM; s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; return P, s
CL, CLS = centerline(); LAP = CLS[-1]
def cl_at(s):
    """точка оси и курс (рад) на длине s вдоль оси (по кругу)"""
    s = np.mod(s, LAP); x = np.interp(s, CLS, CL[:, 0]); y = np.interp(s, CLS, CL[:, 1]); e = 1.
    x2 = np.interp(np.mod(s + e, LAP), CLS, CL[:, 0]); y2 = np.interp(np.mod(s + e, LAP), CLS, CL[:, 1]); return np.stack([x, y], -1), np.arctan2(y2 - y, x2 - x)
def progress(Y):
    """длина вдоль оси до ближайшей точки оси (0 — финиш, растёт по ходу)"""
    from scipy.spatial import cKDTree
    global _TREE
    try: _TREE
    except NameError: _TREE = cKDTree(CL)
    return CLS[_TREE.query(np.asarray(Y)[..., :2])[1]]

class Race(common.Env):
    """5D Картленд; старты — 20 по длине оси; T_ref — OCP IPOPT (data/race_ref.json, вне таймера); keep — где эталон есть"""
    name = 'race'; n = 5; m = 2; period = np.array([0., 0., 2 * np.pi, 0., 0.]); lo = np.array([0., 0., -np.pi, 0., -KMAX]); hi = np.array([SDF.shape[1] * PXM, SDF.shape[0] * PXM, np.pi, VMAX, KMAX])
    umin = np.array([-1., -1.]); umax = np.array([1., 1.]); fl_lo = lo; fl_hi = hi
    def __init__(s, nst=20, seed=3):
        rng = np.random.default_rng(seed); ss = (np.arange(nst) + rng.uniform(.1, .9, nst)) / nst * LAP; xy, th = cl_at(ss)
        s.X0 = np.c_[xy, th + np.radians(rng.uniform(-20, 20, nst)), rng.uniform(0, VMAX * .9, nst), np.zeros(nst)]; s.s0 = ss
        s.X0 = s.X0[~s.in_goal(s.X0)]; s.Tref = np.full(len(s.X0), np.nan); s.keep = np.zeros(len(s.X0), bool)
        fn = os.path.join(common.DATA, 'race_ref.json')
        if os.path.exists(fn):
            import json; R = json.load(open(fn)); s.Tref = np.array([np.nan if r['Tref'] is None else r['Tref'] for r in R[:len(s.X0)]]); s.keep = np.isfinite(s.Tref)
    def f(s, Y, u):
        s.nf += 1; Y = np.asarray(Y, float); u = np.asarray(u, float); th, v, k = Y[..., 2], Y[..., 3], Y[..., 4]; sk, sa = u[..., 0], u[..., 1]
        ad = np.sqrt(np.maximum(MUG ** 2 - (v * v * k) ** 2, 0.)); a = np.where(sa > 0, sa * np.minimum(AMAX * np.maximum(1 - v / VMAX, 0), ad), sa * np.minimum(BMAX, ad))
        return np.stack([v * np.cos(th), v * np.sin(th), v * k, a + 0 * v, KRATE * sk + 0 * v], -1)
    def gfree(s, Y):
        Y = np.asarray(Y, float); v, k = Y[..., 3], Y[..., 4]
        return np.min(np.stack([sdf_m(Y[..., 0], Y[..., 1]) - MARGIN, MUG - v * v * np.abs(k), KMAX - np.abs(k), v, VMAX - v], -1), -1)
    def out_of_box(s, Y): return s.gfree(Y) < 0
    def in_goal(s, Y):
        Y = np.asarray(Y, float); return (Y[..., 0] >= FIN_X[0]) & (Y[..., 0] <= FIN_X[1]) & (Y[..., 1] >= FIN_Y) & (Y[..., 1] <= FIN_Y + FIN_W) & (np.sin(Y[..., 2]) > 0)
    def goal_dist(s, Y):
        Y = np.asarray(Y, float); return np.hypot(np.maximum(np.maximum(FIN_X[0] - Y[..., 0], Y[..., 0] - FIN_X[1]), 0), Y[..., 1] - FIN_Y - FIN_W / 2)
    def clip_u(s, u): return np.clip(u, -1, 1)
