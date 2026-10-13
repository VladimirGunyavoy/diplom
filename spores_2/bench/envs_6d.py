"""линия A: среды 6D/8D для бенча CA1 (наследники common.Env). Импорт регистрирует их в common.make (monkeypatch — common.py не правим).
M3D — 3D двузвенный манипулятор без g (manip3d_dyn.py = r21/manip3d.py): состояние (q0,q1,q2,w0,w1,w2), моменты |τi| ≤ 1, цель окно |q|<.3, |w|<.6,
|w| ≤ 3 по пути (коробка проверки), старты — 20 из потока rng(0) (после 32 затравок окна), эталон OCP CasADi (data/m3d_ref/refm3d_m3d_q*_G0_N50_W3.npy: [T, U(50×3)])."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common
from common import Env, wrap, DATA
import manip3d_dyn as M3
import manip3g_dyn as M3G_
PI = np.pi

def m3d_starts(nq=20):
    rng = np.random.default_rng(0)
    for _ in range(32): rng.uniform(-1, 1, 6)
    return np.array([[*rng.uniform(-PI, PI, 3), *rng.uniform(-1, 1, 3)] for _ in range(nq)])

class M3D(Env):
    name = 'm3d'; n = 6; m = 3; period = np.array([2 * PI] * 3 + [0.] * 3); half = np.array([.3] * 3 + [.6] * 3); lo = np.array([-PI] * 3 + [-3.] * 3); hi = -lo
    umin = -np.ones(3); umax = np.ones(3); fl_lo = np.array([-PI] * 3 + [-1.] * 3); fl_hi = -fl_lo
    def __init__(s, nst=20):
        s.X0 = m3d_starts(nst); T = np.full(nst, np.nan)
        for i in range(nst):
            fn = os.path.join(DATA, 'm3d_ref', 'refm3d_m3d_q%d_G0_N50_W3.npy' % (i + 1))
            if os.path.exists(fn): T[i] = np.load(fn)[0]
        s.keep = np.isfinite(T); s.Tref = np.where(s.keep, T, 8.)
    TOLG, TOLW = 1e-3, 1e-2                                   # допуски эталона OCP (ref_m3d.py: окно +1e-3, |w| ≤ 3 + 1e-2) — одинаково для всех методов
    def f(s, Y, u): s.nf += 1; return M3.f(Y, u)
    def in_goal(s, Y): return (np.abs(wrap(np.asarray(Y), s.period)) <= s.half + s.TOLG).all(-1)
    def goal_dist(s, Y): return (np.abs(wrap(np.asarray(Y, float), s.period)) / (s.half + s.TOLG)).max(-1) - 1.
    def out_of_box(s, Y): return (np.abs(np.asarray(Y)[..., 3:]) > 3. + s.TOLW).any(-1)

class M3G(M3D):
    """3 звена 6D + гравитация g=.3 (all_systems №8): v6 manip3dyn (массы 1.5/1/.5, длины 1, плоский, θ от горизонтали); цель — покой в q*=(∓π/2,0,0): вниз (C0=-π/2) / вверх (+π/2); окно |q−q*|<.3, |w|<.6, |w|≤3 по пути;
    старты те же 20 (rng(0)); T_ref — лучшее известное OCP (|w|≤3 вдоль пути) из r5 `ref6d_T.json` best.{down,up}.W3; для «вверх» мультистарт OCP не сошёлся на 7/31 — неопределённость до ~6 %."""
    TOLW = .05                                                 # эталон OCP (N50) ограничивает |w|≤3 только в узлах: внутри интервалов до 3.05 ⇒ допуск .05 для ВСЕХ методов (m3d: .01)
    name = 'm3gd'; G = .3; C0 = -PI / 2; KEY = 'down'
    def __init__(s, nst=20):
        s.X0 = m3d_starts(nst); T = np.array(json.load(open(os.path.join(DATA, 'm3g_ref.json')))[s.KEY][:nst], float); s.keep = np.isfinite(T); s.Tref = np.where(s.keep, T, 8.)
        s.C3 = np.array([s.C0, 0, 0, 0, 0, 0]); n = 3; s.A = np.tril(np.ones((n, n))); s.Ai = np.linalg.inv(s.A); m = M3G_.MS[:n]; ln = M3G_.L[:n]
        s.S = np.array([[m[max(i, j):].sum() * ln[i] * ln[j] for j in range(n)] for i in range(n)]); s.gw = s.G * ln * np.array([m[i:].sum() for i in range(n)])
    def f(s, Y, u):
        s.nf += 1; Y = np.asarray(Y, float); q, w = Y[..., :3], Y[..., 3:]; th = q @ s.A.T; thd = w @ s.A.T; d = th[..., :, None] - th[..., None, :]
        Mt = s.S * np.cos(d); h = np.einsum('ij,...ij,...j->...i', s.S, np.sin(d), thd ** 2); u = np.broadcast_to(u, Y[..., :3].shape); Qt = u @ s.Ai.T.T - s.gw * np.cos(th)      # A^{-T} τ
        Qt = u @ np.linalg.inv(s.A.T).T - s.gw * np.cos(th); tdd = np.linalg.solve(Mt, (Qt - h)[..., None])[..., 0]; return np.concatenate([w, tdd @ s.Ai.T], -1)
    def _d(s, Y): return wrap(np.asarray(Y, float) - s.C3, s.period)
    def in_goal(s, Y): return (np.abs(s._d(Y)) <= s.half + s.TOLG).all(-1)
    def goal_dist(s, Y): return (np.abs(s._d(Y)) / (s.half + s.TOLG)).max(-1) - 1.
    def embed(s, Y):
        Y = np.asarray(Y, float) - s.C3; return M3D.embed(s, Y + 0)        # метрика: относительно цели
class M3GU(M3G): name = 'm3gu'; C0 = PI / 2; KEY = 'up'

class M8(Env):
    """4 звена 8D + g .3, цель вниз (q1=-π/2; v6 stats_manip4.py): окно |q−q*|<.3, |w|<.6, |w|≤3(+.05); моменты |τ|≤1; массы 1.5/1/.5/.25. Старты — тот же поток rng(0) (после 32 затравок окна 8D), 20 шт.
    Эталона OCP нет в данных: T_ref = лучшее известное среди методов (data/m8_ref.json, если есть; иначе 10 с-заглушка) — отношения «к лучшему найденному», не к оптимуму."""
    name = 'm8'; NL = 4; G = .3; n = 8; m = 4; TOLG, TOLW = 1e-3, .05
    period = np.array([2 * PI] * 4 + [0.] * 4); half = np.array([.3] * 4 + [.6] * 4); lo = np.array([-PI] * 4 + [-3.] * 4); hi = -lo; umin = -np.ones(4); umax = np.ones(4); fl_lo = np.array([-PI] * 4 + [-1.] * 4); fl_hi = -fl_lo
    def __init__(s, nst=20):
        rng = np.random.default_rng(0)
        for _ in range(32): rng.uniform(-1, 1, 8)
        s.X0 = np.array([[*rng.uniform(-PI, PI, 4), *rng.uniform(-1, 1, 4)] for _ in range(nst)]); s.keep = np.ones(nst, bool); s.Tref = np.full(nst, 10.)
        fn = os.path.join(DATA, 'm8_ref.json')
        if os.path.exists(fn): T = np.array(json.load(open(fn))[:nst], float); s.keep = np.isfinite(T); s.Tref = np.where(s.keep, T, 10.)
        n = 4; s.C3 = np.zeros(8); s.C3[0] = -PI / 2; s.A = np.tril(np.ones((n, n))); s.Ai = np.linalg.inv(s.A); s.AiT = np.linalg.inv(s.A.T).T; m = M3G_.MS[:n]; ln = M3G_.L[:n]
        s.S = np.array([[m[max(i, j):].sum() * ln[i] * ln[j] for j in range(n)] for i in range(n)]); s.gw = s.G * ln * np.array([m[i:].sum() for i in range(n)])
    def f(s, Y, u):
        s.nf += 1; Y = np.asarray(Y, float); q, w = Y[..., :4], Y[..., 4:]; th = q @ s.A.T; thd = w @ s.A.T; d = th[..., :, None] - th[..., None, :]
        Mt = s.S * np.cos(d); h = np.einsum('ij,...ij,...j->...i', s.S, np.sin(d), thd ** 2); u = np.broadcast_to(u, Y[..., :4].shape); Qt = u @ s.AiT - s.gw * np.cos(th)
        tdd = np.linalg.solve(Mt, (Qt - h)[..., None])[..., 0]; return np.concatenate([w, tdd @ s.Ai.T], -1)
    def _d(s, Y): return wrap(np.asarray(Y, float) - s.C3, s.period)
    def in_goal(s, Y): return (np.abs(s._d(Y)) <= s.half + s.TOLG).all(-1)
    def goal_dist(s, Y): return (np.abs(s._d(Y)) / (s.half + s.TOLG)).max(-1) - 1.
    def out_of_box(s, Y): return (np.abs(np.asarray(Y)[..., 4:]) > 3. + s.TOLW).any(-1)
    def embed(s, Y): return Env.embed(s, np.asarray(Y, float) - s.C3)

def register():
    if getattr(common, '_a6d', False): return
    old = common.make
    def make(name):
        tab = dict(m3d=M3D, m3gd=M3G, m3gu=M3GU, m8=M8)
        try: from envs_6d_b import TAB; tab.update(TAB)
        except ImportError: pass
        return tab[name]() if name in tab else old(name)
    common.make = make; common._a6d = True
register()
