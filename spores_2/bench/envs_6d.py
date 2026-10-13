"""линия A: среды 6D/8D для бенча CA1 (наследники common.Env). Импорт регистрирует их в common.make (monkeypatch — common.py не правим).
M3D — 3D двузвенный манипулятор без g (manip3d_dyn.py = r21/manip3d.py): состояние (q0,q1,q2,w0,w1,w2), моменты |τi| ≤ 1, цель окно |q|<.3, |w|<.6,
|w| ≤ 3 по пути (коробка проверки), старты — 20 из потока rng(0) (после 32 затравок окна), эталон OCP CasADi (data/m3d_ref/refm3d_m3d_q*_G0_N50_W3.npy: [T, U(50×3)])."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common
from common import Env, wrap, DATA
import manip3d_dyn as M3
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

def register():
    if getattr(common, '_a6d', False): return
    old = common.make
    def make(name):
        tab = dict(m3d=M3D)
        try: from envs_6d_b import TAB; tab.update(TAB)
        except ImportError: pass
        return tab[name]() if name in tab else old(name)
    common.make = make; common._a6d = True
register()
