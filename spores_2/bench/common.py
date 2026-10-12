"""bench: общий каркас. Env (динамика, цель, коробка, старты, T_ref, счётчик вызовов f) + единый проверяльщик verify().
Метод отдаёт последовательность (u_k, dt_k); verify интегрирует её сам (RK4 h=.01, у ДИ точно) и считает T (первое попадание в цель) и успех. v8/src — только чтение."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'v8')))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'reports', 'bench', 'data'))

def wrap(d, period):
    p = np.where(period > 0, period, 1.); return np.where(period > 0, (d + p / 2) % p - p / 2, d)

class Env:
    name = ''; n = 0; m = 1; period = None; ball = None; half = None; umin = None; umax = None; lo = None; hi = None; X0 = None; Tref = None; keep = None
    fl_lo = None; fl_hi = None
    nf = 0                                                   # счётчик вызовов f (RK4-шагов считаются как 4 f; точный шаг ДИ — как 1)
    def f(self, Y, u): raise NotImplementedError
    def in_goal(self, Y):
        if self.ball is not None: return np.linalg.norm(Y, axis=-1) <= self.ball + 1e-6
        return (np.abs(wrap(Y, self.period)) <= self.half).all(-1)
    def step(self, Y, u, h):
        k1 = self.f(Y, u); k2 = self.f(Y + h / 2 * k1, u); k3 = self.f(Y + h / 2 * k2, u); k4 = self.f(Y + h * k3, u); self.nf += 4
        Z = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); return np.where(self.period > 0, wrap(Z, self.period), Z)
    def out_of_box(self, Y): return ((Y < self.lo) | (Y > self.hi) & (self.period == 0)).any(-1)
    def sample_field(self, rng): return rng.uniform(self.fl_lo, self.fl_hi)
    def embed(self, Y):
        Y = np.asarray(Y, float); s = (self.hi - self.lo) / 2 + 1e-9; cols = []
        for i in range(self.n):
            if self.period[i] > 0: a = Y[..., i] * 2 * np.pi / self.period[i]; cols += [np.cos(a), np.sin(a)]
            else: cols.append(Y[..., i] / s[i])
        return np.stack(cols, -1)
    def embed_phys(self, Y):
        """метрика в физических единицах (углы — sin/cos): радиусы SST .05–.4 имеют смысл"""
        Y = np.asarray(Y, float); cols = []
        for i in range(self.n):
            if self.period[i] > 0: a = Y[..., i] * 2 * np.pi / self.period[i]; cols += [np.cos(a), np.sin(a)]
            else: cols.append(Y[..., i])
        return np.stack(cols, -1)
    def goal_dist(self, Y):
        Y = np.asarray(Y, float)
        if self.ball is not None: return np.linalg.norm(Y, axis=-1) - self.ball
        return (np.abs(wrap(Y, self.period)) / self.half).max(-1) - 1.
    def clip_u(self, u): return np.clip(u, self.umin, self.umax)

class DI(Env):
    exact = True; name = "di"; n = 2; m = 1; period = np.zeros(2); ball = .2; umin = np.array([-1.]); umax = np.array([1.]); lo = np.array([-6., -4.]); hi = np.array([6., 4.]); fl_lo = np.array([-3., -2.5]); fl_hi = -fl_lo
    def __init__(s, nst=20, seed=1):
        rng = np.random.default_rng(seed); out = []
        while len(out) < nst:
            p = rng.uniform([-3, -2.5], [3, 2.5])
            if np.linalg.norm(p) > s.ball: out.append(p)
        s.X0 = np.array(out); from src.algo import ref_di_disk as R; s.Tref, info = R.T_ref(s.X0, s.ball, return_details=True)
        ext = []
        for x, (sg, t1, tot) in zip(s.X0, info):                       # макс |x| вдоль эталонной траектории (как solve8.path_extent)
            t1_ = tot if t1 is None else t1; tt = np.linspace(0, tot, 300); a = np.minimum(tt, t1_); b = tt - a; x1 = x[0] + x[1] * a + sg * a * a / 2; v1 = x[1] + sg * a; ext.append(np.abs(x1 + v1 * b - sg * b * b / 2).max())
        s.ext = np.array(ext); s.keep = s.ext <= 3.5            # эталон выходит из поля (|x| > 3.5 = XL+GM) → исключён для всех методов
    def f(s, Y, u): s.nf += 1; return np.stack([Y[..., 1], np.broadcast_to(u[..., 0], Y[..., 1].shape)], -1)
    def step(s, Y, u, h):
        s.nf += 1; a = np.broadcast_to(u[..., 0], Y[..., 0].shape); h = np.asarray(h, float); return np.stack([Y[..., 0] + Y[..., 1] * h + a * h * h / 2, Y[..., 1] + a * h], -1)
    def out_of_box(s, Y): return (np.abs(Y[..., 0]) > s.hi[0]) | (np.abs(Y[..., 1]) > s.hi[1])

def make(name):
    if name == 'di': return DI()
    raise KeyError(name)

def verify(env, y0, us, dts, h=.01, tfac=3.):
    """единый проверяльщик: (u_k, dt_k) → T первого попадания в цель, успех (цель, T ≤ tfac·T_ref не проверяется здесь — делает вызывающий, коробка не нарушена). Возврат (T или None, причина)."""
    y = np.asarray(y0, float)[None].copy(); t = 0.
    if env.in_goal(y)[0]: return 0., 'ok'
    for u, dt in zip(us, dts):
        u = env.clip_u(np.asarray(u, float).reshape(1, -1)); k = max(1, int(np.ceil(dt / h - 1e-9))); hh = dt / k
        for _ in range(k):
            y = env.step(y, u, hh); t += hh
            if env.out_of_box(y)[0]: return None, 'box'
            if env.in_goal(y)[0]: return t, 'ok'
    return None, 'no_goal'

def summarize(env, Ts, extra=None):
    """Ts: список T (None — провал) по ВСЕМ старам; статистика по keep"""
    T = np.array([np.nan if t is None else t for t in Ts]); k = env.keep; ok = np.isfinite(T) & (T <= 3 * env.Tref); r = T[k & ok] / env.Tref[k & ok]
    return dict(n=int(k.sum()), excluded=int((~k).sum()), reached=float(ok[k].mean()), ratio_med=float(np.median(r)) if len(r) else None, ratio_mean=float(r.mean()) if len(r) else None,
                ratio_p90=float(np.percentile(r, 90)) if len(r) else None, ratio_max=float(r.max()) if len(r) else None)
