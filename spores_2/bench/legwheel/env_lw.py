"""линия D: среда бенча «колёсно-ножной робот пользователя» (наследник common.Env). Состояния методов — МАСШТАБИРОВАННЫЕ z (lw_dyn.py), время — в долях TU;
проверяльщик verify_real интегрирует ТЕМ ЖЕ rk4_step, что env робота (dt .002, постоянное управление на шаг, клип моментов, проверка исхода на каждом шаге:
выход α за механический предел / |x|>BOUND_X / расходимость → провал; все 6 в окне → успех), и отдаёт T в СЕКУНДАХ. Старты — из сетки eval пользователя
(41×41 по (ψ0, ψ̇0), α0=α*, x0=ẋ0=α̇0=0; ψ ±10°, ψ̇ ±12°/с), N=50 выбраны случайно (seed 0) из тех, что не в окне при t=0."""
import os, sys, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import lw_dyn as L
from lw_dyn import C
from legwheel_robot import env as E
import common
from common import Env

def grid_starts(n=41):
    """сетка eval (training/eval.py _build_grid): psi0 × psi_dot0 в физических единицах → список (psi0, psi_dot0)"""
    ps = np.linspace(-C.SPAWN_PSI_RANGE, C.SPAWN_PSI_RANGE, n); pd = np.linspace(-C.SPAWN_PSI_DOT_RANGE, C.SPAWN_PSI_DOT_RANGE, n)
    return np.array([[p, d] for p in ps for d in pd])

def phys_start(psi0, psid0): return np.array([0., psi0, C.ALPHA_STAR, 0., psid0, 0.])
def at_goal_phys(y):
    return (abs(y[0]) < C.SUCCESS_X_EPS and abs(y[1]) < C.SUCCESS_PSI_EPS and abs(y[2] - C.ALPHA_STAR) < C.SUCCESS_ALPHA_EPS and abs(y[3]) < C.SUCCESS_X_DOT_EPS
            and abs(y[4]) < C.SUCCESS_PSI_DOT_EPS and abs(y[5]) < C.SUCCESS_ALPHA_DOT_EPS)
def oob_phys(y): return (not all(math.isfinite(v) for v in y)) or y[2] < C.ALPHA_MIN_VALID or y[2] > C.ALPHA_MAX_VALID or abs(y[0]) > C.BOUND_X

def rollout_real(y0_phys, policy_steps):
    """policy_steps: итератор (tau_w, tau_h) на каждый шаг dt .002 → (T_с или None, причина, число шагов). Первая проверка — после первого шага (как env робота)."""
    y = list(y0_phys); n = 0
    for tw, th in policy_steps:
        tw = max(-C.MAX_TORQUE_WHEEL, min(C.MAX_TORQUE_WHEEL, tw)); th = max(-C.MAX_TORQUE_HIP, min(C.MAX_TORQUE_HIP, th))
        try: y = E.rk4_step(y, tw, th, C.PHYSICS_DT)
        except ZeroDivisionError: return None, 'diverged', n
        n += 1
        if oob_phys(y): return None, 'oob', n
        if at_goal_phys(y): return n * C.PHYSICS_DT, 'ok', n
    return None, 'no_goal', n

def verify_real(z0, us, dts_scaled):
    """единый проверяльщик: кусочно-постоянное u (в [-1,1]², масштабированное) на отрезках dts (в TU) из z0 (масштабированное). Отрезок разбит на шаги ≤ .002 с (равные).
    Возврат: (T_с, причина). Цель в t=0 (все 6 в окне) → T=0 (в отличие от env робота, где проверка после первого шага — такие старты из набора исключены)."""
    y0 = L.to_phys(np.asarray(z0, float)); t = 0.
    if at_goal_phys(y0): return 0., 'ok'
    y = list(y0)
    for u, dt in zip(us, dts_scaled):
        u = np.clip(np.asarray(u, float), -1, 1); tw, th = float(L.UMAX[0] * u[0]), float(L.UMAX[1] * u[1]); sec = float(dt) * L.TU
        if sec <= 0: continue
        k = max(1, int(math.ceil(sec / C.PHYSICS_DT - 1e-9))); h = sec / k
        for _ in range(k):
            try: y = E.rk4_step(y, tw, th, h)
            except ZeroDivisionError: return None, 'diverged'
            t += h
            if oob_phys(y): return None, 'oob'
            if at_goal_phys(y): return t, 'ok'
    return None, 'no_goal'

class LW(Env):
    """6D, 2 входа (u0 колесо ±4 Н·м, u1 бедро ±.5 Н·м). Состояния — масштабированные z (окно: |z_pos|<.3, |z_vel|<.6, ровно окно успеха робота)."""
    name = 'lw'; n = 6; m = 2; period = np.zeros(6); half = np.r_[np.full(3, L.RQ), np.full(3, L.RW)]; umin = -np.ones(2); umax = np.ones(2)
    fl_lo = None; fl_hi = None
    ALPHA_LO = (C.ALPHA_MIN_VALID - C.ALPHA_STAR) / L.SQ[2]; ALPHA_HI = (C.ALPHA_MAX_VALID - C.ALPHA_STAR) / L.SQ[2]
    def __init__(s, nst=50, seed=0, ref_file=None):
        G = grid_starts(); rng = np.random.default_rng(seed); ok = np.array([not at_goal_phys(phys_start(*g)) for g in G]); idx = np.nonzero(ok)[0]
        s.sel = rng.choice(idx, nst, replace=False); s.X0 = np.array([L.from_phys(phys_start(*G[i])) for i in s.sel]); s.G = G[s.sel]
        s.keep = np.ones(nst, bool); s.Tref = np.full(nst, 1.)
        ref_file = ref_file or os.path.join(common.DATA, 'lw_ref.json')
        if os.path.exists(ref_file):
            T = np.array(json.load(open(ref_file))[:nst], float); s.keep = np.isfinite(T); s.Tref = np.where(s.keep, T, 1.)
    def f(s, Y, u): raise NotImplementedError('используйте lw_dyn.f_ca / f_np')
    def in_goal(s, Y): return (np.abs(np.asarray(Y)) <= s.half).all(-1)
    def goal_dist(s, Y): return (np.abs(np.asarray(Y, float)) / s.half).max(-1) - 1.
    def out_of_box(s, Y): Y = np.asarray(Y); return (Y[..., 2] < s.ALPHA_LO) | (Y[..., 2] > s.ALPHA_HI) | (np.abs(Y[..., 0] * L.SQ[0]) > C.BOUND_X)
    def verify(s, i, us, dts): return verify_real(s.X0[i], us, dts)
