"""линия D: динамика колёсно-ножного робота пользователя (legwheel_robot, ЧУЖОЙ проект — только чтение) в CasADi + масштабирование для споров.
Физика — те же формулы, что colab-mhlb/legwheel_robot/env.py (Лагранжиан: M(q)q̈ + c = Q − ∇V, четырёхзвенник, пружина); производные — автодиффом
(у оригинала центральные конечные разности EPS 1e-6 — расхождение ~1e-8). Истина для проверяльщика — САМ env.rk4_step робота (см. env_lw.py).
Масштаб: z = (x/Sx, ψ/Sψ, (α−α*)/Sα, ẋ/Sxd, ψ̇/Sψd, α̇/Sαd), так что окно цели робота = |z_pos| < .3, |z_vel| < .6 (как у манипуляторов линии A);
время t' = t/TU (TU = .1 с); управление u∈[-1,1]² → (τ_wheel, τ_hip) = (4 u0, .5 u1) Н·м."""
import os, sys, math
import numpy as np
import casadi as ca
CM = os.environ.get('LWROOT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'colab')      # aida: ~/spore_v5/d1/colab (копия legwheel_robot)
sys.path.insert(0, os.path.abspath(CM))
from legwheel_robot import config as C

# ---- масштабы ----
EPS_Q = np.array([C.SUCCESS_X_EPS, C.SUCCESS_PSI_EPS, C.SUCCESS_ALPHA_EPS])               # окно по положениям
EPS_W = np.array([C.SUCCESS_X_DOT_EPS, C.SUCCESS_PSI_DOT_EPS, C.SUCCESS_ALPHA_DOT_EPS])    # окно по скоростям
RQ, RW = .3, .6
SQ = EPS_Q / RQ; SW = EPS_W / RW; SCALE = np.r_[SQ, SW]
TU = float(os.environ.get('LWTU', .1))
UMAX = np.array([C.MAX_TORQUE_WHEEL, C.MAX_TORQUE_HIP])

def _geom(a):
    """символьно: D, C, E_local, s_spring от угла бедра a (ветвь +1, как env.py)"""
    D = ca.vertcat(C.l4 * ca.cos(a), C.l4 * ca.sin(a)); B = ca.DM([C.Bx, C.By]); dv = D - B; d = ca.norm_2(dv)
    aa = (C.l2 ** 2 - C.l3 ** 2 + d ** 2) / (2 * d); h = ca.sqrt(ca.fmax(C.l2 ** 2 - aa ** 2, 0.))
    Mp = B + aa * dv / d; perp = ca.vertcat(-dv[1], dv[0]) / d; Cc = Mp + h * perp
    dr = D - Cc; E = D + C.l5 * dr / ca.norm_2(dr); F = .75 * D; s = ca.norm_2(Cc - F)
    return D, Cc, E, s

_a = ca.SX.sym('a'); _E_star = float(ca.Function('e', [_a], [_geom(_a)[2]])(C.ALPHA_STAR)[0])
R_CM = ca.DM([_E_star, 0.04])

def f_phys():
    """Function(x(6), tau(2)) → ẋ(6) в физических единицах"""
    x = ca.SX.sym('x', 6); tau = ca.SX.sym('tau', 2); q = x[:3]; qd = x[3:]; psi = q[1]; a = q[2]
    _, _, E, s = _geom(a); u = R_CM - E
    rot = lambda p, v: ca.vertcat(ca.cos(p) * v[0] - ca.sin(p) * v[1], ca.sin(p) * v[0] + ca.cos(p) * v[1])
    P = ca.vertcat(q[0], C.WHEEL_RADIUS) + rot(psi, u)                  # положение центра масс тела
    J = ca.jacobian(P, q)
    M = C.BODY_MASS * (J.T @ J); M[1, 1] += C.BODY_INERTIA; M[0, 0] += C.WHEEL_MASS + C.WHEEL_INERTIA / C.WHEEL_RADIUS ** 2
    V = C.BODY_MASS * C.GRAVITY * P[1] + .5 * C.K_SPRING * (s - C.L0_SPRING) ** 2
    dM = [ca.reshape(ca.jacobian(ca.vec(M), q[k]), 3, 3) for k in range(3)]
    c = ca.vertcat(*[sum((dM[k][i, j] - .5 * dM[i][j, k]) * qd[j] * qd[k] for j in range(3) for k in range(3)) for i in range(3)])
    dV = ca.jacobian(V, q).T
    rhs = ca.vertcat(tau[0] / C.WHEEL_RADIUS, -tau[0], tau[1]) - c - dV
    qdd = ca.solve(M, rhs)
    return ca.Function('f_phys', [x, tau], [ca.vertcat(qd, qdd)])

_FP = None
def fp():
    global _FP
    if _FP is None: _FP = f_phys()
    return _FP

def to_phys(z):
    z = np.asarray(z, float); y = z * SCALE; y[..., 2] += C.ALPHA_STAR; return y
def from_phys(y):
    y = np.array(y, float); y[..., 2] -= C.ALPHA_STAR; return y / SCALE

def f_ca(z, u):
    """масштабированная динамика (casadi): z (6) безразмерное, u (2) в [-1,1] → dz/dt'"""
    sc = ca.DM(SCALE); y = z * sc; y = ca.vertcat(y[0], y[1], y[2] + C.ALPHA_STAR, y[3], y[4], y[5])
    return TU * fp()(y, ca.vertcat(UMAX[0] * u[0], UMAX[1] * u[1])) / sc

LAYERS = [(a, b) for a in (-1., 1.) for b in (-1., 1.)]                # 4 угла прямоугольника управления (bang-bang)
LAYERS9 = [(a, b) for a in (-1., 0., 1.) for b in (-1., 0., 1.)]
