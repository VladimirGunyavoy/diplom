"""линия D: поток масштабированной динамики робота для коридора v6 — компилированный (gcc) CasADi: шаг RK4 (x,u,h), пакет map(CH), цикл по шагам в питоне;
flow_all(P, t) — все слои из одной точки за один вызов (8 шагов RK4, полудуга t/2 бесплатно) для tree_fast. Слои — LAYERS (4 угла) или LAYERS9."""
import os, sys, subprocess
import numpy as np, casadi as ca
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lw_dyn as L
TUTAG = ('%g' % L.TU).replace('.', 'p')
def _so(name, F, nmap_args=None):
    so = os.path.join(HERE, '%s_tu%s.so' % (name, TUTAG)); c = so[:-3] + '.c'
    if not os.path.exists(so):
        cwd = os.getcwd(); os.chdir(HERE); F.generate(os.path.basename(c), {'with_header': False}); os.chdir(cwd); subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c, '-o', so, '-lm'])
    return so
def _step_expr(steps):
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2); t = ca.SX.sym('t'); fx = ca.Function('f', [x, u], [L.f_ca(x, u)]); h = t / steps; z = x; zm = x
    for i in range(steps):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i == steps // 2 - 1: zm = z
    return x, u, t, z, zm
def make_flow_all(layers, steps=8):
    nl = len(layers); x, u, t, z, zm = _step_expr(steps); F = ca.Function('F', [x, u, t], [ca.vertcat(z, zm)])
    Xs, Us, Ts = ca.SX.sym('X', 6, nl), ca.SX.sym('U', 2, nl), ca.SX.sym('T', 1, nl); G = ca.Function('FM_lw%d' % nl, [Xs, Us, Ts], [F.map(nl)(Xs, Us, Ts)])
    so = os.path.join(HERE, 'flm_lw%d_s%d_tu%s.so' % (nl, steps, TUTAG)); c = so[:-3] + '.c'
    if not os.path.exists(so):
        cwd = os.getcwd(); os.chdir(HERE); G.generate(os.path.basename(c), {'with_header': False}); os.chdir(cwd); subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c, '-o', so, '-lm'])
    Fe = ca.external('FM_lw%d' % nl, so); U = np.asarray(layers, float).T
    def flow_all(P, t):
        r = np.asarray(Fe(np.tile(np.asarray(P, float)[:, None], (1, nl)), U, np.full((1, nl), t))).T; flow_all.half = r[:, 6:]; return r[:, :6]
    return flow_all
def make_flow(layers, hmax=.05, CH=128):
    """flow(P, s, t, dt_max): P (6,) или (n,6); шаги RK4 по hmax (в TU), пакеты map(CH) на один шаг"""
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2); h = ca.SX.sym('h'); fx = ca.Function('f', [x, u], [L.f_ca(x, u)])
    k1 = fx(x, u); k2 = fx(x + h / 2 * k1, u); k3 = fx(x + h / 2 * k2, u); k4 = fx(x + h * k3, u); F1 = ca.Function('St_lw', [x, u, h], [x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)])
    so = os.path.join(HERE, 'st_lw_tu%s.so' % TUTAG); c = so[:-3] + '.c'
    if not os.path.exists(so):
        cwd = os.getcwd(); os.chdir(HERE); F1.generate(os.path.basename(c), {'with_header': False}); os.chdir(cwd); subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c, '-o', so, '-lm'])
    Fe = ca.external('St_lw', so).map(CH); Uv = np.asarray(layers, float)
    def flow(P, s, t, dt_max=hmax):
        P = np.asarray(P, float); one = P.ndim == 1; P = np.atleast_2d(P); k = max(1, int(np.ceil(abs(t) / dt_max))); hh = t / k; out = np.empty_like(P)
        for a in range(0, len(P), CH):
            blk = P[a:a + CH]; m = len(blk); X = np.zeros((6, CH)); X[:, :m] = blk.T; Uc = np.tile(Uv[s][:, None], (1, CH)); Hc = np.full((1, CH), hh)
            for _ in range(k): X = np.asarray(Fe(X, Uc, Hc))
            out[a:a + m] = X[:, :m].T
        return out[0] if one else out
    return flow
