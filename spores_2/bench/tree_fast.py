"""линия A, SPD6: быстрый _tree коридора v6: (1) полудуга для blocked не считается, если препятствий нет (has_obst False — раньше лишний поток на каждое ребро);
(2) все L слоёв из узла — одним вызовом скомпилированной (gcc) CasADi-функции map(L): flow_all(P, t) → (L, d), 8 шагов RK4 на τ=.4 (как dt_max .05).
install(name): подменяет adaptive_nd._tree и corridor_nd._tree; S.flow_all навешивает вызывающий (spores_6d.py)."""
import os, sys, heapq, subprocess
import numpy as np
import casadi as ca
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def make_flow_all(name, layers, steps=8, tag=''):
    nx, nu = (8, 4) if name == 'm8' else (6, 3)
    if name == 'm8': os.environ.setdefault('G', '0.3'); import ocp_m8_lib as L; fca = lambda x, u: L.f_ca(x, u, .3); steps = 20      # dt_max .02 на τ=.4
    elif name == 'm3d': import ocp_m3d_lib as L; fca = lambda x, u: L.f_ca(x, u)
    else: os.environ.setdefault('G', '0.3'); import ocp_m3g_lib as L; fca = lambda x, u: L.f_ca(x, u, .3)
    x = ca.SX.sym('x', nx); u = ca.SX.sym('u', nu); t = ca.SX.sym('t'); fx = ca.Function('f', [x, u], [fca(x, u)]); h = t / steps; z = x; zm = x
    for _i in range(steps):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if _i == steps // 2 - 1: zm = z      # состояние на полудуге (t/2) — бесплатно, тем же вызовом (w29)
    F = ca.Function('F_%s' % name, [x, u, t], [ca.vertcat(z, zm)]).map(len(layers)); so = os.path.join(HERE, 'flm_%s_%d.so' % (name, len(layers)))
    if not os.path.exists(so):
        c = so[:-3] + '.c'; Xs, Us, Ts = ca.SX.sym('X', nx, len(layers)), ca.SX.sym('U', nu, len(layers)), ca.SX.sym('T', 1, len(layers))
        G = ca.Function('FM_%s' % name, [Xs, Us, Ts], [F(Xs, Us, Ts)]); cwd = os.getcwd(); os.chdir(HERE); G.generate(os.path.basename(c), {'with_header': False}); os.chdir(cwd)
        subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c, '-o', so, '-lm'])
    Fe = ca.external('FM_%s' % name, so); U = np.asarray(layers, float).T
    def flow_all(P, t):
        r = np.asarray(Fe(np.tile(np.asarray(P, float)[:, None], (1, U.shape[1])), U, np.full((1, U.shape[1]), t))).T; flow_all.half = r[:, nx:]; return r[:, :nx]
    return flow_all
def make_flow_batch(name, layers, flow_orig, steps=4, CH=64):
    """пакетный поток для стыковки (t=±.2, 4 шага): скомпилированная map(CH), добивка нулями; прочие t — исходный flow."""
    if name == 'm3d': import ocp_m3d_lib as L; fca = lambda x, u: L.f_ca(x, u)
    else: os.environ.setdefault('G', '0.3'); import ocp_m3g_lib as L; fca = lambda x, u: L.f_ca(x, u, .3)
    x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 3); t = ca.SX.sym('t'); fx = ca.Function('f', [x, u], [fca(x, u)]); h = t / steps; z = x
    for _ in range(steps):
        k1 = fx(z, u); k2 = fx(z + h / 2 * k1, u); k3 = fx(z + h / 2 * k2, u); k4 = fx(z + h * k3, u); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    F1 = ca.Function('Fb_%s' % name, [x, u, t], [z]); so = os.path.join(HERE, 'flb_%s.so' % name); c = so[:-3] + '.c'
    if not os.path.exists(so):
        cwd = os.getcwd(); os.chdir(HERE); F1.generate(os.path.basename(c), {'with_header': False}); os.chdir(cwd); subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', c, '-o', so, '-lm'])
    Fe = ca.external('Fb_%s' % name, so).map(CH); Uv = np.asarray(layers, float)
    def flow(P, s, t, dt_max=.05, g=0.):
        P = np.asarray(P, float)
        if P.ndim != 2 or abs(abs(t) - .2) > 1e-12 or len(P) < 8: return flow_orig(P, s, t)
        out = np.empty_like(P)
        for a in range(0, len(P), CH):
            blk = P[a:a + CH]; m = len(blk); X = np.zeros((6, CH)); X[:, :m] = blk.T
            out[a:a + m] = np.asarray(Fe(X, np.tile(Uv[s][:, None], (1, CH)), np.full((1, CH), t)))[:, :m].T
        return out
    return flow
def _tree(S, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
    key = lambda p, s: (s,) + tuple(int(np.floor(p[k] / (rho * S.scale[k]))) for k in range(S.d))
    seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0; fa = getattr(S, 'flow_all', None); tt = tau if fwd else -tau
    def add(p, s, g, pa, ls):
        k = key(p, s)
        if k in seen: return -1
        seen.add(k); P.append(p); G.append(g); par.append(pa); lay.append(ls); return len(P) - 1
    def push(i, ns, ls):
        nonlocal cnt
        ch = []; E = S.wrap(fa(P[i], tt)) if fa is not None else None
        bl = None
        if fa is not None and S.has_obst: bl = np.asarray(S.blocked(np.vstack([E, fa.half]))); bl = bl[:S.L] | bl[S.L:]       # все L концов и полудуг — одним векторным вызовом (было L скалярных потоков на питоне)
        for s in range(S.L):
            e = E[s] if fa is not None else S.wrap(S.flow(P[i], s, tt))
            if not S.ok(e) or (S.has_obst and (bl[s] if bl is not None else (S.blocked(e) or S.blocked(S.flow(P[i], s, tt / 2))))): continue
            ch.append((s, e))
        hh = hfun(np.array([e for _, e in ch])) if (hfun is not None and ch) else [0.0] * len(ch)
        for (s, e), hv in zip(ch, hh):
            n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
            heapq.heappush(pq, (sw_w * n2 + G[i] + tau + hv, cnt, s, e, G[i] + tau, n2, i))
    for st in starts:
        i = add(S.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1)
    while pq and len(P) < N:
        _, _, s, e, g, n2, pa = heapq.heappop(pq)
        i = add(e, s, g, pa, s)
        if i >= 0: push(i, n2, s)
    return P, G, par, lay
def install():
    from src.atlas6 import adaptive_nd, corridor_nd
    adaptive_nd._tree = _tree; corridor_nd._tree = _tree
