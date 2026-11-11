"""ENV5 «круг на время» (НАША постановка, не метрика journal-2026): точка SafetyPointCircle2 (MuJoCo), 8 ворот по кругу, стены |x|,|y| = 1.125.
Состояние 5D (x, y, θ, vx, vy), ω квазистационарна (τ_ω ≈ .3 мс): ω = 3·u₂ (u₂∈{−1,0,1}); тяга F = .3·clip(u₁,±.05) вдоль курса; демпфирование .01; смещение ЦМ a=.01927 → центростремительный член a·ω²(cosθ,sinθ). Коридор v6 (NB/NF + SLSQP), проверка — шагами MuJoCo.
Запуск: python spores_pcircle.py NB NF K [tag]"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
import numpy as np, numba as nb, multiprocessing as mp
from src.atlas6.adaptive_nd import SysN, build_back
from src.atlas6.corridor_nd import corridor_query
NB, NF, K = [int(a) for a in sys.argv[1:4]]; tag = sys.argv[4] if len(sys.argv) > 4 else ''
M_, A_, CD, FM, WS = .005188790204786391, .0192723151357623, .01, .015, 3.0
WALL = 1.125; PAD = float(os.environ.get('PAD', .08)); RG = .1; GM = float(os.environ.get('GMARG', .03)); NG = 8; RC = 1.5; VM = 2.0; TAU = float(os.environ.get('TAU', .2))
# ворота: радиус min(R, .9·стена/max(|cos|,|sin|)) (решение research-32)
ang = 2 * np.pi * np.arange(NG) / NG; rr = np.minimum(RC, .9 * WALL / np.maximum(np.abs(np.cos(ang)), np.abs(np.sin(ang)))); GATES = np.stack([rr * np.cos(ang), rr * np.sin(ang)], 1)
U_ = np.array([(a, b) for a in (1., -1., 0.) for b in (1., -1., 0.)]); L = len(U_)   # слой: (u1 тяга, u2 поворот)
@nb.njit
def _fl(x, y, th, vx, vy, u0, u1, t, CD, M, A, FM, WS):
    sg = 1. if t >= 0 else -1.; n = int(abs(t) / .01 + 1e-9); hr = t - sg * .01 * n; w = WS * u1; F = FM * u0
    for k in range(n + 1):
        h = sg * .01 if k < n else hr
        if h == 0.: continue
        c = np.cos(th); s = np.sin(th)   # RK2 (средняя точка)
        ax = (F * c - CD * vx) / M + A * w * w * c; ay = (F * s - CD * vy) / M + A * w * w * s
        vxm = vx + .5 * h * ax; vym = vy + .5 * h * ay; thm = th + .5 * h * w; cm = np.cos(thm); sm = np.sin(thm)
        axm = (F * cm - CD * vxm) / M + A * w * w * cm; aym = (F * sm - CD * vym) / M + A * w * w * sm
        x += h * vxm; y += h * vym; th += h * w; vx += h * axm; vy += h * aym
    return x, y, th, vx, vy
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = U_[s]
    if P.ndim == 1: return np.array(_fl(P[0], P[1], P[2], P[3], P[4], u[0], u[1], float(t), CD, M_, A_, FM, WS))
    return np.array([_fl(p[0], p[1], p[2], p[3], p[4], u[0], u[1], float(t), CD, M_, A_, FM, WS) for p in P.reshape(-1, 5)]).reshape(P.shape)
clr = lambda P: WALL - np.maximum(np.abs(P[..., 0]), np.abs(P[..., 1]))
blocked = lambda P: clr(np.asarray(P)) < PAD
okf = lambda p: (abs(p[0]) <= WALL + .1) and (abs(p[1]) <= WALL + .1) and (abs(p[3]) <= VM) and (abs(p[4]) <= VM)
wr = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
CONE = float(os.environ.get('CONE', .5))   # скорость на воротах не против следующего перегона: v·e ≥ CONE·|v| (иначе жадность упирается в стену — угол)
VN = float(os.environ.get('VN', .25))   # нормальная скорость К стене на воротах ≤ VN (ворота у стены: иначе не успеть повернуть)
def vc(P, e, g=None):
    r = P[..., 3] * e[0] + P[..., 4] * e[1] - CONE * np.hypot(P[..., 3], P[..., 4])
    if g is not None:
        for a in (0, 1):
            if abs(g[a]) > .9: r = np.minimum(r, VN - np.sign(g[a]) * P[..., 3 + a])
    return r
def mkS(g, e):
    ing = lambda P: (np.hypot(P[..., 0] - g[0], P[..., 1] - g[1]) <= RG - GM) & (vc(P, e, g) >= 0)
    rng = np.random.default_rng(0); seeds = []
    while len(seeds) < 32:
        r = (RG - GM) * np.sqrt(rng.uniform()); ph = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0, 1.6); sa = np.arctan2(e[1], e[0]) + rng.uniform(-1, 1) * np.arccos(CONE)
        # (ограничение VN проверяется ниже)
        p = np.array([g[0] + r * np.cos(ph), g[1] + r * np.sin(ph), rng.uniform(-np.pi, np.pi), sp * np.cos(sa), sp * np.sin(sa)])
        if not blocked(p) and vc(p, e, g) >= 0: seeds.append(p)
    return SysN(flow, L, (1.2, 1.2, np.pi, VM, VM), ing, seeds, per=(0, 0, 2 * np.pi, 0, 0), ok=okf, blocked=blocked)
# --- проверка настоящим MuJoCo (ENV SafetyPointCircle2: 10 подшагов .002 на шаг .02)
_ENV = {}
def mj():
    if not _ENV:
        import warnings; warnings.filterwarnings('ignore'); import safety_gymnasium, mujoco
        e = safety_gymnasium.make('SafetyPointCircle2-v0'); e.reset(seed=1); _ENV.update(e=e, m=e.unwrapped.task.model, d=e.unwrapped.task.data, mu=mujoco)
    return _ENV
def mj_exec(x0, seq, dts, g, tmax=12.):
    """Выполнить план в MuJoCo из x0 (5D); u₁=±1 → полная тяга, u₂ → ω через velocity-актуатор. → (T до входа в окно ворот, касание стены, конечное 6D-состояние в этот момент)"""
    E = mj(); m, d, mu = E['m'], E['d'], E['mu']; d.qpos[:3] = x0[:3] * [1, 1, 1]; d.qpos[0], d.qpos[1], d.qpos[2] = x0[0], x0[1], x0[2]; d.qvel[:] = [x0[3], x0[4], 0.]; mu.mj_forward(m, d)
    ends = np.cumsum(dts); T = 0.; k = 0; tch = False; ntot = int(round(ends[-1] / .02))
    for step in range(ntot + 150):
        while k < len(seq) - 1 and T >= ends[k] - 1e-9: k += 1
        u = U_[seq[k]] if T < ends[-1] - 1e-9 else U_[seq[-1]]; d.ctrl[:] = [u[0], u[1]]
        for _ in range(10): mu.mj_step(m, d)
        T += .02; x, y = d.qpos[0], d.qpos[1]
        if max(abs(x), abs(y)) > WALL: tch = True
        if np.hypot(x - g[0], y - g[1]) <= RG: return T, tch, np.array([x, y, d.qpos[2], d.qvel[0], d.qvel[1]])
        if T > tmax: break
    return None, tch, None
def run_circle(args):
    gi_seq, NBb, NFf, Kk = args
    t_all = 0.; tq = []; x = None; res = []; tch_any = False
    th0 = ang[0] + np.pi / 2   # старт у ворот 0, v=0, курс по касательной (против часовой)
    x = np.array([GATES[0, 0], GATES[0, 1], th0, 0., 0.])
    for j in range(NG):
        g = GATES[(j + 1) % NG]; e = GATES[(j + 2) % NG] - g; e = e / np.hypot(*e); S = mkS(g, e); back = build_back(S, TAU, NBb, .15)
        gx = lambda z, g=g, e=e: np.array([(RG - GM) ** 2 - ((z[0] - g[0]) ** 2 + (z[1] - g[1]) ** 2), vc(z, e, g)])
        mx = lambda Z, g=g, e=e: np.maximum(np.hypot(Z[..., 0] - g[0], Z[..., 1] - g[1]) - (RG - GM), 0) + np.maximum(-vc(Z, e, g), 0)
        t1 = time.perf_counter()
        b = corridor_query(S, flow, x, back, TAU, NFf, .15, gx, mx, K=Kk, tries=3, kn=8, neigh=2, nbest=1, clear=lambda P: clr(P) - PAD, wlim=None, fb=(6400, 5))
        dq = time.perf_counter() - t1; tq.append(dq)
        if b is None: res.append(dict(j=j, why='no_path', t_query=dq, x=list(map(float, x)))); break
        seq, dts = list(b[1]), np.array(b[2], float); e = np.round(np.cumsum(dts) / .02) * .02; dqq = np.diff(np.concatenate([[0.], e])); kk = dqq > 1e-9; seq = [q for q, z in zip(seq, kk) if z]; dts = dqq[kk]
        Tm, tch, xe = mj_exec(x, seq, list(dts), g)
        res.append(dict(j=j, T_agent=float(b[0]), T_mj=Tm, touch=bool(tch), t_query=dq, nseg=len(seq)))
        tch_any |= tch
        if Tm is None: res[-1]['why'] = 'miss'; break
        t_all += Tm; x = xe
    return dict(res=res, T_circle=t_all if len(res) == NG and all(r.get('T_mj') for r in res) else None, touch=tch_any, tq=tq)
if __name__ == '__main__':
    t0 = time.perf_counter(); out = run_circle((None, NB, NF, K)); print('круг:', {k: v for k, v in out.items() if k != 'res'}, 'wall %.1f' % (time.perf_counter() - t0))
    for r in out['res']: print(r)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True); json.dump(out, open(os.path.join(HERE, 'data', 'pcircle%s.json' % tag), 'w'), indent=1, default=float)
