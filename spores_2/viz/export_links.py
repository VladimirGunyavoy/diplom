"""линия D: экспорт данных визуализации (viz/data/<система>.json) для систем со звеньями. ТРЕБУЕТ код спор (запускается на aida, не на ноуте).
python export_links.py pend [старты через запятую] — споры v7 grow_cells2d (как bench/spores_pend.py) + эталон OCP CasADi (min T, мультистарт по ветвям 2π);
траектории пересчитываются единым проверяльщиком (common.Env.step, RK4) и записываются с постоянным шагом DT=.02 с. aida: ~/spore_v5/d1/spores_2 (sync_lw.sh), venv-sb3 + PYTHONPATH=~/spore_v5/r5/pylib."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(HERE, '..', 'bench'); sys.path.insert(0, BENCH)
import numpy as np
DT = .02; OUT = os.path.join(HERE, 'data')

def resample(env, y0, us, dts, dt=DT, h=.01):
    """(u_k, dt_k) → X на сетке dt (включая t=0 и конец), U на шагах; конец = момент первого попадания в цель (T) либо конец плана"""
    y = np.asarray(y0, float)[None].copy(); X = [y[0].copy()]; U = []; t = 0.; nxt = dt; T = None; cur_u = None
    def adv(y, u, d):
        k = max(1, int(np.ceil(d / h - 1e-9))); hh = d / k
        for _ in range(k): y = env.step(y, u, hh)
        return y
    if env.in_goal(y)[0]: return np.array(X), np.zeros((1, env.m)), 0.
    for u, d in zip(us, dts):
        u = env.clip_u(np.asarray(u, float).reshape(1, -1)); rem = float(d)
        while rem > 1e-12:
            step = min(rem, nxt - t); y2 = adv(y, u, step); tn = t + step
            if env.in_goal(y2)[0]:                                    # первое попадание внутри шага — уточняем делением
                lo, hi = 0., step
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if env.in_goal(adv(y, u, mid) if mid > 0 else y)[0]: hi = mid
                    else: lo = mid
                T = t + hi; y2 = adv(y, u, hi); X.append(y2[0].copy()); U.append(u[0].copy()); return np.array(X), np.array(U), T
            if env.out_of_box(y2)[0]: return np.array(X), np.array(U), None
            y = y2; t = tn; rem -= step
            if abs(t - nxt) < 1e-9: X.append(y[0].copy()); U.append(u[0].copy()); nxt += dt
    return np.array(X), np.array(U), T

def ocp_pend(env, x0, N=60, seeds=6, guess=None):
    import casadi as ca
    um = float(env.umax[0]); T = ca.SX.sym('T'); U = ca.SX.sym('U', N); X = ca.SX.sym('X', 2, N); x0s = ca.SX.sym('x0', 2); tf = ca.SX.sym('tf'); h = T / N / 4; g = []; z = x0s
    f = lambda z, u: ca.vertcat(z[1], ca.sin(z[0]) + u)
    for k in range(N):
        for _ in range(4):
            k1 = f(z, U[k]); k2 = f(z + h / 2 * k1, U[k]); k3 = f(z + h / 2 * k2, U[k]); k4 = f(z + h * k3, U[k]); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k < N - 1: g.append(z - X[:, k]); z = X[:, k]
    G = ca.vertcat(*g, z[0] - tf, z[1]); Gf = ca.Function('Gf', [ca.vertcat(T, U, ca.vec(X[:, :N - 1])), ca.vertcat(x0s, tf)], [G]); S = ca.nlpsol('S', 'ipopt', dict(x=ca.vertcat(T, U, ca.vec(X[:, :N - 1])), f=T, g=G, p=ca.vertcat(x0s, tf)), {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 300, 'ipopt.sb': 'yes'})
    best = None; rng = np.random.default_rng(0); hf = env.half; cands = []
    if guess is not None:                                           # тёплый старт: план споры, пересэмплированный на N равных интервалов (min T на swing-up без него не сходится — Infeasible)
        us, dts = guess; T0 = float(sum(dts)); tt = (np.arange(N) + .5) * T0 / N; ends = np.cumsum(dts); U0 = np.array([us[min(int(np.searchsorted(ends, t)), len(us) - 1)][0] for t in tt]); y = np.array(x0, float); Xn = []; fp = lambda z, u: np.array([z[1], np.sin(z[0]) + u]); hh = T0 / N / 4          # без обёртки угла (в NLP θ развёрнут)
        for k in range(N - 1):
            for _ in range(4): k1 = fp(y, U0[k]); k2 = fp(y + hh / 2 * k1, U0[k]); k3 = fp(y + hh / 2 * k2, U0[k]); k4 = fp(y + hh * k3, U0[k]); y = y + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Xn.append(y.copy())
        cands.append((float(np.round(np.round((Xn[-1][0]) / (2 * np.pi)) * 2 * np.pi, 6)), T0, U0, np.array(Xn).T))
    for tfin in (0., 2 * np.pi, -2 * np.pi):
        for s in range(seeds): cands.append((tfin, rng.uniform(3, 14), rng.uniform(-um, um, N), np.linspace(x0, [tfin, 0], N + 1)[1:N].T))
    for tfin, T0, U0, X0 in cands:
        if True:
            v0 = np.r_[T0, U0, X0.ravel(order='F')]
            if os.environ.get('VIZDBG'): rr = np.array(Gf(v0, np.r_[x0, tfin])).ravel(); print('  resid', np.abs(rr[:-2]).max(), rr[-2:], flush=True)
            lb = np.r_[1e-2, -um * np.ones(N), -1e3 * np.ones(2 * (N - 1))]; ub = np.r_[40., um * np.ones(N), 1e3 * np.ones(2 * (N - 1))]
            lbg = np.r_[np.zeros(2 * (N - 1)), -hf * .9]; ubg = np.r_[np.zeros(2 * (N - 1)), hf * .9]
            r = S(x0=v0, p=np.r_[x0, tfin], lbx=lb, ubx=ub, lbg=lbg, ubg=ubg); v = np.array(r['x']).ravel()
            if os.environ.get('VIZDBG'): print('  ocp', tfin, s, S.stats()['return_status'], round(float(v[0]), 3), flush=True)
            if S.stats()['return_status'] in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and (best is None or v[0] < best[0]): best = (float(v[0]), v[1:1 + N].copy())
    return best

def export_pend(envn, starts):
    import common
    os.environ['VARIANT'] = '56' if envn == 'pend' else '84'; srcf = 'pend_spores_T.json' if envn == 'pend' else 'pend15_spores.json'          # главная строка бенча (compare.md): pend — LOOK 20 (мед 1.016), pend15 — основная
    e = common.make(envn); import spores_pend as SP; G = SP.G
    t0 = time.perf_counter(); A = G.Atlas(); A.solve(); print('атлас', time.perf_counter() - t0, flush=True)
    eps = []
    for i in starts:
        T, rec = SP.rollout_rec(A, e.X0[i:i + 1]); us = [[r[0][0]] for r in rec if r[2][0]]; dts = [float(r[1][0]) for r in rec if r[2][0]]
        X, U, Tv = resample(e, e.X0[i], us, dts)
        eps.append(dict(method='споры', start=int(i), dt=DT, X=np.round(X, 5).tolist(), U=np.round(U, 4).tolist(), T=Tv, T_ref=float(e.Tref[i]), ok=Tv is not None, src=srcf))
        b = ocp_pend(e, e.X0[i], guess=(us, dts))
        if b is not None:
            Tb, Ub = b; Xr, Ur, Tr = resample(e, e.X0[i], [[u] for u in Ub], [Tb / len(Ub)] * len(Ub))
            eps.append(dict(method='эталон (OCP)', start=int(i), dt=DT, X=np.round(Xr, 5).tolist(), U=np.round(Ur, 4).tolist(), T=Tr, T_ref=float(e.Tref[i]), ok=Tr is not None, src='ocp_pend (CasADi, тёплый старт от плана спор); T_ref — pend_ref_T.npy'))
        print('старт', i, 'споры', eps[-2 if b else -1]['T'], 'эталон', eps[-1]['T'] if b else None, 'Tref', e.Tref[i], flush=True)
    um = float(e.umax[0]); title = 'Маятник, |u| ≤ %g (θ=0 — вверх)' % um
    data = dict(sys=envn, title=title, params=dict(L=1.0, umax=[um], view=[-1.9, -1.35, 1.9, 1.5], unames=['момент']), goal=dict(theta=float(e.half[0])), obstacles=[], episodes=eps)
    os.makedirs(OUT, exist_ok=True); json.dump(data, open(os.path.join(OUT, envn + '.json'), 'w'), ensure_ascii=False, separators=(',', ':'))
    print('записано', envn, os.path.getsize(os.path.join(OUT, envn + '.json')), 'байт')

if __name__ == '__main__':
    sysn = sys.argv[1]; st = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    if sysn in ('pend', 'pend15'):
        os.environ['ENVN'] = sysn; export_pend(sysn, st or [5, 12, 10, 15, 2, 18])
