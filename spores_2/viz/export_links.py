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

def _refwork(i):
    best, _ = _OL.solve_query(_E.X0[i], 50, 4, 4, seed=i, solver=_SOL, nearest=False); return i, best
VIEWS = dict(m3gd=[-3.3, -3.2, 3.3, 3.2], m3gu=[-3.3, -3.2, 3.3, 3.2], m3d=[-3.7, -1.8, 3.7, 1.9])
def export_m3(envn, starts=None, nshow=5):
    """6D: m3d / m3gd / m3gu — споры = коридор v6 как bench/spores_6d.py (финальные конфиги из bench_6d8d.md: *_spores6d_final.json), эталон — OCP CasADi (m3d: готовые refm3d_*.npy, m3g: ocp_m3g_lib multi)"""
    import common
    cfg = dict(m3d=(12000, 1000, 3, False, '0'), m3gd=(12000, 3000, 3, True, '2'), m3gu=(12000, 3000, 3, True, '2'))[envn]; NB, NF, K, wl, neigh = cfg
    if envn != 'm3d': os.environ['G'] = '0.3'; os.environ['DOWN' if envn == 'm3gd' else 'UP'] = '1'          # ДО импорта ocp_m3g_lib (цель C3 читается при импорте; refine_ca его тянет внутри spores_6d)
    os.environ.update(CAREF='1', TREEFAST='1', BATCHFAST='1', CAMARG='.004', CAPRE='12', NEIGH=neigh)
    sys.argv = ['spores_6d.py', envn, str(NB), str(NF), str(K), ''] + (['wlim'] if wl else [])
    sys.path.insert(0, os.path.join(BENCH)); import spores_6d as SP
    e = SP.e; us_all = {}
    def query(i):
        b = SP.corridor_query(SP.S, SP.fl, e.X0[i], SP.back, SP.tau, NF, SP.rho, SP.g, SP.miss, K=K, tries=3, kn=8, neigh=int(neigh), nbest=1, wlim=((3, 4, 5), SP.WM) if wl else None)
        return None if b is None else ([SP.US_[s_] for s_ in b[1]], list(b[2]))
    res = {}
    for i in range(len(e.X0)):
        q = query(i)
        if q is None: continue
        X, U, Tv = resample(e, e.X0[i], q[0], q[1])
        if Tv is not None: res[i] = (X, U, Tv, e.Tref[i])
        print('споры', i, Tv, 'Tref', round(float(e.Tref[i]), 3), flush=True)
    if starts is None:
        good = [i for i, r in res.items() if r[2] / r[3] <= 1.2 and e.keep[i]]; starts = sorted(good, key=lambda i: -e.Tref[i])[:nshow]
    print('показываем старты', starts, flush=True)
    refs = {}
    if envn == 'm3d':
        for i in starts:
            a = np.load(os.path.join(common.DATA, 'm3d_ref', 'refm3d_m3d_q%d_G0_N50_W3.npy' % (i + 1))); T = float(a[0]); U = np.asarray(a[1:], float).reshape(50, 3) if a.size == 151 else np.asarray(a[1], float); refs[i] = (T, U)
    else:
        import ocp_m3g_lib as OL; assert abs(OL.C3[0] - (-np.pi / 2 if envn == 'm3gd' else np.pi / 2)) < 1e-9; from multiprocessing import get_context
        global _OL, _SOL, _E; _OL, _SOL, _E = OL, OL.make_solver(50, 4), e
        with get_context('fork').Pool(len(starts)) as pool:
            for i, best in pool.map(_refwork, starts):
                if best is not None: refs[i] = (best[0], best[1])
    eps = []; srcf = '%s_spores6d_final.json' % envn
    for i in starts:
        X, U, Tv, Tr_ = res[i]; eps.append(dict(method='споры', start=int(i), dt=DT, X=np.round(X, 4).tolist(), U=np.round(U, 3).tolist(), T=Tv, T_ref=float(Tr_), ok=True, src=srcf))
        if i in refs:
            T, Ur = refs[i]; Xr, Ur2, Tvr = resample(e, e.X0[i], list(Ur), [T / len(Ur)] * len(Ur))
            eps.append(dict(method='эталон (OCP)', start=int(i), dt=DT, X=np.round(Xr, 4).tolist(), U=np.round(Ur2, 3).tolist(), T=Tvr, T_ref=float(Tr_), ok=Tvr is not None, src='OCP CasADi multiple shooting N50 (%s)' % ('refm3d npy' if envn == 'm3d' else 'ocp_m3g_lib multi')))
    names = dict(m3d='3D двузвенник (основание + плечо + локоть), без g, |τ| ≤ 1', m3gd='3 звена, g = 0.3, цель — висит вниз, |τ| ≤ 1', m3gu='3 звена, g = 0.3, цель — стойка вверх, |τ| ≤ 1')
    goal = {} if envn == 'm3d' else dict(q=[(-np.pi / 2 if envn == 'm3gd' else np.pi / 2), 0., 0.])
    par = dict(umax=[1., 1., 1.], view=VIEWS[envn], unames=['база', 'плечо', 'локоть'] if envn == 'm3d' else ['сустав 1', 'сустав 2', 'сустав 3'])
    par.update(links=[.5, 1.] if envn == 'm3d' else [1., 1., 1.]);
    if envn != 'm3d': par['n'] = 3
    data = dict(sys=envn, title=names[envn], params=par, goal=goal, obstacles=[], episodes=eps)
    os.makedirs(OUT, exist_ok=True); json.dump(data, open(os.path.join(OUT, envn + '.json'), 'w'), ensure_ascii=False, separators=(',', ':')); print('записано', envn, os.path.getsize(os.path.join(OUT, envn + '.json')), flush=True)

if __name__ == '__main__':
    sysn = sys.argv[1]; st = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 else None
    if sysn in ('pend', 'pend15'):
        os.environ['ENVN'] = sysn; export_pend(sysn, st or [5, 12, 10, 15, 2, 18])
    elif sysn in ('m3d', 'm3gd', 'm3gu'): export_m3(sysn, st)
