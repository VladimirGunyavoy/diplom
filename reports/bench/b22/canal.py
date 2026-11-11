"""b22 КАНАВА круга (V круговой канавой), вариант 2: 5D динамика (x,y,θ,vx,vy; боковой занос — первого порядка, 4D без него ошибается на .37 м/2 с) сведена симметрией
вращения к 4D состоянию кольца (ρ, α=θ−φ, v_рад, v_кас); сетка 4D + итерация Беллмана (numba): V = время дойти до дна M={ρ≈R*, v_кас≈v*, v_рад≈0, курс ≈ по касательной}; стена — круг ρ ≤ 1.125−MARG (вписан в квадрат, строже).
Агент — жадный по V с прогнозом .1 с (5D-модель), исполнение — SafetyPointCircle2 в MuJoCo; метрика их: недисконт. return + cost за эпизод 500 шагов, обёртка: |x|или|y|>1.125 → cost 1, награда −1, обрыв.
Запуск: runpc.sh canal.py sweep|eval|one [R v]"""
import os, sys, time, json, warnings; warnings.filterwarnings('ignore')
import numpy as np, numba as nb
M_, A_, CD, FM, WS = .005188790204786391, .0192723151357623, .01, .015, 3.0
WALL = 1.125; MARG = float(os.environ.get('MARG', .06)); RMAXW = WALL - MARG; TAU = .1; NSUB = 5; H = TAU / NSUB
NR, NA, NVR, NVT = 28, 48, 33, 33
RS = np.linspace(0, RMAXW, NR); DR = RS[1] - RS[0]; DA = 2 * np.pi / NA; VMX = 1.6; VRS = np.linspace(-VMX, VMX, NVR); DVR = VRS[1] - VRS[0]; VTS = VRS.copy()
ACT = np.array([(a, b) for a in (1., -1., 0.) for b in (1., -1., 0.)])
@nb.njit
def f5(x, y, th, vx, vy, u0, u1, h, n):
    w = WS * u1; F = FM * u0
    for _ in range(n):
        c = np.cos(th); s = np.sin(th); ax = (F * c - CD * vx) / M_ + A_ * w * w * c; ay = (F * s - CD * vy) / M_ + A_ * w * w * s
        vxm = vx + .5 * h * ax; vym = vy + .5 * h * ay; thm = th + .5 * h * w; cm = np.cos(thm); sm = np.sin(thm)
        axm = (F * cm - CD * vxm) / M_ + A_ * w * w * cm; aym = (F * sm - CD * vym) / M_ + A_ * w * w * sm
        x += h * vxm; y += h * vym; th += h * w; vx += h * axm; vy += h * aym
    return x, y, th, vx, vy
@nb.njit
def to_ring(x, y, th, vx, vy):
    rho = np.hypot(x, y); phi = np.arctan2(y, x); c = np.cos(phi); s = np.sin(phi)
    return rho, (th - phi) % (2 * np.pi), vx * c + vy * s, -vx * s + vy * c
@nb.njit
def step_ring(rho, al, vr, vt, u0, u1):
    x, y, th, vx, vy = f5(rho, 0., al, vr, vt, u0, u1, H, NSUB); return to_ring(x, y, th, vx, vy)
@nb.njit
def interp(V, rho, al, vr, vt):
    fr = min(max(rho / DR, 0.), NR - 1.001); fa = (al % (2 * np.pi)) / DA; fvr = min(max((vr + VMX) / DVR, 0.), NVR - 1.001); fvt = min(max((vt + VMX) / DVR, 0.), NVT - 1.001)
    i = int(fr); k = int(fa); m = int(fvr); n = int(fvt); ar = fr - i; aa = fa - k; ab = fvr - m; ac = fvt - n; k1 = (k + 1) % NA; r = 0.
    for di in range(2):
        for dk in range(2):
            for dm in range(2):
                for dn in range(2):
                    r += (ar if di else 1 - ar) * (aa if dk else 1 - aa) * (ab if dm else 1 - ab) * (ac if dn else 1 - ac) * V[i + di, k1 if dk else k, m + dm, n + dn]
    return r
@nb.njit(parallel=True)
def succ_all():
    S = np.zeros((NR, NA, NVR, NVT, 9, 4))
    for i in nb.prange(NR):
        for k in range(NA):
            for m in range(NVR):
                for n in range(NVT):
                    for a in range(9):
                        r, al, vr, vt = step_ring(RS[i] + 1e-6, k * DA, VRS[m], VTS[n], ACT[a, 0], ACT[a, 1])
                        S[i, k, m, n, a, 0] = r; S[i, k, m, n, a, 1] = al; S[i, k, m, n, a, 2] = vr; S[i, k, m, n, a, 3] = vt
    return S
@nb.njit(parallel=True)
def solve_(S, R, vs, tolr, tolv, tolrad, tola, nit):
    INF = 1e3; V = np.full((NR, NA, NVR, NVT), INF); M = np.zeros((NR, NA, NVR, NVT), np.bool_)
    for i in nb.prange(NR):
        for k in range(NA):
            for m in range(NVR):
                for n in range(NVT):
                    da = abs((k * DA - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
                    if abs(RS[i] - R) <= tolr and abs(VTS[n] - vs) <= tolv and abs(VRS[m]) <= tolrad and da <= tola: M[i, k, m, n] = True; V[i, k, m, n] = 0.
    for it in range(nit):
        Vn = V.copy()
        for i in nb.prange(NR):
            for k in range(NA):
                for m in range(NVR):
                    for n in range(NVT):
                        if M[i, k, m, n]: continue
                        best = INF
                        for a in range(9):
                            r = S[i, k, m, n, a, 0]
                            if r > RMAXW: continue
                            c = TAU + interp(V, r, S[i, k, m, n, a, 1], S[i, k, m, n, a, 2], S[i, k, m, n, a, 3])
                            if c < best: best = c
                        Vn[i, k, m, n] = best
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < 1e-4: break
    return V, M
@nb.njit
def policy(V, x, y, th, vx, vy, lam, ACTS):
    best = 1e9; ba = 0
    for a in range(9):
        xn, yn, thn, vxn, vyn = f5(x, y, th, vx, vy, ACTS[a, 0], ACTS[a, 1], H, NSUB)
        rho, al, vr, vt = to_ring(xn, yn, thn, vxn, vyn)
        if rho > RMAXW: continue
        rw = .1 * vt / (1. + abs(rho - 1.5)) * NSUB
        c = TAU + interp(V, rho, al, vr, vt) - lam * rw
        if c < best: best = c; ba = a
    return ba
def episode(env, V, seed, lam=.3, T=500):
    t = env.unwrapped.task; env.reset(seed=seed); ret = 0.; cost = 0; term = False
    for step in range(T):
        d = t.data; p = t.agent.pos; R_ = t.agent.mat; th = np.arctan2(R_[1, 0], R_[0, 0]); th0 = th - d.qpos[2]; c0, s0 = np.cos(th0), np.sin(th0)
        vx = c0 * d.qvel[0] - s0 * d.qvel[1]; vy = s0 * d.qvel[0] + c0 * d.qvel[1]   # мировая скорость (qvel — в рамке, повёрнутой на θ0)
        a = policy(V, p[0], p[1], th, vx, vy, lam, ACT); _, r, c, te, tr, _ = env.step(ACT[a]); p = t.agent.pos
        if max(abs(p[0]), abs(p[1])) > WALL: ret += -1.; cost += 1; term = True; break
        ret += float(r)
    return ret, cost, term
def mkenv():
    import safety_gymnasium; e = safety_gymnasium.make('SafetyPointCircle2-v0'); e.reset(seed=1); return e
if __name__ == '__main__':
    mode = sys.argv[1]; t0 = time.perf_counter(); S = succ_all(); print('succ %.1f с' % (time.perf_counter() - t0), flush=True)
    env = mkenv(); TR, TV, TRAD, TA = [float(os.environ.get(k, d)) for k, d in (('TOLR', .1), ('TOLV', .2), ('TOLRAD', .2), ('TOLA', .6))]; LAM = float(os.environ.get('LAM', .3)); out = {}
    pairs = [(float(sys.argv[2]), float(sys.argv[3]))] if mode in ('one', 'eval') else [(R, v) for R in (.7, .8, .9, 1.0) for v in (.6, .9, 1.2, 1.5)]
    seeds = list(range(200000, 200020)) if mode == 'eval' else list(range(100000, 100010))
    for R, vs in pairs:
        t1 = time.perf_counter(); V, M = solve_(S, R, vs, TR, TV, TRAD, TA, 400); fin = V[V < 1e2]
        rs = [episode(env, V, s, LAM) for s in seeds]; ret = np.array([r[0] for r in rs]); cs = sum(r[1] for r in rs)
        print('R*=%.1f v*=%.1f: return %.2f±%.2f cost %d обрывов %d V мед %.2f доля V<inf %.2f solve+eval %.0f с' % (R, vs, ret.mean(), ret.std(), cs, sum(r[2] for r in rs), np.median(fin), len(fin) / V.size, time.perf_counter() - t1), flush=True)
        out['%.1f_%.1f' % (R, vs)] = dict(ret=ret.tolist(), cost=cs, term=[r[2] for r in rs])
    json.dump(out, open('canal_%s_%s.json' % (mode, os.environ.get('TAG', '')), 'w'))
