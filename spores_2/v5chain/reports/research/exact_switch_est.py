"""research hub-v5chain-research-8: точный момент переключения на маятнике (слово пользователя: «не размазывание, а точный момент»).
Маятник φ̈ = sin φ + u, |u| ≤ .3, цель |φ|,|ω| ≤ .1 (верх), чистое время, V — сетка HJB (pend_cost_grid.Grid, bang ±1).
Агенты (управление всегда чистое ±UM):
  P0 — по шагам: каждые dt слой argmin [dt + V(φ_k(q,dt))] (как раньше; дребезг у сепаратрисы и цели).
  P1 — событие сопряжённой переменной: p = ∇V(q) (центр. разности δ), u = −sign(p_ω)·UM, (q, p) интегрируются вместе
       (ṗ_φ = −p_ω cos φ, ṗ_ω = −p_φ); переключение — в момент нуля p_ω на подшаге (линейная интерполяция). p заново из V
       только «вдали» от переключений: после прошлого ≥ τ_f и до следующего предсказанного ≥ τ_f (у кривой ∇V с изломом).
  P3 — P2, но финиш пересчитывается каждые R с (обратная связь: подход к верху — устойчивое многообразие седла, ошибка ×e^t).
  P2 — P1 + точный финиш: как только хоть один слой в одиночку доходит до цели (T1_k конечно), считаем
       min( T1_k(q), min_τ [τ + T1_k'(φ_k(q,τ))] ) — двухдуговая стрельба по сетке τ + уточнение, и исполняем её открыто.
"""
import numpy as np, json, sys, time, os
GAIN = float(os.environ.get('GAIN', 1))                                    # исполнение: u·GAIN (неточность модели), план — по модели
sys.path.insert(0, '.')
from pend_cost_grid import Grid, step, wrap, UM

def fz(z, u):
    x, w, px, pw = z
    return np.stack([w, np.sin(x) + u, -pw * np.cos(x), -px])
def rk4z(z, u, h):
    k1 = fz(z, u); k2 = fz(z + h / 2 * k1, u); k3 = fz(z + h / 2 * k2, u); k4 = fz(z + h * k3, u)
    return z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def goal(x, w, r=.1): return (np.abs(wrap(x)) <= r) & (np.abs(w) <= r)
TF = float(os.environ.get('TF', 1.))                                        # P4: за сколько секунд до прибытия перейти на спектр
RP = float(os.environ.get('RP', .1))                                        # план финиша — в цель ±RP (запас против ошибки модели), успех — ±.1

EST = int(os.environ.get('EST', 0)); G = [1., 0, 0., 0.]                                # research-9: оценка коэффициента при u на ходу; G = [ĝ, число наблюдений]
def observe(x0, w0, x1, w1, ucmd, h):
    """ĝ из одного шага: Δω = ∫sin φ dt + ĝ·u·h (трапеция по наблюдаемым φ); скользящее среднее. Состояние без шума."""
    if not EST or abs(ucmd) < 1e-9 or h < 1e-6: return
    if NOISE: x0, w0, x1, w1 = (v + NOISE * RNG.standard_normal() for v in (x0, w0, x1, w1))          # шум датчика — только в оценке ĝ (планировщик видит точное состояние)
    d = w1 - w0 - h * (np.sin(x0) + np.sin(x1)) / 2; G[1] += 1
    if EST == 1: G[0] += (d / (ucmd * h) - G[0]) / min(G[1], 50)                                      # скользящее среднее 50 шагов
    elif EST == 3:                                                                                   # длинная база: по дуге с постоянным u — (ω_сейчас − ω_начала − ∫sin φ dt)/(u·T); дуги — МНК с весом (u·T)²
        if ARC[0] != ucmd: G[2] += ARC[4]; G[3] += ARC[5]; ARC[:] = [ucmd, w0, 0., 0., 0., 0.]       # новая дуга: закрыть старую
        ARC[2] += h * (np.sin(x0) + np.sin(x1)) / 2; ARC[3] += h; dd = w1 - ARC[1] - ARC[2]; ARC[4], ARC[5] = ucmd * ARC[3] * dd, (ucmd * ARC[3]) ** 2
        if G[3] + ARC[5] > (.3 * .3) ** 2: G[0] = float(np.clip((G[2] + ARC[4]) / (G[3] + ARC[5]), .5, 1.5))
    else: G[2] += ucmd * h * d; G[3] += (ucmd * h) ** 2; G[0] = float(np.clip(G[2] / G[3], .5, 1.5)) if G[1] >= 30 else 1.   # EST=2: накопительная МНК
ARC = [0., 0., 0., 0., 0., 0.]; NOISE = float(os.environ.get('NOISE', 0)); RNG = np.random.default_rng(7)
def T1(x, w, u, H=4.0, h=.01):
    """Время, за которое слой u в одиночку входит в цель (inf — не входит за H). Векторно; вход уточняется до h/8."""
    u = u * G[0]; x, w = np.array(x, float), np.array(w, float); T = np.full(x.shape, np.inf); t = 0.; act = ~goal(x, w, RP); T[~act] = 0.
    for _ in range(int(H / h)):
        a = np.nonzero(act)[0]
        if not len(a): break
        xn, wn = step(x[a], w[a], u, h); g = goal(xn, wn, RP)
        if g.any():                                                       # уточнить момент входа внутри шага
            ag = a[g]; xs, ws = x[ag], w[ag]; tt = np.full(len(ag), h)
            for j in range(1, 9):
                xs2, ws2 = step(x[ag], w[ag], u, h * j / 8); hit = goal(xs2, ws2, RP) & ~np.isfinite(T[ag]) & (tt == h); tt = np.where(hit, h * j / 8, tt)
            T[ag] = t + tt; act[ag] = False
        x[a], w[a] = xn, wn; t += h
    return T

def finish_plan(x, w, k, H=4.0):
    """Точный финиш из (x,w): лучший из 4 вариантов — «k1 до цели», «k1 на τ, затем −k1 до цели», k1 ∈ {k, −k}.
    Возвращает (J, k1 — слой первой дуги, τ_switch|inf)."""
    best = (np.inf, k, np.inf)
    for k1 in (k, -k):
        J1 = T1(np.array([x]), np.array([w]), k1 * UM, H)[0]
        if J1 < best[0]: best = (J1, k1, np.inf)
        taus = np.arange(0, min(H, J1 if np.isfinite(J1) else H), .02); xs, ws = [x], [w]
        for _ in taus[1:]: xn, wn = step(np.array(xs[-1]), np.array(ws[-1]), k1 * UM * G[0], .02); xs.append(float(xn)); ws.append(float(wn))
        f = taus + T1(np.array(xs), np.array(ws), -k1 * UM, H)
        if np.isfinite(f).any():
            i = int(np.nanargmin(f))
            if f[i] < best[0]:                                            # уточнить τ на ±.02 сеткой .001
                t2 = np.clip(taus[i] + np.arange(-.02, .0201, .001), 0, None); xx, ww = np.full(len(t2), x), np.full(len(t2), w)
                xx, ww = step(xx, ww, k1 * UM * G[0], t2); f2 = t2 + T1(xx, ww, -k1 * UM, H); j = int(np.argmin(f2))
                best = (f2[j], k1, t2[j]) if f2[j] < f[i] else (f[i], k1, taus[i])
    return best

EPS = float(os.environ.get('EPS', 0))                                       # >0: план принимается, только если доходит и при u·(1±EPS)
def robust_ok(x, w, k1, ts, J, h=.01):
    if EPS <= 0: return True
    for g in (1 - EPS, 1 + EPS):
        xx, ww, t = np.array([x]), np.array([w]), 0.
        while t < J * 1.2 + .3:
            if goal(xx, ww)[0]: break
            kk = k1 if t < ts else -k1; xx, ww = step(xx, ww, kk * UM * g * G[0], h); t += h
        else: return False
        if not goal(xx, ww)[0]: return False
    return True

def run(S, Q, mode, dt=.03, delta=.05, tau_f=.15, h=.003, tmax=40., R=np.inf):
    n = len(Q); T = np.full(n, np.inf); SW = np.zeros(n, int); out = []
    for i in range(n):
        G[:] = [1., 0, 0., 0.]; ARC[:] = [0.] * 6; x, w = Q[i]; t = 0.; k = None; sw = 0; last_sw = 1e9; plan = None; pz = None; last_fp = -1e9; p0next = False; pu4 = None
        while t < tmax:
            if goal(x, w): T[i] = t; break
            if mode == 'P0' or (mode != 'P0' and pz is None and False):
                pass
            if plan is not None:                                          # открытый финиш: τ_switch, потом до цели
                ts, kk, tm, tarr = plan
                if mode == 'P4' and tarr - t < TF:                               # P4: последняя секунда — обратная связь спектром каждые dt
                    US = np.linspace(-1, 1, 11); c = [dt + S.Vq(*step(np.array([x]), np.array([w]), uu * UM, dt))[0] for uu in US]; uu = US[int(np.argmin(c))]
                    if pu4 is not None and abs(uu - pu4) > .5: sw += 1
                    pu4 = uu
                    for _ in range(int(round(dt / h))):
                        xn, wn = step(np.array([x]), np.array([w]), uu * UM * GAIN, h); x, w = float(xn[0]), float(wn[0]); t += h
                        if goal(x, w): break
                    continue
                if t >= ts and kk == k: k = -k; sw += 1
                if t - tm >= R:                                                   # P3: пересчёт точного финиша каждые R с
                    J, k1, ts2 = finish_plan(x, w, k)
                    if J <= S.Vq(np.array([x]), np.array([w]))[0] + .1 and robust_ok(x, w, k1, ts2, J):   # та же проверка + устойчивость к ошибке модели
                        if k1 != k: k = k1; sw += 1
                        plan = (t + ts2, k, t, t + J)
                    else: plan, pz, p0next = None, None, True; continue         # ушли из трубки финиша — шаг по V (как P0), потом снова финиш
                xn, wn = step(np.array([x]), np.array([w]), k * UM * GAIN, h); observe(x, w, float(xn[0]), float(wn[0]), k * UM, h); x, w = float(xn[0]), float(wn[0]); t += h; continue
            if mode == 'P0' or p0next:
                if p0next: p0next = False; last_fp = -1e9
                c = [dt + S.Vq(*step(np.array([x]), np.array([w]), u * UM, dt))[0] for u in (1., -1.)]; kn = 1. if c[0] <= c[1] else -1.
                if k is not None and kn != k: sw += 1
                k = kn
                for _ in range(int(round(dt / h))):
                    xn, wn = step(np.array([x]), np.array([w]), k * UM * GAIN, h); observe(x, w, float(xn[0]), float(wn[0]), k * UM, h); x, w = float(xn[0]), float(wn[0]); t += h
                    if goal(x, w): break
                continue
            # P1/P2: сопряжённая переменная
            if pz is None or (t - last_sw >= tau_f and pz[1] >= tau_f):
                gx = (S.Vq(np.array([x + delta]), np.array([w]))[0] - S.Vq(np.array([x - delta]), np.array([w]))[0]) / (2 * delta)
                gw = (S.Vq(np.array([x]), np.array([w + delta]))[0] - S.Vq(np.array([x]), np.array([w - delta]))[0]) / (2 * delta)
                p = np.array([gx, gw])
                if k is None: k = -np.sign(gw) or 1.
                elif -np.sign(gw) != k and abs(gw) > 1e-9: k = -k; sw += 1; last_sw = t
                z = np.array([x, w, p[0], p[1]]); zz = z.copy(); tpred = np.inf           # предсказать следующий ноль p_ω
                for j in range(int(1.0 / .01)):
                    zn = rk4z(zz, k * UM, .01)
                    if np.sign(zn[3]) == np.sign(k): tpred = (j + 1) * .01; break         # u = −sign(p_ω): переключение, когда sign(p_ω) = sign(k)
                    zz = zn
                pz = [p, tpred]
            if mode in ('P2', 'P3', 'P4') and t - last_fp >= .15:
                last_fp = t; J, k1, ts = finish_plan(x, w, k)
                if J <= S.Vq(np.array([x]), np.array([w]))[0] + .1 and robust_ok(x, w, k1, ts, J):   # финиш не хуже V и доходит при u·(1±EPS)
                    if k1 != k: k = k1; sw += 1
                    plan = (t + ts, k, t, t + J); continue
            z = np.array([x, w, pz[0][0], pz[0][1]]); moved = 0.
            while moved < dt - 1e-12:
                zn = rk4z(z, k * UM * GAIN, h)
                if np.sign(zn[3]) == np.sign(k) and np.sign(z[3]) != np.sign(k):       # ноль p_ω внутри подшага → точный момент
                    s = h * z[3] / (z[3] - zn[3]); z = rk4z(z, k * UM * GAIN, s); k = -k; sw += 1; last_sw = t + moved + s; moved += s; continue
                observe(z[0], z[1], zn[0], zn[1], k * UM, h); z = zn; moved += h
                if goal(z[0], z[1]): break
            x, w, pz[0] = float(z[0]), float(z[1]), z[2:].copy(); t += moved; pz[1] = max(pz[1] - moved, 0) if np.isfinite(pz[1]) else np.inf
        SW[i] = sw
    return T, SW

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(0); Q = np.stack([rng.uniform(-np.pi, np.pi, 100), rng.uniform(-2, 2, 100)], 1)[:N]
    if os.environ.get('BOTTOM'): Q = np.stack([np.pi + rng.uniform(-.5, .5, N), rng.uniform(-.3, .3, N)], 1)   # у низа: долгая раскачка
    t0 = time.time(); S = Grid(); S.solve((-1., 1.), 0.); print('grid', round(time.time() - t0), 'iter', S.n, flush=True)
    res = {}
    modes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['P0', 'P1', 'P2']
    for mode, kw in [(m.split(':')[0], dict(R=float(m.split(':')[1])) if ':' in m else {}) for m in modes]:
        t0 = time.time(); T, SW = run(S, Q, mode, **kw); mode = mode + (':%g' % kw['R'] if kw else ''); res[mode] = (T, SW)
        print(json.dumps(dict(mode=mode, reach=float(np.isfinite(T).mean()), T_mean=round(float(np.mean(T[np.isfinite(T)])), 3),
                              sw_med=float(np.median(SW)), sw_max=int(SW.max()), sec=round(time.time() - t0))), flush=True)
    T0 = res['P0'][0]
    for m in [m for m in res if m != 'P0']:
        ok = np.isfinite(T0) & np.isfinite(res[m][0]); r = res[m][0][ok] / T0[ok]
        print(json.dumps(dict(mode=m, vs_P0=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 3), max=round(float(r.max()), 3)))), flush=True)
    print('g_hat последнего старта', round(G[0], 5), 'наблюдений', G[1]); np.save('exact_switch_est%d_eps%g_n%g_%s%s%s%s.npy' % (EST, EPS, NOISE, '_'.join(res).replace(':', ''), '_g%s' % GAIN if GAIN != 1 else '', '_bot' if os.environ.get('BOTTOM') else '', '_rp%g' % RP if RP != .1 else ''), np.array([res[m][0] for m in res] + [res[m][1] for m in res]))
