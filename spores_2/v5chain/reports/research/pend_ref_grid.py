"""research-15 (слово пользователя 2026-10-06: «движок слабее — несколько сепаратрис»): эталон маятника для любого u_max.
Время-оптимальная V на мелкой сетке (полулагранжево: V = min_u dt + V(φ_u(x, dt)), билинейно, φ периодично, управления ±UM, 0),
затем агент по V с честной симуляцией rk4 (argmin_u dt + V(шаг), попадание в цель внутри шага) — время агента = реальная траектория,
верхняя оценка T*. Калибровка: UM=.3 против pend_ref_T.npy (стрельба research-9).
Запуск: UM=.15 NX=721 NW=601 python3 pend_ref_grid.py → pend_ref_grid_u0.15.npy (100 стартов как в grow_cells2d.starts_ref)."""
import numpy as np, os, time, json
from tqdm import tqdm
UM = float(os.environ.get('UM', .3)); RHO = float(os.environ.get('RHO', .1)); NX = int(os.environ.get('NX', 721)); NW = int(os.environ.get('NW', 601)); W = 4.
DT = float(os.environ.get('DT', .02)); TMAXV = float(os.environ.get('TMAXV', 60.)); BIG = 1e3; US = (-UM, 0., UM)
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
def f(x, w, u): return w, np.sin(x) + u
def rk4(x, w, u, h, n=1):
    for _ in range(n):
        k1 = f(x, w, u); k2 = f(x + h / 2 * k1[0], w + h / 2 * k1[1], u); k3 = f(x + h / 2 * k2[0], w + h / 2 * k2[1], u); k4 = f(x + h * k3[0], w + h * k3[1], u)
        x, w = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), w + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return x, w
gx = np.linspace(-np.pi, np.pi, NX, endpoint=False); gw = np.linspace(-W, W, NW); hx, hw = gx[1] - gx[0], gw[1] - gw[0]
def stencil(x, w):
    """билинейные индексы/веса на сетке (φ периодично, ω вне [−W, W] — BIG)."""
    fx = (wrap(x) + np.pi) / hx; i0 = np.floor(fx).astype(int); a = fx - i0; i0 %= NX; i1 = (i0 + 1) % NX
    fw = (w + W) / hw; j0 = np.clip(np.floor(fw).astype(int), 0, NW - 2); b = np.clip(fw - j0, 0, 1); out = (w < -W) | (w > W)
    return (i0, i1, j0, j0 + 1), (a, b), out
def interp(V, st):
    (i0, i1, j0, j1), (a, b), out = st; v = (1 - a) * (1 - b) * V[i0, j0] + a * (1 - b) * V[i1, j0] + (1 - a) * b * V[i0, j1] + a * b * V[i1, j1]
    return np.where(out, BIG, v)
def ingoal(x, w): return (np.abs(wrap(x)) <= RHO + 1e-9) & (np.abs(w) <= RHO + 1e-9)
if __name__ == '__main__':
    t0 = time.time(); X, Wg = np.meshgrid(gx, gw, indexing='ij'); goal = ingoal(X, Wg); ST = []; TG = []
    for u in US:
        xn, wn = rk4(X, Wg, u, DT / 4, 4); ST.append(stencil(xn, wn))
        tg = np.full(X.shape, np.inf); x_, w_ = X, Wg                                         # попадание в цель внутри шага (8 подшагов)
        for k in range(1, 9): x_, w_ = rk4(x_, w_, u, DT / 8); tg = np.where(np.isinf(tg) & ingoal(x_, w_), DT * k / 8, tg)
        TG.append(tg)
    V = np.full(X.shape, BIG); V[goal] = 0.
    for it in tqdm(range(int(TMAXV / DT) * 2), desc='value iteration u=%g' % UM, mininterval=float(os.environ.get('TQDM_MI', 10))):
        Vn = np.minimum.reduce([np.minimum(TG[k], DT + interp(V, ST[k])) for k in range(3)]); Vn[goal] = 0.; Vn = np.minimum(Vn, BIG)
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < 1e-7: break
    print('V: итераций', it, 'сек', round(time.time() - t0), 'недостижимо', round(float((V >= BIG / 2).mean()), 4), flush=True)
    rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)
    UL = []                                                                                   # управление на каждом шаге H — для доводки дугами (pend_ref_refine.py)
    x, w = Q[:, 0].copy(), Q[:, 1].copy(); T = np.zeros(100); done = ingoal(x, w); sw = np.zeros(100, int); pu = np.full(100, np.nan); H = DT / 2
    for _ in tqdm(range(int(TMAXV / H)), desc='agent rollout', mininterval=float(os.environ.get('TQDM_MI', 10))):
        if done.all(): break
        J = []; TGs = []
        for u in US:
            tg = np.full(100, np.inf); x_, w_ = x, w
            for k in range(1, 9): x_, w_ = rk4(x_, w_, u, H / 8); tg = np.where(np.isinf(tg) & ingoal(x_, w_), H * k / 8, tg)
            xn, wn = rk4(x, w, u, H / 4, 4); J.append(np.minimum(tg, H + interp(V, stencil(xn, wn)))); TGs.append(tg)
        J = np.stack(J, 1); k = J.argmin(1); u = np.array(US)[k]; UL.append(np.where(done, np.nan, u)); tg = np.stack(TGs, 1)[np.arange(100), k]; act = ~done
        sw += act & np.isfinite(pu) & (u != pu); pu = np.where(act, u, pu)
        hh = np.where(np.isfinite(tg), tg, H); xn, wn = x.copy(), w.copy()
        for ui in US:
            m = act & (u == ui)
            if m.any(): xn[m], wn[m] = rk4(x[m], w[m], ui, hh[m] / 4, 4)
        x, w = np.where(act, xn, x), np.where(act, wn, w); T += np.where(act, hh, 0.); done |= ingoal(x, w)
    T[~done] = np.inf; suf = '' if RHO == .1 else '_R%g' % RHO; np.save('pend_ref_grid_u%g%s.npy' % (UM, suf), T); np.savez('pend_ref_grid_u%g%s_ctl.npz' % (UM, suf), Q=Q, T=T, U=np.array(UL), H=H)
    out = dict(UM=UM, RHO=RHO, NX=NX, NW=NW, DT=DT, reach=float(done.mean()), T_med=float(np.median(T[done])), T_max=float(T[done].max()), sw_med=float(np.median(sw[done])), sec=round(time.time() - t0))
    if UM == .3 and RHO == .1 and os.path.exists('pend_ref_T.npy'):
        R = np.load('pend_ref_T.npy'); ok = np.isfinite(T) & np.isfinite(R) & (R > .05); r = T[ok] / R[ok]; out.update(vs_shoot_mean=round(float(r.mean()), 4), vs_shoot_min=round(float(r.min()), 4), vs_shoot_max=round(float(r.max()), 4))
    print(json.dumps(out), flush=True)
