"""research-15 (идея пользователя 2026-10-06): эталон маятника = раскачка энергией + финиш стрельбой по моменту переключения.
Раскачка u = UM·sign(ω·(1 − E)) (E = ω²/2 + cos φ, верх E = 1); пока |φ| ≤ ANG (30°) от верха — каждые HO с пробуем финиш стрельбой
pend_shoot_ref.shoot (≤ 3 дуг, точная проверка rk4) из текущего состояния; T = t_раскачки + T_финиша, минимум по моментам передачи.
Все кандидаты — реальные траектории (верхняя оценка T*). Запуск: UM=.15 python3 pend_ref_eshoot.py → pend_ref_eshoot_u0.15.npy."""
import numpy as np, os, sys, json, time
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pend_cost_grid as PCG
UM = float(os.environ.get('UM', .3)); PCG.UM = UM
import pend_shoot_ref as PS                                                                    # берёт UM из pend_cost_grid при импорте
ANG = np.deg2rad(float(os.environ.get('ANG', 30))); HO = float(os.environ.get('HO', .05)); TMAX = float(os.environ.get('TMAXE', 60.)); H = .005
B = None
def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
def f(y, u): return np.array([y[1], np.sin(y[0]) + u])
def rk4(y, u, h):
    k1 = f(y, u); k2 = f(y + h / 2 * k1, u); k3 = f(y + h / 2 * k2, u); k4 = f(y + h * k3, u); return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def init():
    global B; B = {s: PS.band(s * UM) for s in (1., -1.)}
def one(q):
    y = np.array(q, float); t = 0.; best = np.inf; nxt = 0.
    T0, _ = PS.shoot(y, B); best = min(best, T0)                                               # прямой финиш из старта (≤ 3 дуг)
    while t < min(TMAX, best):
        if abs(wrap(y[0])) <= ANG and t >= nxt:
            nxt = t + HO; Tf, _ = PS.shoot(np.array([wrap(y[0]), y[1]]), B); best = min(best, t + Tf)
        E = y[1] ** 2 / 2 + np.cos(y[0]); u = UM * np.sign(y[1] * (1 - E) + 1e-12); y = rk4(y, u, H); t += H
    return best
if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time(); rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)
    with Pool(int(os.environ.get('NP', 30)), initializer=init) as pool: T = np.array(list(tqdm(pool.imap(one, list(Q)), total=len(Q), desc='eshoot u=%g' % UM, mininterval=float(os.environ.get('TQDM_MI', 10)))))
    suf = '' if PS.R0 == .1 else '_R%g' % PS.R0; np.save('pend_ref_eshoot_u%g%s.npy' % (UM, suf), T)
    out = dict(UM=UM, ANG_deg=round(float(np.rad2deg(ANG)), 1), reach=float(np.isfinite(T).mean()), T_med=float(np.median(T[np.isfinite(T)])), sec=round(time.time() - t0))
    if UM == .3 and PS.R0 == .1 and os.path.exists('pend_ref_T.npy'):
        S = np.load('pend_ref_T.npy'); ok = np.isfinite(S) & np.isfinite(T) & (S > .05); r = T[ok] / S[ok]; out.update(vs_shoot_mean=round(float(r.mean()), 4), vs_shoot_min=round(float(r.min()), 4), vs_shoot_max=round(float(r.max()), 4))
    print(json.dumps(out), flush=True)
