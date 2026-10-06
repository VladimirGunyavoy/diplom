"""PLAN п.19 (hub-v5chain-worker-b1, слово пользователя 2026-10-06): хороший эталон маятника для слабого мотора.
Энергия + оптимальный финиш + УМНОЕ НАПРАВЛЕНИЕ: перед раскачкой u = UM·sign(ω(1 − E)) — первая дуга s0·UM длительностью τ0 из перебора (оба знака; у дна ω ≈ 0 — знак раскачки
произволен, а от него зависит фаза всего качания); финиш стрельбой ≤ 3 дуг (pend_shoot_ref.shoot) каждые HO с в 30° от верха; затем ДОВОДКА всего плана (дуги bang-bang:
пред-дуга, раскачка, финиш) SLSQP по длительностям дуг (min Σ τ, конец в коробке цели) — «оптимальный момент переключения», а не сетка HO.
Эталон = min(все кандидаты, pend_ref_eshoot_u*.npy) — всё реальные траектории (верхние оценки T*). Запуск: UM=.15 NP=14 python3 pend_ref_best.py → pend_ref_best_u<UM>.npy (T) и _plan."""
import numpy as np, os, sys, json, time, math
from tqdm import tqdm
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pend_cost_grid as PCG
UM = float(os.environ.get('UM', .3)); PCG.UM = UM
import pend_shoot_ref as PS
RHO = PS.R0; ANG = math.radians(float(os.environ.get('ANG', 30))); HO = float(os.environ.get('HO', .05)); TMAX = float(os.environ.get('TMAXE', 60.)); H = .005
TAU0 = [float(x) for x in os.environ.get('TAU0', '0.25,0.6,1.2').split(',')]; NPOL = int(os.environ.get('NPOL', 3)); SUB = float(os.environ.get('SUB', .02))
B = None
def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi
def rk(x, w, u, h):                                                                            # rk4 на числах (в ~15 раз быстрее numpy на малых массивах)
    k1x = w; k1w = math.sin(x) + u; x2 = x + h / 2 * k1x; w2 = w + h / 2 * k1w; k2x = w2; k2w = math.sin(x2) + u
    x3 = x + h / 2 * k2x; w3 = w + h / 2 * k2w; k3x = w3; k3w = math.sin(x3) + u; x4 = x + h * k3x; w4 = w + h * k3w; k4x = w4; k4w = math.sin(x4) + u
    return x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), w + h / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
def init():
    global B; B = {s: PS.band(s * UM) for s in (1., -1.)}
def run(y, sg, tau, sub=SUB):
    x, w = float(y[0]), float(y[1])
    for s_, t_ in zip(sg, tau):
        n = max(1, int(math.ceil(t_ / sub - 1e-9))); h = t_ / n
        for _ in range(n): x, w = rk(x, w, s_ * UM, h)
    return x, w
def ingoal(x, w, k=1.): return abs(wrap(x)) <= RHO * k + 1e-9 and abs(w) <= RHO * k + 1e-9
def merge(sg, tau):
    S, T = [], []
    for s_, t_ in zip(sg, tau):
        if t_ <= 1e-9: continue
        if S and S[-1] == s_: T[-1] += t_
        else: S.append(s_); T.append(t_)
    return S, T
def pump(q, s0, tau0):
    """пред-дуга s0·UM τ0, затем u = UM·sign(ω(1−E)); финиш-проба каждые HO в ANG от верха. Возвращает (T, план дуг) лучшего захода."""
    x, w = float(q[0]), float(q[1]); t = 0.; best = (np.inf, None); nxt = 0.; sg = []; tau = []
    if tau0 > 0:
        n = int(round(tau0 / H)); sg.append(s0); tau.append(n * H)
        for _ in range(n): x, w = rk(x, w, s0 * UM, H)
        t = n * H
    while t < min(TMAX, best[0]):
        if abs(wrap(x)) <= ANG and t >= nxt:
            nxt = t + HO; Tf, pl = PS.shoot(np.array([wrap(x), w]), B)
            if np.isfinite(Tf) and pl is not None and t + Tf < best[0]:
                s, t1, t2, t3 = pl; S, T = merge(sg + [s, -s, s], tau + [t1, t2, t3]); best = (t + Tf, (S, T))
        E = w * w / 2 + math.cos(x); u = (1. if w * (1 - E) + 1e-12 > 0 else -1.) if PUMP == 1 else (1. if w > 0 else -1.) if PUMP == 2 else (1. if w * (1 - E) + 1e-12 > 0 else -1.) if abs(1 - E) > PEPS else (sg[-1] if sg else 1.)   # r17 PUMP 2: sign(w) only (one arc per half-swing); 3: hold u near E=1
        x, w = rk(x, w, u * UM, H); t += H
        if sg and sg[-1] == u: tau[-1] += H
        else: sg.append(u); tau.append(H)
    return best
NARC = int(os.environ.get('NARC', 12)); PUMP = int(os.environ.get('PUMP', 1)); PEPS = float(os.environ.get('PEPS', .05))   # r17: polish only plans with <= NARC arcs (pump chatters at E~1 -> hundreds of arcs -> SLSQP O(n^2) runs for hours)
def polish(q, sg, tau):
    if len(tau) > NARC: return np.inf, None
    """SLSQP по длительностям дуг: min Σ τ, конец в коробке цели (с запасом .98); результат проверяется мелким шагом."""
    cons = [{'type': 'ineq', 'fun': lambda z: np.array([.98 * RHO - abs(wrap(run(q, sg, z)[0])), .98 * RHO - abs(run(q, sg, z)[1])])}]
    r = minimize(lambda z: z.sum(), np.array(tau), jac=lambda z: np.ones_like(z), constraints=cons, bounds=[(0, None)] * len(tau), method='SLSQP', options=dict(maxiter=150))
    S, T = merge(sg, list(r.x)); x, w = run(q, S, T, sub=.005)
    return (float(sum(T)), (S, T)) if ingoal(x, w) else (np.inf, None)
def one(q):
    q = np.array(q, float); cands = []; T0, p0 = PS.shoot(q, B)
    if np.isfinite(T0) and p0 is not None: s, t1, t2, t3 = p0; cands.append((T0, merge([s, -s, s], [t1, t2, t3])))
    for s0, tau0 in [(1., 0.)] + [(s, t) for s in (1., -1.) for t in TAU0]:
        r = pump(q, s0, tau0)
        if r[1] is not None: cands.append(r)
    cands.sort(key=lambda c: c[0]); best = cands[0][0] if cands else np.inf; bp = cands[0][1] if cands else None
    for c in cands[:NPOL]:
        try: T, pl = polish(q, *c[1])
        except Exception: continue
        if T < best: best, bp = T, pl
    return best, (len(bp[0]) if bp else 0), (cands[0][0] if cands else np.inf)
def lambda_one(a): return a[0], one(a[1])
if __name__ == '__main__':
    from multiprocessing import Pool
    t0 = time.time(); rq = np.random.default_rng(0); Q = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)[:int(os.environ.get('NQ', 100))]; IX = os.environ.get('IX'); Q = Q[[int(i) for i in IX.split(',')]] if IX else Q
    R = np.full((len(Q), 3), np.inf); ck = 'pend_ref_best_ck_u%g%s.npy' % (UM, os.environ.get('TAG', ''))                       # r17: per-start checkpoint (imap_unordered)
    with Pool(int(os.environ.get('NP', 14)), initializer=init) as pool:
        for i, r_ in tqdm(pool.imap_unordered(lambda_one, list(enumerate(Q))), total=len(Q), desc='best u=%g' % UM, mininterval=float(os.environ.get('TQDM_MI', 10))): R[i] = r_; np.save(ck, R)
    if IX: print('IXRES', json.dumps(R[:, 0].tolist()), flush=True); sys.exit(0)
    T = R[:, 0]; fn = 'pend_ref_eshoot_u%g.npy' % UM; E = np.load(fn)[:len(T)] if os.path.exists(fn) else np.full(len(T), np.inf); Tb = np.minimum(T, E)
    np.save('pend_ref_best_u%g%s.npy' % (UM, PS.SUF), Tb); np.save('pend_ref_best_raw_u%g%s.npy' % (UM, PS.SUF), R)
    out = dict(UM=UM, reach=float(np.isfinite(Tb).mean()), T_med=float(np.median(Tb[np.isfinite(Tb)])), better_than_eshoot=int((T < E - 1e-6).sum()), vs_eshoot_mean=round(float(np.mean(Tb / E)), 4), sec=round(time.time() - t0))
    if UM == .3 and PS.R0 == .1 and os.path.exists('pend_ref_T.npy'):
        S = np.load('pend_ref_T.npy')[:len(T)]; ok = np.isfinite(S) & (S > .05); r = Tb[ok] / S[ok]; out.update(vs_shoot_mean=round(float(r.mean()), 4), vs_shoot_min=round(float(r.min()), 4), vs_shoot_max=round(float(r.max()), 4), n_over_1_05=int((r > 1.05).sum()))
    print(json.dumps(out), flush=True)
