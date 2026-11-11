"""Numba-refine для дрона ENV4 (линия B, b19–b20; перенос из bench/spores_drone_c.py). ВСЁ ПОД ФЛАГОМ: файл не импортируется corridor_nd, пока не вызван install(); по умолчанию выкл.
Что делает: SLSQP по длительностям сегментов; зазор до препятствий (эллипс + круг, опц. потолок y=4 вне окна) на КАЖДОМ подшаге Эйлера .02 (как среда), число точек на сегмент фиксировано по d0 (Lm = int(UB/.02)+3, недостающие — повтор последней),
агрегация KS (ρ) по сегменту + якобиан целиком в njit; проверка на КВАНТОВАННОМ (кратно .02) плане и ужесточение запаса (px, gx) до QITER раз. Цель: 19–20/20, 0 касаний, запрос ~.5 с.
Использование: from src.atlas6 import drone_nb; drone_nb.warmup(P); drone_nb.install(P) (до fork!); P = dict(CD, M, I, G, LAT, EC, EA, EB, CC, CR, PAD, HW, YG, ceil, rho, qiter); P['xp']=[pad_x, gm_x] — запас на повтор; drone_nb.uninstall()."""
import numpy as np, numba as nb
from . import corridor_nd as _cnd

@nb.njit
def _cons(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil):
    out = np.empty(2 + Lm.sum()); x, y, th, vx, vy, w = x0[0], x0[1], x0[2], x0[3], x0[4], x0[5]; k = 2
    US0 = np.array([1., 1., -1., -1.]); US1 = np.array([.5, -.5, .5, -.5])
    for j in range(len(seq)):
        u0 = US0[seq[j]]; u1 = US1[seq[j]]; t = d[j]; nst = int(abs(t) / .02 + 1e-9); hr = t - .02 * nst; cnt = 0; lastc = 0.
        for q in range(nst + 1):
            h = .02 if q < nst else hr
            if h == 0.: continue
            c = np.cos(th); s_ = np.sin(th); sp = np.sqrt(vx * vx + vy * vy)
            ax = (c * u0 - s_ * u1 - CD * sp * vx) / M; ay = (s_ * u0 + c * u1 - CD * sp * vy - G) / M; aw = LAT * u1 / I
            vx += ax * h; vy += ay * h; w += aw * h; x += vx * h; y += vy * h; th += w * h
            dx = x - ECx; dy = y - ECy; rho = np.sqrt((dx / EA) ** 2 + (dy / EB) ** 2) + 1e-12; ct = dx / (EA * rho); st = dy / (EB * rho)
            cl = min((rho - 1) * np.sqrt((EA * ct) ** 2 + (EB * st) ** 2), np.sqrt((x - CCx) ** 2 + (y - CCy) ** 2) - CR) - PADT
            if ceil: cl = min(cl, max(4. - y, HW - abs(x)))
            if cnt < Lm[j]: out[k] = cl; k += 1; cnt += 1
            lastc = cl
        while cnt < Lm[j]: out[k] = lastc; k += 1; cnt += 1
    out[0] = HW * HW - x * x; out[1] = y - YGT
    return out

@nb.njit
def _ksc(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil, rho):
    c = _cons(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil); out = np.empty(2 + len(Lm)); out[0] = c[0]; out[1] = c[1]; k = 2
    for j in range(len(Lm)):
        m = c[k]
        for q in range(Lm[j]): m = min(m, c[k + q])
        sm = 0.
        for q in range(Lm[j]): sm += np.exp(-rho * (c[k + q] - m))
        out[2 + j] = m - np.log(sm) / rho; k += Lm[j]
    return out

@nb.njit
def _ksjac(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil, rho):
    b = _ksc(x0, seq, d, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil, rho); J = np.empty((len(b), len(d)))
    for j in range(len(d)):
        dp = d.copy(); dp[j] += 1e-6; J[:, j] = (_ksc(x0, seq, dp, Lm, CD, M, I, G, LAT, ECx, ECy, EA, EB, CCx, CCy, CR, PADT, HW, YGT, ceil, rho) - b) / 1e-6
    return J

def _args(P, x0a, sa, d, Lm, px, gx):
    xp = P.get('xp', (0., 0.))
    return (x0a, sa, np.asarray(d, float), Lm, P['CD'], P['M'], P['I'], P['G'], P['LAT'], P['EC'][0], P['EC'][1], P['EA'], P['EB'], P['CC'][0], P['CC'][1], P['CR'],
            P['PAD'] + xp[0] + px, P['HW'] - xp[1] - gx, P['YG'] + xp[1] + gx, int(P.get('ceil', 0)))

def warmup(P):
    """Компиляция njit ДО fork (иначе ~1.5 с JIT в каждом воркере на первом запросе — был «p90 запроса 3 с» b19)."""
    x = np.zeros(6); sq = np.array([0, 1], np.int64); dd = np.array([.3, .3]); Lm = ((2. * dd + .2) / .02).astype(np.int64) + 3; A = _args(P, x, sq, dd, Lm, 0., 0.)
    _cons(*A); _ksc(*A, 300.); _ksjac(*A, 300.)

def make_refine(P):
    from scipy.optimize import minimize
    rho = float(P.get('rho', 300.)); qiter = int(P.get('qiter', 5))
    def refine_nb(flow, x0, path, g, tries=1, seed=0, clear=None, ns=16, wlim=None):
        seq, d0 = _cnd.merge(path); sa = np.array(seq, np.int64); x0a = np.array(x0, float); UB = 2. * d0 + .2; Lm = (UB / .02).astype(np.int64) + 3
        px = gx = 0.; ret = None
        for it in range(qiter + 1):
            rng = np.random.default_rng(seed); best = (np.inf, None)
            A = lambda d: _args(P, x0a, sa, d, Lm, px, gx) + (rho,)
            con = {'type': 'ineq', 'fun': lambda d: _ksc(*A(d)), 'jac': lambda d: _ksjac(*A(d))}
            for k in range(tries):
                di = d0 if k == 0 else d0 * rng.uniform(0.8, 1.2, len(d0))
                r = minimize(lambda d: d.sum(), di, jac=lambda d: np.ones_like(d), bounds=[(0, u) for u in UB], constraints=[con], method='SLSQP', options=dict(maxiter=80, ftol=1e-9))
                if np.all(con['fun'](r.x) >= -1e-6) and r.x.sum() < best[0]: best = (float(r.x.sum()), r.x)
            if best[1] is None: break
            e = np.round(np.cumsum(best[1]) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); c = _cons(*_args(P, x0a, sa, dq, Lm, px, gx)); ret = best
            if c.min() >= -1e-6: break
            if c[2:].min() < 0: px += -c[2:].min() + .01
            if c[:2].min() < 0: gx += -c[:2].min() + .01
        ok = ret is not None
        return (ret[0] if ok else float(d0.sum())), seq, (ret[1] if ok else d0), ok
    return refine_nb

_orig = None
def install(P):
    """Подменяет corridor_nd.refine; uninstall() возвращает штатный. Вызывать ДО fork пула."""
    global _orig
    if _orig is None: _orig = _cnd.refine
    warmup(P); _cnd.refine = make_refine(P)
def uninstall():
    global _orig
    if _orig is not None: _cnd.refine = _orig; _orig = None
