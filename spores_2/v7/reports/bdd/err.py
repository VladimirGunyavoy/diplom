"""Дифдрайв-бабочки v7 при ошибке модели (PLAN «Открыто» п.2, hub-v5chain-worker-15): реальный привод v·GV, ω·GW (план — по номиналу).
Агент без привязки к центру споры (в run() v7 позиция «защёлкивается» на центр — идеализация): после каждого сегмента пересчёт из РЕАЛЬНОГО состояния.
Цель — коробка |xy| ≤ TOL, |θ| ≤ THT (точный ноль при ошибке недостижим). Запуск из spores_2/v7: RW=.9 GV=.95 GW=.95 python3 reports/bdd/err.py N."""
import sys, os, time, json
sys.path.insert(0, '.'); here = os.getcwd()
from src.cells7 import butterfly_dd as BD          # chdir в research внутри
import numpy as np
M, wrap, BIG, move, tgt, tgtgt = BD.M, BD.wrap, BD.BIG, BD.move, BD.tgt, BD.tgtgt
GV, GW = float(os.environ.get('GV', 1)), float(os.environ.get('GW', 1)); TOL, THT = float(os.environ.get('TOL', .03)), float(os.environ.get('THT', .05)); SNAP = int(os.environ.get('SNAP', 0))
def ingoal(y): return np.hypot(y[0], y[1]) <= TOL and abs(wrap(y[2])) <= THT
ARCV = int(os.environ.get('ARCV', 0)); SIG = float(os.environ.get('SIG', 0)); NRNG = np.random.default_rng(7)     # SIG>0: агент видит состояние с шумом σ (xy и θ), цель/истина — по реальному
EMIN, PMIN = float(os.environ.get('EMIN', 1e-2)), float(os.environ.get('PMIN', 1e-3))   # пороги длины хорды/угла для оценки ĝ (при шуме — больше)
FILT = float(os.environ.get('FILT', 0))   # FILT=K∈(0,1]: комплементарный фильтр — предсказание по команде и ĝ, коррекция K·(замер−предсказание)
EST = int(os.environ.get('EST', 0))     # EST=1: оценка ĝV, ĝW по уже выполненным сегментам (наблюдаем реальное состояние), команда делится на ĝ
def run(B, q, smax=400):
    y = np.array(q, float); t = 0.; at = -1; segs = 0; gv = gw = 1.; hv = hw = False; ye = None
    for _ in range(smax):
        if ingoal(y): return t, segs
        yo = y + SIG * NRNG.normal(size=3) if SIG else y
        if FILT and ye is not None: yo_use = ye
        else: yo_use = yo; ye = yo.copy()
        J, act = B.best(yo_use, at)
        if act is None or J >= BIG / 2: return np.inf, segs
        kind, phi, L, k = act; y0 = y.copy(); y0o = yo.copy()
        gm = np.sqrt(gv * gw) if EST == 2 else 1.; cv, cw = (gv / gm, gw / gm) if (EST != 2 or (hv and hw)) else (1., 1.)    # EST=2: компенсируем только РАЗНИЦУ осей (общий масштаб оставляем замкнутой пересчётке)
        if kind == 'turn':
            y = np.array([y[0], y[1], y[2] + GW * phi / cw]); t += abs(phi / cw)
            yo1 = y + SIG * NRNG.normal(size=3) if SIG else y
            if EST and abs(phi) > PMIN: gw = float(np.clip(wrap(yo1[2] - y0o[2]) * cw / phi, .3, 3.)); hw = True
        else:
            y = move(y, GW * phi / cw, GV * L / cv); t += abs(L / cv) + abs(phi / cw)       # время = длительность КОМАНДЫ (с ĝ растягивается)
            yo1 = y + SIG * NRNG.normal(size=3) if SIG else y
            if EST:
                yn = move(y0o, phi, L); cn = np.hypot(*(yn[:2] - y0o[:2])); ca = np.hypot(*(yo1[:2] - y0o[:2]))
                if abs(phi) > PMIN: gw = float(np.clip(wrap(yo1[2] - y0o[2]) * cw / phi, .3, 3.)); hw = True                                 # дуга — набег курса ⇒ ĝW
                if cn > EMIN and (abs(phi) < PMIN or ARCV): gv = float(np.clip(cv * ca / cn, .3, 3.)); hv = True   # прямой участок — длина хорды ⇒ ĝV
            if SNAP: y[:2] = B.C[k]
        if FILT:
            yp = np.array([ye[0], ye[1], ye[2] + gw * phi / cw]) if kind == 'turn' else move(ye, gw * phi / cw, gv * L / cv); yo2 = y + SIG * NRNG.normal(size=3); d = yo2 - yp; d[2] = wrap(d[2]); ye = yp + FILT * d; ye[2] = wrap(ye[2])
        y[2] = wrap(y[2]); at = k if np.hypot(*((ye if FILT else yo1 if SIG else y)[:2] - B.C[k])) < 1e-3 + 3 * SIG else -1    ; segs += 1   # «стою в центре споры» — по НАБЛЮДАЕМОМУ состоянию с допуском 3σ; segs += 1
    return np.inf, segs
if __name__ == '__main__':
    N = int(sys.argv[1]); os.chdir(BD.RES); ref = np.load('dd_ref_60.npy'); rng = np.random.default_rng(1)
    Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]; t0 = time.time(); B = BD.ButterflyDDObs(N=N, obs=[]).solve(); print('готово', B.K, round(time.time() - t0), flush=True)
    R = [run(B, q) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / ref[f] if f.any() else np.array([np.nan])
    print(json.dumps(dict(N=N, GV=GV, GW=GW, TOL=TOL, THT=THT, SNAP=SNAP, SIG=SIG, FILT=FILT, n=len(Q), reach=round(float(f.mean()), 3), mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3), segs_med=float(np.median(S[f])), sec=round(time.time() - t0))), flush=True)
