"""Покрытие слоя nD-клетками (как cover.py): случайные пробы в коробке, новая клетка — в непокрытой пробе; ядро укорачивается до непересечения
(симметрично, по выборкам ядер). Метрики: покрытие, перекрытие ядер/гало среди покрытого, выход клетки в ядро соседа. В высоких размерностях покрытие
растёт медленно — max_cells и время ограничивают; результат — кривая покрытия (log)."""
import time
import numpy as np
from .celln import CellN, HALO


def cover_layer_n(S, k, NP=4000, max_cells=300, tmax=600, seed=0, log=None):
    rng = np.random.default_rng(seed); P = S.sample(rng, NP); covered = np.zeros(NP, bool); cells = []; t0 = time.time(); curve = []
    for i in rng.permutation(NP):
        if covered[i]: continue
        if len(cells) >= max_cells or time.time() - t0 > tmax: break
        C = CellN(S, k, P[i], rng=rng)
        for _ in range(12):
            Z = C.kernel_samples(rng); hit = False
            for D in cells:
                if np.linalg.norm(S.wrap(D.c - C.c)) > D.rad + C.rad: continue
                if D.locate(Z, 1.0)[2].any() or C.locate(D.Z, 1.0)[2].any(): hit = True; break
            if not hit: break
            if C.tau > 0.05: C.tau *= 0.6
            else: C.r *= 0.6
            C._finish()
        C.Z = C.kernel_samples(rng, 96); cells.append(C)
        rem = np.nonzero(~covered)[0]; covered[rem[C.locate(P[rem], 1.0)[2]]] = True
        if len(cells) % 20 == 0:
            curve.append((len(cells), float(covered.mean()), round(time.time() - t0)))
            if log: log(*curve[-1])
    return cells, P, covered, curve


def metrics_n(S, cells, P, covered):
    kc = np.zeros(len(P), int); hc = np.zeros(len(P), int)
    for C in cells: kc += C.locate(P, 1.0)[2]; hc += C.locate(P, HALO)[2]
    cov = max((kc >= 1).sum(), 1)
    r = dict(cells=len(cells), covered=float(covered.mean()), kernel_overlap=float((kc >= 2).sum() / cov), halo_overlap=float((hc >= 2).sum() / cov),
             r_mean=float(np.mean([C.r for C in cells])), tau_mean=float(np.mean([C.tau for C in cells])), tau_max_frac=float(np.mean([C.tau >= 0.99 for C in cells])),
             err_mean=float(np.mean([C.err for C in cells])))
    EX = np.array([C.exit for C in cells]); inn = np.zeros(len(cells), bool)
    for j, C in enumerate(cells):
        inn[j] = any(D is not C and D.locate(EX[j:j + 1], 1.0)[2][0] for D in cells)
    r['exit_in_other_kernel'] = float(inn.mean()); return r
