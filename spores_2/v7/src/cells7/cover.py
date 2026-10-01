"""Покрытие слоя клетками с минимальным перекрытием (PLAN 0з п.2): новая клетка — только в точку, не лежащую в ЯДРЕ клеток слоя;
ядро новой клетки укорачивается (τ, затем r), пока в нём нет точек ядер других клеток слоя (проверка по выборке). Перекрываются только гало."""
import numpy as np
from .cell import Cell, HALO, R_MAX, TAU_MAX


def probe(S, m):
    (a0, b0), (a1, b1) = S.box; g0 = a0 + (np.arange(m) + .5) * (b0 - a0) / m; g1 = a1 + (np.arange(m) + .5) * (b1 - a1) / m
    return np.stack(np.meshgrid(g0, g1, indexing='ij'), -1).reshape(-1, 2)


def kernel_samples(C, ns=7, nt=11):
    s = np.linspace(-C.r, C.r, ns) * 0.98; t = np.linspace(0, C.tau, nt) * 0.98 + 0.01 * C.tau
    X = np.array([np.interp(t, C.ts, C.X[:, i]) for i in range(2)]).T; V = np.array([np.interp(t, C.ts, C.V[:, i]) for i in range(2)]).T
    return (X[None, :, :] + s[:, None, None] * V[None, :, :]).reshape(-1, 2)


def cover_layer(S, k, m=120, seed=0, order='random', max_cells=4000, log=None):
    P = probe(S, m); rng = np.random.default_rng(seed); idx = rng.permutation(len(P)) if order == 'random' else np.arange(len(P))
    covered = np.zeros(len(P), bool); cells = []
    for i in idx:
        if covered[i]: continue
        if len(cells) >= max_cells: break
        C = Cell(S, k, P[i])
        for _ in range(16):                              # укорачивание ядра до непересечения с ядрами слоя (симметрично: точки C в D и точки D в C)
            Z = kernel_samples(C); hit = False
            if cells:
                cc = np.array([D.c for D in cells]); rad = np.array([D.tau for D in cells])
                near = np.nonzero(np.sum(S.wrap(cc - C.c) ** 2, 1) <= (2.5 * (rad + C.tau) + 1) ** 2)[0]
                for jd in near:
                    D = cells[jd]
                    if D.locate(Z, 1.0)[2].any() or C.locate(D.Z, 1.0)[2].any(): hit = True; break
            if not hit: break
            if C.tau > 0.05: C.tau *= 0.6
            else: C.r *= 0.6
            C._build()
        C.Z = kernel_samples(C)
        cells.append(C)
        rem = np.nonzero(~covered)[0]; covered[rem[C.locate(P[rem], 1.0)[2]]] = True
        if log and len(cells) % 100 == 0: log(len(cells), covered.mean())
    return cells, P, covered


def metrics(S, cells, P, covered):
    """Перекрытие (по пробам): ядра (≥2 ядер), гало (≥2 гало) / покрытая площадь; выход клетки → в ядро другой клетки; линейное перекрытие
    соседей = глубина проникновения точек выхода и бокового сдвига гало в ядро соседа / линейный размер клетки."""
    kc = np.zeros(len(P), int); hc = np.zeros(len(P), int)
    for C in cells: kc += C.locate(P, 1.0)[2]; hc += C.locate(P, HALO)[2]
    cov = (kc >= 1).sum(); r = dict(cells=len(cells), covered=float(covered.mean()), kernel_overlap=float((kc >= 2).sum() / max(cov, 1)), halo_overlap=float((hc >= 2).sum() / max(cov, 1)),
                                    r_mean=float(np.mean([C.r for C in cells])), tau_mean=float(np.mean([C.tau for C in cells])))
    EX = np.array([C.exit for C in cells]); inn = np.zeros(len(cells), bool); lin = []
    for j, C in enumerate(cells):
        ins = [D.locate(EX[j:j + 1], 1.0)[2][0] and D is not C for D in cells]
        inn[j] = any(ins)
    r['exit_in_other_kernel'] = float(inn.mean())
    return r
