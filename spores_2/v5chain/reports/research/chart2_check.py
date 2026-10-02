"""research hub-v5chain-research-8: карта клетки Ψ(x0; s,t,u) = Φ(x0 + N s, t, u) — квадратичная по (s,t,u) вместе (идея пользователя 2026-10-02:
«динамика как функция начальной точки, времени, управления и координат нормального криволинейного базиса»). Коэффициенты — МНК по точным
прогонам в узлах Чебышёва (без производных). Ошибка на случайных точках клетки, в долях размера клетки (диаметр образа)."""
import numpy as np, itertools
def rk(f, x, u, T, n=60):
    h = T / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
def feats(Z, deg):                                                            # мономы степени ≤ deg
    cols = [np.ones(len(Z))]
    for d in range(1, deg + 1):
        for c in itertools.combinations_with_replacement(range(Z.shape[1]), d): cols.append(np.prod(Z[:, c], 1))
    return np.stack(cols, 1)
pend = (lambda x, u: np.stack([x[1], np.sin(x[0]) + u[0]]), 2, 1, .3)        # f, n, m, |u|max
dd = (lambda x, u: np.stack([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]]), 3, 2, 1.)
rng = np.random.default_rng(0)
for nm, (f, n, m, um) in (('маятник', pend), ('дифдрайв', dd)):
    for r, tau in ((.1, .3), (.2, .3), (.1, .5), (.2, .5), (.2, 1.)):
        errs = {2: [], 3: []}
        for _ in range(30):
            x0 = rng.uniform(-2, 2, n); u0 = rng.choice([-um, um], m)                # спора: точка + вершина U
            F = f(x0[:, None], u0[:, None])[:, 0]; Fn = F / np.linalg.norm(F)
            N = np.linalg.svd(np.eye(n) - np.outer(Fn, Fn))[0][:, :n - 1]          # нормальная гиперплоскость
            def Psi(Z):                                                           # Z = (s(n−1), t∈[0,1]·τ, du(m)) нормировано в [−1,1]
                s, t, du = Z[:, :n - 1] * r, (Z[:, n - 1] + 1) / 2 * tau, Z[:, n:] * um * .5
                X = x0[:, None] + N @ s.T; U = np.clip(u0[:, None] + du.T, -um, um)
                return np.stack([rk(lambda x, u: f(x, u), X[:, i], U[:, i], t[i], 30) for i in range(len(Z))])
            cheb = np.cos(np.pi * (np.arange(4) + .5) / 4); d = n - 1 + 1 + m
            G = np.array(list(itertools.product(cheb, repeat=d))); Y = Psi(G)
            Zt = rng.uniform(-1, 1, (200, d)); Yt = Psi(Zt); diam = np.ptp(Yt, 0).max()
            for deg in (2, 3):
                c = np.linalg.lstsq(feats(G, deg), Y, rcond=None)[0]; errs[deg].append(np.abs(feats(Zt, deg) @ c - Yt).max() / diam)
        print(f'{nm:8s} r={r} τ={tau} |du|≤{um*.5}: ошибка/размер клетки  2-й порядок med {np.median(errs[2]):.1e} max {np.max(errs[2]):.1e} | 3-й med {np.median(errs[3]):.1e}', flush=True)
