"""research hub-research-6: модель ПОЛЯ механической системы только по q. q̈ = M(q)⁻¹(u − C(q)[ω,ω] − g·G(q)) — точно квадратично по ω и аффинно по u,
поэтому локальная модель «коэффициенты M⁻¹, C, G — Тейлор-2 по q в центре клетки» точна по (ω,u); ошибка только от Тейлора по q.
Замер: радиус ρ клетки в q (шар), при котором max относит. ошибка ускорения ≤ tol (по случайным q0, δq, |ω|≤WM, |u|≤1); число клеток на тор (2π)^n."""
import numpy as np, itertools
rng = np.random.default_rng(0)

def chain(n, g=1.0):
    """n-звенник, точечные массы 1 на концах, l=1, q_i — относительные углы. Возвращает acc(q, w, u)."""
    def acc(q, w, u):
        th = np.cumsum(q); dth = np.cumsum(w)
        J = np.zeros((n, 2, n))                                  # якобианы точек k по q (абс. углы → относит. через cumsum)
        for k in range(n):
            for j in range(k+1): J[k, :, j] = [np.cos(th[j]), np.sin(th[j])]
        T = np.tril(np.ones((n, n)))                             # dth/dq
        Jq = J @ T                                               # (n,2,n)
        M = sum(Jq[k].T @ Jq[k] for k in range(n))
        # C[ω,ω]: Σ_k Jqᵀ · d/dt(Jq)·ω ;  d/dt J[k,:,j] = [−sin, cos]·dth_j
        Cw = np.zeros(n)                                         # Mq̈ + Cw + gG = u,  Cw = Σ Jqᵀ a_k
        for k in range(n):
            a = sum(np.array([-np.sin(th[j]), np.cos(th[j])]) * dth[j]**2 for j in range(k+1))   # p_k = Σ(sin θ_j, −cos θ_j): p̈ = Jq̈ + a
            Cw += Jq[k].T @ a
        # потенциал: y_k = −Σ cos th_j (вниз), G = ∂/∂q Σ_k y_k
        Gp = np.zeros(n)
        for k in range(n):
            for j in range(k+1): Gp += np.sin(th[j]) * T[j]
        return np.linalg.solve(M, u - Cw - g * Gp)
    return acc

def taylor_err(acc, n, rho, WM=3.0, N=60):
    e = 0.0; h = 1e-3; E = np.eye(n)
    for _ in range(N):
        q0 = rng.uniform(-np.pi, np.pi, n); w = rng.uniform(-WM, WM, n); u = rng.uniform(-1, 1, n)
        a0 = acc(q0, w, u); Jq = np.array([(acc(q0+h*E[i], w, u) - acc(q0-h*E[i], w, u))/(2*h) for i in range(n)]).T
        H = np.zeros((n, n, n))
        for i, j in itertools.product(range(n), repeat=2):
            H[:, i, j] = (acc(q0+h*E[i]+h*E[j], w, u) - acc(q0+h*E[i]-h*E[j], w, u) - acc(q0-h*E[i]+h*E[j], w, u) + acc(q0-h*E[i]-h*E[j], w, u))/(4*h*h)
        for _ in range(8):
            d = rng.normal(size=n); d *= rho/np.linalg.norm(d)
            ex = acc(q0+d, w, u); md = a0 + Jq@d + 0.5*np.einsum('kij,i,j->k', H, d, d)
            e = max(e, np.linalg.norm(ex-md)/max(np.linalg.norm(ex), 1.0))
    return e

exec(open('reports/research/spectrum_image.py').read().split('def errs')[0])
x = rng.uniform(-2, 2, 4); u2 = rng.uniform(-1, 1, 2)
print('сверка n=2 с dpend:', np.abs(chain(2)(x[:2], x[2:], u2) - dpend(x, u2)[2:]).max())
for n in (2, 3):
    acc = chain(n)
    # проверка по ω: ошибка квадратичной модели по ω = 0 (численно)
    q0 = rng.uniform(-3, 3, n); u = rng.uniform(-1, 1, n); ws = [rng.uniform(-3, 3, n) for _ in range(3)]
    for rho in (.1, .2, .3, .5):
        e = taylor_err(acc, n, rho); cells = (2*np.pi/(2*rho/np.sqrt(n)))**n
        print(f'n={n} ({2*n}D)  ρ={rho}: отн. ошибка ускорения max {e:.1e}   клеток на тор q ≈ {cells:.0f}', flush=True)
