"""c14 блок 7: пакетный поток m8 с УПРАВЛЕНИЕМ ПО СТРОКАМ (слой на каждую точку) — та же схема RK4 с шагом dt_max=.02, что manip3g_dyn.flow (n=4);
numba, одна строка = один проход без аллокаций. Используется в проигрыше стыка (candidates / cand) вместо 16 вызовов numpy-потока по слоям."""
import numpy as np, numba as nb
import manip3g_dyn as G3

@nb.njit(cache=True)
def _f(z, tau, S, gw, n, out, c, s, M, td):
    th = 0.0; thd = 0.0
    for i in range(n):
        th += z[i]; thd += z[n + i]; c[i] = np.cos(th); s[i] = np.sin(th); td[i] = thd      # td временно хранит thd
    for i in range(n):
        h = 0.0
        for j in range(n):
            M[i, j] = S[i, j] * (c[i] * c[j] + s[i] * s[j])
            h += S[i, j] * (s[i] * c[j] - c[i] * s[j]) * td[j] * td[j]
        qt = tau[i] - (tau[i + 1] if i + 1 < n else 0.0)
        M[i, n] = qt - gw[i] * c[i] - h
    for cc in range(n):
        for k in range(cc + 1, n):
            f_ = M[k, cc] / M[cc, cc]
            for j in range(cc, n + 1): M[k, j] -= f_ * M[cc, j]
    for cc in range(n - 1, -1, -1):
        v = M[cc, n]
        for j in range(cc + 1, n): v -= M[cc, j] * td[j]
        td[cc] = v / M[cc, cc]
    for i in range(n):
        out[i] = z[n + i]; out[n + i] = td[i] - (td[i - 1] if i else 0.0)

@nb.njit(cache=True)
def rk4_rows(X, Sidx, t, dt_max, S, gw, n):
    B = X.shape[0]; Y = np.empty_like(X); k = max(1, int(np.ceil(abs(t) / dt_max))); h = t / k; d = 2 * n
    tau = np.empty(n); z = np.empty(d); k1 = np.empty(d); k2 = np.empty(d); k3 = np.empty(d); k4 = np.empty(d); zz = np.empty(d)
    c = np.empty(n); s = np.empty(n); M = np.empty((n, n + 1)); td = np.empty(n)
    for r in range(B):
        for i in range(n): tau[i] = 1.0 if (Sidx[r] >> i) & 1 else -1.0
        for i in range(d): z[i] = X[r, i]
        for _ in range(k):
            _f(z, tau, S, gw, n, k1, c, s, M, td)
            for i in range(d): zz[i] = z[i] + h / 2 * k1[i]
            _f(zz, tau, S, gw, n, k2, c, s, M, td)
            for i in range(d): zz[i] = z[i] + h / 2 * k2[i]
            _f(zz, tau, S, gw, n, k3, c, s, M, td)
            for i in range(d): zz[i] = z[i] + h * k3[i]
            _f(zz, tau, S, gw, n, k4, c, s, M, td)
            for i in range(d): z[i] = z[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
        for i in range(d): Y[r, i] = z[i]
    return Y

def make(g, n=4):
    mm = np.asarray(G3.MS[:n], float); ln = np.asarray(G3.L[:n], float)
    S = np.array([[mm[max(i, j):].sum() * ln[i] * ln[j] for j in range(n)] for i in range(n)]); gw = np.array([g * ln[i] * mm[i:].sum() for i in range(n)])
    return lambda X, Sidx, t, dt_max=.02: rk4_rows(np.ascontiguousarray(X, float), np.ascontiguousarray(Sidx, np.int64), float(t), dt_max, S, gw, n)
