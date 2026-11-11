"""c8: патч growCar.py (копия, оригинал не трогаем) — по профилю build кара: numba-ядра для f/rk4 (650k вызовов, 15%), _contract (294k, 11%), HexIdx._key (686k, 7.5%).
Включается NBF=1 (по умолчанию выкл. — оригинал). Проверка: bench/w_c8car/test_nbf.py (побитово/ulp на случайных входах) + сверка V атласа.
python make_growCar_c8.py <growCar.py> <out>"""
import sys
src = open(sys.argv[1]).read()
key = "if int(E('STGPU', 0)):"
assert src.count(key) == 1
block = '''
if int(E('NBF', 0)) and SYS == 'car' and not VSAT:      # c8: numba-ядра (NBF=1)
    import numba as nb
    _f0, _rk40, _contract0, _key0 = f, rk4, _contract, HexIdx._key
    @nb.njit(cache=True)
    def _f_k(Y, ua, uw):
        n = Y.shape[0]; o = np.empty((n, 4))
        for i in range(n):
            o[i, 0] = Y[i, 3] * np.cos(Y[i, 2]); o[i, 1] = Y[i, 3] * np.sin(Y[i, 2]); o[i, 2] = uw + 0 * Y[i, 2]; o[i, 3] = ua - KV * Y[i, 3]
        return o
    @nb.njit(cache=True)
    def _rk4_k(Y, ua, uw, h, n):
        for _ in range(n):
            k1 = _f_k(Y, ua, uw); k2 = _f_k(Y + h / 2 * k1, ua, uw); k3 = _f_k(Y + h / 2 * k2, ua, uw); k4 = _f_k(Y + h * k3, ua, uw); Y = Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Y
    def f(y, u):
        y = np.asarray(y, float); return _f_k(np.ascontiguousarray(y.reshape(-1, 4)), float(u[0]), float(u[1])).reshape(y.shape)
    def rk4(y, u, h, n):
        y = np.asarray(y, float); return _rk4_k(np.ascontiguousarray(y.reshape(-1, 4)), float(u[0]), float(u[1]), float(h), int(n)).reshape(y.shape)
    @nb.njit(cache=True)
    def _contract_k(X, S, j):
        K = X.shape[0]; out = np.empty((K, 4)) if j < 0 else np.empty((K, 4)); T = np.empty((16, 4))
        for e in range(K):
            for v in range(16):
                for z in range(4): T[v, z] = X[e, v, z]
            half = 8
            for k in range(4):
                for v in range(half):
                    for z in range(4):
                        d = T[v + half, z] - T[v, z]
                        T[v, z] = d if k == j else T[v, z] + S[e, k] * d
                half //= 2
            for z in range(4): out[e, z] = T[0, z]
        return out
    def _contract(Xa, sa, j=-1):
        if isinstance(Xa, np.ndarray) and Xa.shape[1:] == (16, 4): return _contract_k(np.ascontiguousarray(Xa), np.ascontiguousarray(sa, dtype=np.float64), j)
        return _contract0(Xa, sa, j)
    @nb.njit(cache=True)
    def _key_k(ib):
        n = ib.shape[0]; key = np.zeros(n, np.int64)
        for i in range(n):
            kk = np.int64(0)
            for k in range(4): kk = kk * 1024 + (ib[i, k] + 512)
            key[i] = kk
        return key
    HexIdx._key = staticmethod(lambda ib: _key_k(np.ascontiguousarray(ib, dtype=np.int64)) if (N == 4 and getattr(ib, 'ndim', 0) == 2 and ib.shape[1] == 4) else _key0(ib))
    print('NBF: numba f/rk4/_contract/_key', flush=True)
'''
open(sys.argv[2], 'w').write(src.replace(key, block + key))
