"""PLAN п.27 (а): самоналожение клеток 2D-атласа. Для каждой клетки (сетка G: строки × столбцы × (θ, ω)) доля узлов строки i, лежащих внутри четырёхугольников ТОЙ ЖЕ клетки из
строк q (q, q+1) с i ∉ [q−1, q+2] (дальше соседних строк), с копиями θ ± 2π. Четырёхугольник — два треугольника. Запуск: python3 selfov_measure.py <папка эксперимента>/data/cells.pkl ..."""
import sys, pickle, numpy as np
PER = 2 * np.pi
def in_tri(P, A, B, C):
    """P (n,2), A,B,C (m,2) → (n,m) булева: P внутри треугольника."""
    def cr(a, b, p): return (b[None, :, 0] - a[None, :, 0]) * (p[:, None, 1] - a[None, :, 1]) - (b[None, :, 1] - a[None, :, 1]) * (p[:, None, 0] - a[None, :, 0])
    d1, d2, d3 = cr(A, B, P), cr(B, C, P), cr(C, A, P)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0); pos = (d1 > 0) | (d2 > 0) | (d3 > 0); return ~(neg & pos)
def measure(c):
    G = c['G'].astype(float); nr, nc = G.shape[:2]; th0 = G[nr // 2, nc // 2, 0]
    G = G.copy(); G[..., 0] = th0 + (G[..., 0] - th0 + PER / 2) % PER - PER / 2                      # развернуть θ клетки вокруг центра (клетка меньше периода)
    qs = [(q, j) for q in range(nr - 1) for j in range(nc - 1)]; q_ = np.array([a for a, _ in qs]); j_ = np.array([b for _, b in qs])
    A, B, C, D = G[q_, j_], G[q_, j_ + 1], G[q_ + 1, j_ + 1], G[q_ + 1, j_]
    P = G.reshape(-1, 2); ri = np.repeat(np.arange(nr), nc); hit = np.zeros(len(P), bool)
    for k in (-1, 0, 1):
        Pk = P + np.array([k * PER, 0.]); m = in_tri(Pk, A, B, C) | in_tri(Pk, A, C, D)
        far = (ri[:, None] < q_[None, :] - 1) | (ri[:, None] > q_[None, :] + 2); hit |= (m & far).any(1)
    return hit.mean(), th0, float(G[nr // 2, nc // 2, 1])
if __name__ == '__main__':
    for p in sys.argv[1:]:
        C = pickle.load(open(p, 'rb')); R = np.array([measure(c) for c in C]); f = R[:, 0]; bad = f > 0
        top = np.argsort(-f)[:3]
        print(p.split('/')[-3], '| клеток', len(C), '| с долей >0:', int(bad.sum()), '(%.1f%%)' % (100 * bad.mean()), '| макс доля %.3f' % f.max(), '| ср. доля по плохим %.3f' % (f[bad].mean() if bad.any() else 0),
              '| топ-3 (θ, ω, доля):', [(round(R[i, 1], 2), round(R[i, 2], 2), round(f[i], 2)) for i in top], flush=True)
