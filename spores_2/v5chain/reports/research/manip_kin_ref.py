"""research: эталон для кинематического 2-звенного манипулятора (q̇ = u, |u_i| ≤ ω, тор T²) — для v6 ступеней 1–2 (manipulator_plan.md).
Без препятствий: T* = max_i |Δq_i|_{mod 2π} / ω. С препятствиями (диски в рабочем пространстве, звенья — отрезки L1, L2 из начала):
Дейкстра на сетке тора n×n с 8 соседями и ценой хода max(|Δq1|,|Δq2|)/ω — в свободном пространстве это ТОЧНО метрика L∞ (Чебышёв),
ошибка только от разрешения сетки у препятствий. API: t_free(q0, qg, w); grid_T(qg, obstacles, n, w) -> T[n,n] (inf — занято/недостижимо)."""
import numpy as np, heapq
wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
def t_free(q0, qg, w=1.0): return np.max(np.abs(wrap(np.asarray(qg) - np.asarray(q0)))) / w
def seg_disc(p, q, c, r):                          # отрезок p–q пересекает диск (c, r)?
    d = q - p; t = np.clip(np.dot(c - p, d) / max(np.dot(d, d), 1e-12), 0, 1); return np.linalg.norm(p + t*d - c) <= r
def collide(q, obstacles, L1=1.0, L2=1.0):
    e = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])]); t = e + L2*np.array([np.cos(q[0]+q[1]), np.sin(q[0]+q[1])])
    return any(seg_disc(np.zeros(2), e, np.array(c), r) or seg_disc(e, t, np.array(c), r) for c, r in obstacles)
def grid_T(qg, obstacles=(), n=128, w=1.0):
    g = -np.pi + 2*np.pi*np.arange(n)/n; h = 2*np.pi/n
    free = np.array([[not collide((a, b), obstacles) for b in g] for a in g])
    gi = tuple(int(round((wrap(x) + np.pi)/h)) % n for x in qg); T = np.full((n, n), np.inf)
    if not free[gi]: return g, T
    T[gi] = 0.0; pq = [(0.0, gi)]
    while pq:
        t, (i, j) = heapq.heappop(pq)
        if t > T[i, j]: continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0: continue
                a, b = (i+di) % n, (j+dj) % n
                if free[a, b] and t + h/w < T[a, b]: T[a, b] = t + h/w; heapq.heappush(pq, (T[a, b], (a, b)))
    return g, T
if __name__ == "__main__":
    n = 96; g, T = grid_T((0.0, 0.0), (), n)
    err = max(abs(T[i, j] - t_free((g[i], g[j]), (0, 0))) for i in range(n) for j in range(n))
    print(f"без препятствий: max |T_сетка − T*| = {err:.2e} (должно быть ~0: L∞ точна на сетке)")
    obs = [((1.2, 0.8), 0.3), ((-0.5, 1.4), 0.25)]; g, T2 = grid_T((0.0, 0.0), obs, n)
    fin = np.isfinite(T2); occ = np.mean([[collide((a, b), obs) for b in g] for a in g])
    print(f"с 2 дисками: занято {occ:.1%} тора, достижимо {fin.mean():.1%}, T max {T2[fin].max():.3f} против {T[np.isfinite(T)].max():.3f} без препятствий, "
          f"обход дороже в {np.nanmean(np.where(fin, T2/np.maximum(T, 1e-9), np.nan)[T > 0]):.3f} раза в среднем")
