"""research hub-v5chain-research-10: эталон времени дифдрайва (ромб |v| + |ω| ≤ 1, точка-робот) с дисками-препятствиями, цель (0, 0, 0).
Довод: путь = прямые (|v| = 1, передом или задом) + повороты на месте (|ω| = 1), время = длина + суммарный поворот. Обход диска радиуса ρ на угол β
ломаной из k хорд стоит → ρβ + β, ровно как дуга по ребру ромба (v = ρω, |v| + |ω| = 1 ⇒ время (ρ + 1)β) ⇒ граф «прямые + повороты» не теряет дуг.
Граф: точки = цель + KB точек на окружностях радиуса r/cos(π/KB) (хорды не задевают диск) + сетка NG×NG (промежуточные вершины TGTGT);
ребро = видимая пара, курс = направление (передом) или направление + π (задом). Цена-до-цели по рёбрам: togo(e) = min(финиш: |θ_e|, min_e' |θ_e' − θ_e|
+ len(e') + togo(e')), итерации векторно по узлам. Запрос: min_e |θ0 − θ_e| + len(e) + togo(e) по рёбрам из старта. Верхняя оценка (сетка, хорды)."""
import numpy as np, os, sys, json, time
sys.path.insert(0, '.')
from dd_rhombus_ref import tgt, tgtgt
DISKS = np.array([[1.0, 0.3, .45], [-0.8, -0.9, .4]]) if not os.environ.get('NODISK') else np.zeros((0, 3))
KB, NG, LIM = int(os.environ.get('KB', 40)), int(os.environ.get('NG', 11)), float(os.environ.get('LIM', 2.6))
def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def free_pts(P): return np.all([np.hypot(P[:, 0] - d[0], P[:, 1] - d[1]) > d[2] for d in DISKS], 0) if len(DISKS) else np.ones(len(P), bool)
def visible(A, B):
    """A (n,2), B (n,2): отрезок не входит внутрь дисков."""
    ok = np.ones(len(A), bool)
    for cx, cy, r in DISKS:
        d = B - A; f = A - [cx, cy]; L2 = np.maximum((d ** 2).sum(1), 1e-18); t = np.clip(-(f * d).sum(1) / L2, 0, 1)
        ok &= np.hypot(*(f + t[:, None] * d).T) >= r - 1e-9
    return ok
def build():
    P = [np.zeros((1, 2))]
    for cx, cy, r in DISKS:
        a = np.linspace(0, 2 * np.pi, KB, endpoint=False); P.append(np.c_[cx + r / np.cos(np.pi / KB) * (1 + 1e-6) * np.cos(a), cy + r / np.cos(np.pi / KB) * (1 + 1e-6) * np.sin(a)])
    g = np.linspace(-LIM, LIM, NG); G = np.array([(x, y) for x in g for y in g]); P.append(G[free_pts(G)]); P = np.concatenate(P)
    n = len(P); I, J = np.nonzero(~np.eye(n, dtype=bool)); ok = visible(P[I], P[J]); I, J = I[ok], J[ok]
    d = P[J] - P[I]; L = np.hypot(*d.T); h = np.arctan2(d[:, 1], d[:, 0])
    E = dict(src=np.r_[I, I], dst=np.r_[J, J], L=np.r_[L, L], h=np.r_[h, wrap(h + np.pi)])        # передом / задом
    return P, E
def togo(P, E, it=60):
    src, dst, L, h = E['src'], E['dst'], E['L'], E['h']; n = len(P); out = [np.flatnonzero(src == i) for i in range(n)]; inn = [np.flatnonzero(dst == i) for i in range(n)]
    T = np.where(dst == 0, np.abs(wrap(h)), np.inf)
    for k in range(it):
        T0 = T.copy()
        for i in range(n):
            ei, eo = inn[i], out[i]
            if not len(ei) or not len(eo): continue
            c = np.abs(wrap(h[eo][None, :] - h[ei][:, None])) + (L[eo] + T[eo])[None, :]; v = c.min(1)
            T[ei] = np.minimum(T[ei], v if i else np.minimum(v, np.abs(wrap(h[ei]))))
        if np.allclose(T, T0, equal_nan=True): break
    return T, k
def query(P, E, T, q):
    A = np.repeat(q[None, :2], len(P), 0); ok = visible(A, P); ok[np.hypot(*(P - q[:2]).T) < 1e-12] = False; best = np.inf
    for j in np.flatnonzero(ok):
        d = P[j] - q[:2]; L = np.hypot(*d); hh = np.arctan2(d[1], d[0])
        for hd in (hh, wrap(hh + np.pi)):
            m = (E['src'] == j); c = np.abs(wrap(hd - q[2])) + L + (np.abs(wrap(hd)) if j == 0 else np.min(np.abs(wrap(E['h'][m] - hd)) + E['L'][m] + T[m]))
            best = min(best, c)
    return best
if __name__ == '__main__':
    t0 = time.time(); P, E = build(); T, k = togo(P, E)
    rng = np.random.default_rng(1); Q = np.c_[rng.uniform(-2, 2, (120, 2)), rng.uniform(-np.pi, np.pi, 120)]; Q = Q[free_pts(Q)][:60]
    R = np.array([query(P, E, T, q) for q in Q]); out = dict(disks=len(DISKS), KB=KB, NG=NG, pts=len(P), edges=len(E['L']), iters=k, ref_mean=round(float(R.mean()), 4), sec=round(time.time() - t0))
    if not len(DISKS):
        ex = np.array([min(tgt(*q), tgtgt(*q)) for q in Q]); out.update(exact_mean=round(float(ex.mean()), 4), ratio_mean=round(float((R / ex).mean()), 4), ratio_max=round(float((R / ex).max()), 4))
    print(json.dumps(out), flush=True); np.save('dd_obst_ref_KB%d_NG%d%s.npy' % (KB, NG, '_nodisk' if not len(DISKS) else ''), np.c_[Q, R])
