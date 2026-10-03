"""research hub-v5chain-research-10: бабочки двойного маятника — споры РАСТУТ обратным деревом от цели (гипотеза 4, `butterfly_dp.md`).
Диагноз research-9: пар 0.2–0.4 на узел, потому что ≈ N·2R·объём 3D-трубки приходов / объём области; при равномерных спорах в
q ∈ T², |w| ≤ 3 трубка (слабые моторы: Δw ≈ M⁻¹τt) почти пуста. Здесь споры кладутся НА трубки: спора A = Φ(B + s·n_B, τ, −t) —
конец обратной дуги из существующей споры B (τ: углы ±1 и случайные, t ∈ [.2, TL], s ∈ [−R, R]); новая — только в непокрытую точку
(расстояние в вложении ≥ DMIN, DMIN уменьшается при низком приёме). Без подсказки запроса: растёт от цели во все стороны.
Пары/Беллман/агент — как в butterfly_dp_atlas.py (Ньютон 4×4, окно WIN, TL). Дуга A→B по построению есть ⇒ ≥ 1 пары на спору."""
import numpy as np, sys, os, time, json
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from butterfly_dp import wrap, flow, ingoal, C3, RQ, RW_, WMAX, BIG, TL, WIN, G
import butterfly_dp_atlas as BA
from butterfly_dp_atlas import Atlas, R, MN
Atlas.pairs.__defaults__ = (None, int(os.environ.get('CHUNK', 800)))                       # куски пар меньше: 4000+4000 при 3000 падали по OOM (1.5 ГБ)
DMIN0 = float(os.environ.get('DMIN', .35)); TR = int(os.environ.get('TR', 0)); UMAX = float(os.environ.get('UMAX', .9)); FR = int(os.environ.get('FR', 0)); FWD = int(os.environ.get('FWD', 0)); START = np.array([-np.pi / 2, 0, 0, 0])
def normals(C, rng):
    w = C[:, 2:]; nn = np.hypot(w[:, 0], w[:, 1]); rnd = rng.normal(size=(len(C), 2)); rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    n = np.zeros((len(C), 4)); n[:, 0] = np.where(nn > .05, -w[:, 1] / np.maximum(nn, 1e-9), rnd[:, 0]); n[:, 1] = np.where(nn > .05, w[:, 0] / np.maximum(nn, 1e-9), rnd[:, 1]); return n
def emb(Y): return np.c_[np.cos(Y[:, 0]), np.sin(Y[:, 0]), np.cos(Y[:, 1]), np.sin(Y[:, 1]), Y[:, 2:] / 2]
class GrowAtlas(Atlas):
    def __init__(s, N, seed=0, batch=400):
        rng = np.random.default_rng(seed); g = np.linspace(-1, 1, 3)
        C = np.array([[C3[0] + a * RQ * .8, C3[1] + b * RQ * .8, c * RW_ * .8, d * RW_ * .8] for a in g for b in g for c in g for d in g])
        n = normals(C, rng); tv = np.zeros(len(C)); dmin = DMIN0; s.grow_log = []; s.lam = []; fw = np.zeros(len(C), bool); it = 0
        if FWD: C = np.r_[C, START[None]]; n = np.r_[n, normals(START[None], rng)]; tv = np.r_[tv, 0.]; fw = np.r_[fw, True]   # FWD: корень прямого дерева — старт запроса
        while len(C) < N + 81 + FWD:
            it += 1; sg = 1. if (FWD and it % 2 == 0 and fw.sum() < FWD + 1) else -1.; pool = np.flatnonzero(fw if sg > 0 else ~fw)
            if sg < 0 and (~fw).sum() >= N + 81: continue
            if sg > 0: B = pool[rng.integers(len(pool), size=batch)]
            elif FR:                                                                           # FR: родитель равномерно по уровню «время до цели по построению» ⇒ фронт уходит дальше
                o = pool[np.argsort(tv[pool])]; L = rng.uniform(0, tv[pool].max() + .3, batch); B = o[np.clip(np.searchsorted(tv[o], L) - rng.integers(0, 8, batch), 0, len(o) - 1)]
            else: B = pool[rng.integers(len(pool), size=batch)]
            u = rng.uniform(-1, 1, (2, batch)); corner = rng.random(batch) < .5
            u[:, corner] = np.sign(u[:, corner]); t = rng.uniform(.2, TL, batch); sv = np.linspace(-R, R, MN)[rng.integers(MN, size=batch)]   # точно в узел споры B: интерполяция V по отрезку не нужна
            if TR: u *= UMAX; sv[:] = 0.                                                     # TR: из центра B, запас по τ для соседних узлов
            A = flow((C[B] + sv[:, None] * n[B]).T, u[0], u[1], sg * t).T
            if TR:                                                                           # нормаль A = перенос нормали B назад по дуге ⇒ узлы A ложатся на отрезок B
                d = (flow((C[B] + 1e-5 * n[B]).T, u[0], u[1], sg * t).T - A) / 1e-5; nA = d / np.linalg.norm(d, axis=1, keepdims=True); s.lam.append(np.linalg.norm(d, axis=1))
            A[:, :2] = wrap(A[:, :2])
            ok = np.isfinite(A).all(1) & (np.abs(A[:, 2:]) <= WMAX).all(1) & ~ingoal(A.T)
            ok &= (np.abs(np.c_[wrap(C[B, :2] - A[:, :2]), C[B, 2:] - A[:, 2:]]) <= WIN).all(1)      # дуга должна попасть в окно пар
            ok &= cKDTree(emb(C)).query(emb(np.where(ok[:, None], A, 0.)))[0] >= dmin
            acc = []
            for i in np.flatnonzero(ok):                                                     # внутри пачки — тоже не ближе dmin
                if not acc or np.min(np.linalg.norm(emb(A[acc]) - emb(A[i:i + 1]), axis=1)) >= dmin: acc.append(i)
            acc = acc[:(FWD + 1 - fw.sum()) if sg > 0 else (N + 81 - (~fw).sum())]; fw = np.r_[fw, np.full(len(acc), sg > 0)]; C = np.r_[C, A[acc]]; tv = np.r_[tv, tv[B[acc]] + t[acc]]; n = np.r_[n, nA[acc] if TR else normals(A[acc], rng)]
            rate = len(acc) / batch; s.grow_log.append((len(C), round(dmin, 3), round(rate, 3)))
            if rate < .05: dmin *= .9
        s.C, s.n, s.K = C, n, len(C); s.dmin = dmin; s.tv = tv; s.fw = fw
        s.sn = np.linspace(-R, R, MN); s.P = s.C[:, None, :] + s.sn[None, :, None] * s.n[:, None, :]
        s.ingoal = ingoal(np.moveaxis(s.P, 2, 0)); s.V = np.full((s.K, MN), BIG); s.V[s.ingoal] = 0.
        s.X = emb(s.C); s.tree = cKDTree(s.X)
if __name__ == '__main__':
    N = int(sys.argv[1]); t0 = time.time(); A = GrowAtlas(N); tg = time.time() - t0
    w = np.abs(A.C[:, 2:]).max(1)
    print(json.dumps(dict(N=N, grow_sec=round(tg, 1), dmin_end=round(A.dmin, 3), w_med=round(float(np.median(w)), 2), w_q90=round(float(np.quantile(w, .9)), 2),
                          q1_cover=round(float(np.histogram(A.C[:, 0], 8, (-np.pi, np.pi))[0].min() / len(A.C) * 8), 2),
                          TR=TR, FR=FR, FWD=FWD, tv_q=np.round(np.quantile(A.tv, [.5, .9, 1]), 2).tolist(), lam_med=round(float(np.median(np.concatenate(A.lam))), 2) if A.lam else None)), flush=True)
    A.build(); tp = time.time() - t0; A.solve()
    print(json.dumps(dict(N=N, G=G, WIN=WIN, TL=TL, R=R, DMIN=DMIN0, nodes=A.K * MN, pairs=len(A.e[0]), pairs_per_node=round(len(A.e[0]) / (A.K * MN), 1), sec_pairs=round(tp), iters=A.n_it,
                          BIG=round(float((A.V >= BIG / 2).mean()), 3), sec=round(time.time() - t0))), flush=True)
    np.savez('butterfly_dp_grow_N%d_tr%d_fr%d_fwd%d_R%g.npz' % (N, TR, FR, FWD, R), C=A.C, n=A.n, V=A.V, e=np.array(A.e, dtype=object), allow_pickle=True)
    if FWD:                                                                                  # research-10: агент по рёбрам графа от споры-старта (butterfly_dp_query.rollout_edges)
        import butterfly_dp_query as Q
        T, arcs, wm = Q.rollout_edges(A, 81, 0.)
        print(json.dumps(dict(query='висит→вверх (рёбра)', V=round(float(A.V[81, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), v6=5.28, arcs=arcs, wmax=round(float(wm), 2),
                              fwd_finite=round(float((A.V[A.fw, MN // 2] < BIG / 2).mean()), 3), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    if os.environ.get('BUILD_ONLY') or FWD: sys.exit()
    q = np.array([-np.pi / 2, 0, 0, 0]); V0 = A.best(q)[0]; T, wm, sw = A.rollout(q)
    print(json.dumps(dict(query='висит→вверх', V=round(float(V0), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), v6=5.28, wmax=round(float(wm), 2), sw=sw), ensure_ascii=False), flush=True)
    rng = np.random.default_rng(3); Q = np.c_[rng.uniform(-np.pi, np.pi, (20, 2)), rng.uniform(-1, 1, (20, 2))]; res = []
    for qq in Q: v = A.best(qq)[0]; T, wm, sw = A.rollout(qq); res.append((v, T, wm, sw))
    R_ = np.array(res); f_ = np.isfinite(R_[:, 1]) & (R_[:, 0] < BIG / 2)
    print(json.dumps(dict(random20_reach=round(float(f_.mean()), 2), V_finite=round(float((R_[:, 0] < BIG / 2).mean()), 2),
                          T_over_V=dict(mean=round(float((R_[f_, 1] / R_[f_, 0]).mean()), 3), max=round(float((R_[f_, 1] / R_[f_, 0]).max()), 3)) if f_.any() else None,
                          T_med=round(float(np.median(R_[f_, 1])), 2) if f_.any() else None, wmax_over3=int((R_[:, 2] > 3).sum()), sec=round(time.time() - t0)), ensure_ascii=False), flush=True)
    np.save('butterfly_dp_grow_N%d_d%g_tr%d.npy' % (N, DMIN0, TR), R_)
