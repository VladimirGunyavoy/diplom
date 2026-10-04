"""research-11: g = 2 — выбор родителя при росте деревьев (только центры, без пар): MODE=fr — как в butterfly_dp_grow (равномерно по уровню времени,
случайное τ), MODE=rrt — смещение Вороного (случайная точка области → ближайшая спора дерева → лучшая дуга веера к точке; новая — в непокрытую).
Оценка (эталон OCP только для оценки): расстояние в вложении от точек пути OCP до прямого/обратного дерева, время по построению, мин. расстояние
между деревьями. Запуск: G=2 WIN=1.0 MODE=rrt python3 dp_g2_voronoi.py N"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, ingoal, C3, RQ, RW_, WMAX, TL, WIN, G
from butterfly_dp_grow import emb
from scipy.spatial import cKDTree
MODE = os.environ.get('MODE', 'rrt'); N = int(sys.argv[1]); DMIN0 = float(os.environ.get('DMIN', .2)); UMAX = .9; SEED = int(os.environ.get('SEED', 0)); KN = int(os.environ.get('KN', 1)); KF = float(os.environ.get('KF', 1.5))
START = np.array([-np.pi / 2, 0, 0, 0.]); rng = np.random.default_rng(SEED)
UG = np.array([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]) * UMAX; TG = np.array([.25, .5, .9, 1.5]); TG = TG[TG <= TL + 1e-9]
UF = np.repeat(UG, len(TG), 0); TF = np.tile(TG, len(UG))
def grow(C, sg, N, b=300):
    tv = np.zeros(len(C)); dmin = DMIN0; par = -np.ones(len(C), int)
    while len(C) < N:
        if MODE == 'fr':
            o = np.argsort(tv); L = rng.uniform(0, tv.max() + .3, b); B = o[np.clip(np.searchsorted(tv[o], L) - rng.integers(0, 8, b), 0, len(o) - 1)]
            u = rng.uniform(-1, 1, (2, b)); c = rng.random(b) < .5; u[:, c] = np.sign(u[:, c]); u *= UMAX; t = rng.uniform(.2, TL, b)
            A = flow(C[B].T, u[0], u[1], sg * t).T
        else:
            X = np.c_[rng.uniform(-np.pi, np.pi, (b, 2)), rng.uniform(-WMAX, WMAX, (b, 2))]; dk, Bk_ = cKDTree(emb(C)).query(emb(X), k=min(KN, len(C))); dk, Bk_ = dk.reshape(b, -1), Bk_.reshape(b, -1)
            cst = np.where(dk <= dk[:, :1] * KF + .05, tv[Bk_], np.inf); B = Bk_[np.arange(b), cst.argmin(1)]   # KN: самый дешёвый по времени из k ближайших
            Y = np.repeat(C[B], len(TF), 0); Z = flow(Y.T, np.tile(UF[:, 0], b), np.tile(UF[:, 1], b), sg * np.tile(TF, b)).T
            okz = np.isfinite(Z).all(1) & (np.abs(Z[:, 2:]) <= WMAX).all(1) & (np.abs(np.c_[wrap(Y[:, :2] - Z[:, :2]), Y[:, 2:] - Z[:, 2:]]) <= WIN).all(1)
            d = np.linalg.norm(emb(np.nan_to_num(Z)) - np.repeat(emb(X), len(TF), 0), axis=1); d[~okz] = np.inf; d = d.reshape(b, -1); m = d.argmin(1)
            A = Z.reshape(b, -1, 4)[np.arange(b), m]; t = TF[m]; A[~np.isfinite(d.min(1))] = np.nan
        A[:, :2] = wrap(A[:, :2]); ok = np.isfinite(A).all(1); A = np.nan_to_num(A)
        ok &= (np.abs(A[:, 2:]) <= WMAX).all(1) & ~ingoal(A.T) & (np.abs(np.c_[wrap(C[B, :2] - A[:, :2]), C[B, 2:] - A[:, 2:]]) <= WIN).all(1)
        ok &= cKDTree(emb(C)).query(emb(A))[0] >= dmin; acc = []
        for i in np.flatnonzero(ok):
            if not acc or np.min(np.linalg.norm(emb(A[acc]) - emb(A[i:i + 1]), axis=1)) >= dmin: acc.append(i)
        acc = acc[:N - len(C)]; C = np.r_[C, A[acc]]; tv = np.r_[tv, tv[B[acc]] + t[acc]]; par = np.r_[par, B[acc]]
        if len(acc) / b < .05: dmin *= .9
    return C, tv, dmin
g3 = np.linspace(-1, 1, 3); CG = np.array([[C3[0] + a * RQ * .8, C3[1] + b * RQ * .8, c * RW_ * .8, d * RW_ * .8] for a in g3 for b in g3 for c in g3 for d in g3])
a = np.load('refdp_G2_W3_N80.npy'); T = a[0]; u = a[1:].reshape(80, 2); h = T / 80; z = START.copy(); P = [z]
for k in range(80): z = flow(z[:, None], u[k:k + 1, 0], u[k:k + 1, 1], h, n=8)[:, 0]; P.append(z)
P = np.array(P); P[:, :2] = wrap(P[:, :2]); tt = np.arange(81) * h; t0 = time.time()
F, tf, df_ = grow(START[None].copy(), 1., N); Bk, tb, db_ = grow(CG.copy(), -1., N)
dF, jF = cKDTree(emb(F)).query(emb(P)); dB, jB = cKDTree(emb(Bk)).query(emb(P)); dFB = cKDTree(emb(Bk)).query(emb(F))[0]
print(json.dumps(dict(MODE=MODE, N=N, G=G, sec=round(time.time() - t0), dmin=[round(df_, 3), round(db_, 3)], tv_max=[round(float(tf.max()), 1), round(float(tb.max()), 1)],
      path_cov_fwd=round(float((dF < .3).mean()), 2), path_cov_bwd=round(float((dB < .3).mean()), 2), path_cov_any=round(float((np.minimum(dF, dB) < .3).mean()), 2),
      path_d_any_max=round(float(np.minimum(dF, dB).max()), 2), trees_dmin=round(float(dFB.min()), 3), trees_lt_dmin2=int((dFB < .2).sum()))))
both = (dF < .35) & (dB < .35); print(json.dumps(dict(KN=KN, KF=KF, est_T_via_path=round(float((tf[jF] + tb[jB])[both].min()), 2) if both.any() else None, n_both=int(both.sum()), tvF_at_path=np.round(tf[jF][::16], 1).tolist(), tvB_at_path=np.round(tb[jB][::16], 1).tolist())))
sys.exit()
print(' t   | d_fwd  tv_f | d_bwd  tv_b (остаток OCP)')
for k in range(0, 81, 5): print('%4.2f | %.2f %5.2f | %.2f %5.2f (%.2f)' % (tt[k], dF[k], tf[jF[k]], dB[k], tb[jB[k]], T - tt[k]))
