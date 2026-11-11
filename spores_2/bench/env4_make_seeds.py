"""ENV4: точки посева из траекторий OCP ОБУЧАЮЩИХ стартов (seed 1, не тестовых): env4_q<i>.npz (X: 6×(N+1) в обычных координатах) → SEEDS.npz (P: сдвиг y−2, θ обёрнут), порядок — по времени до цели (ближние к цели первыми, рост идёт от цели наружу).
Запуск: python3 env4_make_seeds.py DIR OUT.npz [STEP=2]"""
import sys, glob, numpy as np
d, out = sys.argv[1], sys.argv[2]; st = int(sys.argv[3]) if len(sys.argv) > 3 else 2; P = []; R = []
for f in glob.glob(d + '/env4_q*.npz'):
    z = np.load(f); X = z['X'].T.copy(); T = float(z['T']); N = len(X) - 1; X[:, 1] -= 2.; X[:, 2] = (X[:, 2] + np.pi) % (2 * np.pi) - np.pi
    for k in range(0, N, st): P.append(X[k]); R.append(T * (1 - k / N))
o = np.argsort(R); np.savez(out, P=np.array(P)[o], rem=np.array(R)[o]); print(len(P), 'точек из', len(glob.glob(d + '/env4_q*.npz')), 'траекторий')
