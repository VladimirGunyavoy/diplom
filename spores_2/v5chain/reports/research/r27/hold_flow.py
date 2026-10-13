"""research-27: 8D (m8) — выживаемость рёбер дерева ±1 при пределе |w|<=WM и прототип слоя «удержание на пределе» (аналог режима a=0 у ДИ с |v|<=vmax).
Слой s (знаки ±1 по 4 моторам) + правило: сустав j на пределе (|w_j|>=WM-D и ускорение наружу) → u_j берётся из условия «к концу шага H ровно на пределе» (qdd аффинно по u), если |u_j|<=1.
Замер: все 16^3 последовательностей глубины 3 (tau=.4) от каждого из 20 стартов: доля выживших (|w|<=WM в концах рёбер, как S.ok в _tree) — обычный поток против потока с удержанием.
python hold_flow.py  (хаб, numpy, ~1–2 мин)"""
import os, sys, json, itertools, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'bench'))
import numpy as np, common, envs_6d
from tqdm import tqdm
e = common.make('m8'); n = 4; WM = 3.; D = .02; H = .01; TAU = .4
US = np.array([[1.0 if (s >> i) & 1 else -1.0 for i in range(n)] for s in range(16)])
def rk4(Y, U, h):
    k1 = e.f(Y, U); k2 = e.f(Y + h / 2 * k1, U); k3 = e.f(Y + h / 2 * k2, U); k4 = e.f(Y + h * k3, U); return Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
def hold_u(Y, U):
    """U (B,4) знаки слоя → U_eff: на суставах у предела с ускорением наружу u_j из qdd_j=0. Возврат (U_eff, bad): bad — удержать нельзя (|u|>1)."""
    B = len(Y); a0 = e.f(Y, np.zeros((B, n)))[:, n:]; J = np.stack([e.f(Y, np.tile(np.eye(n)[j], (B, 1)))[:, n:] - a0 for j in range(n)], -1)   # qdd = a0 + J u
    Ue = U.copy(); S = np.zeros((B, n), bool); bad = np.zeros(B, bool); w = Y[:, n:]
    for _ in range(n):
        qdd = a0 + np.einsum('bij,bj->bi', J, Ue); wn = w + H * qdd; new = (np.abs(wn) > WM) & ~S               # упреждение: за шаг H вышли бы за предел
        tg = (WM * np.sign(wn) - w) / H                                                                           # qdd, приводящее ровно на предел к концу шага
        if not new.any(): break
        S |= new
        for b in np.where(new.any(1))[0]:
            s = S[b]; fr = ~s; us = np.linalg.solve(J[b][np.ix_(s, s)], tg[b][s] - (a0[b][s] + J[b][np.ix_(s, fr)] @ U[b][fr])); Ue[b, s] = np.clip(us, -1, 1); bad[b] |= bool((np.abs(us) > 1 + 1e-6).any())
    return Ue, bad
def flow(Y, U, tau, hold):
    k = int(round(tau / H)); bad = np.zeros(len(Y), bool); Y = Y.copy()
    for _ in range(k):
        if hold: Ue, b = hold_u(Y, U); bad |= b
        else: Ue = U
        Y = rk4(Y, Ue, H)
    return Y, bad
if __name__ == '__main__':
    R = []; DEPTH = 3
    for i in tqdm(range(len(e.X0)), desc='старты', mininterval=10):
        row = dict(i=i)
        for hold in (0, 1):
            Y = e.X0[i][None].copy(); alive = []
            for d in range(DEPTH):
                Y = np.repeat(Y, 16, 0); U = np.tile(US, (len(Y) // 16, 1)); Y, bad = flow(Y, U, TAU, hold)
                ok = (np.abs(Y[:, n:]).max(1) <= WM + (.01 if hold else 0)) & ~bad; alive.append(float(ok.sum()) / 16 ** (d + 1)); Y = Y[ok]
                if not len(Y): alive += [0.] * (DEPTH - d - 1); break
            row['hold' if hold else 'plain'] = [round(a, 4) for a in alive]
        R.append(row); print(row, flush=True)
    json.dump(R, open(os.path.join(HERE, 'hold_flow.json'), 'w'), indent=1)
