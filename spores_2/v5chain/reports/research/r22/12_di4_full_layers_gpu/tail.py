"""research-22, эксп. 12б: хвост агента на di4 L1600 SBLAY 1 — что происходит на худших стартах (T/эт до ×8)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
DTN = G.DTN; US = G.US; BIG = G.BIG
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = np.load('V_sblay1.npy'); T = np.load('T_sblay1.npy'); r = np.load('di4_ref_60_rho35.npy'); Q = r[:, :4]; TR = r[:, 6]; V0 = A.vstar(Q)
q = np.where(np.isfinite(T), T / TR, -1); o = np.argsort(-q)[:6]
for i in o: print('старт %2d Q %s | эталон %.2f V(старт) %.2f T агента %.2f (×%.2f)' % (i, np.round(Q[i], 2), TR[i], V0[i], T[i], q[i]), flush=True)
i = o[0]; y = Q[i:i + 1].copy(); t = 0.; print('--- старт', i, ': t | состояние | V(y) | J по 4 u | выбор')
for n in range(int(T[i] / DTN) + 2):
    J = DTN + np.array([A.vstar(G.step(y, u))[0] for u in US]); tg = np.array([A.tgoal(y, u)[0] for u in US]); J = np.minimum(J, tg); k = int(J.argmin())
    if n % 4 == 0 or n < 6: print('%5.1f | %s | %.2f | %s | %d' % (t, np.round(y[0], 2), A.vstar(y)[0], np.round(np.minimum(J, 99), 2), k), flush=True)
    if G.ingoal(y)[0] or J[k] >= BIG / 2: break
    y = G.step(y, US[k]); t += DTN
