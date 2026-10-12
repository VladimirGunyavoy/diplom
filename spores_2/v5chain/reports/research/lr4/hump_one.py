"""lr4: из чего состоит горб — вдоль одной оптимальной траектории: T_ref, Q своего слоя («продолжать»), Q слоя, на который надо переключиться («переключиться сейчас»), V* = min. Q lin, env M. Запуск из v8."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, hump as H
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho; uvec = np.array(G.us); V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q
X0 = np.array([[-1.5, 0.], [1., 1.], [-2., .8], [1.8, -.3]]); U0, T1, TT = H.opt_switch(X0, rho); TG = np.arange(-.6, .3001, .005); fig, ax = plt.subplots(1, len(X0), figsize=(5.2 * len(X0), 4.6))
for a, x0, u, t1, T in zip(ax, X0, U0, T1, TT):
    t = t1 + TG; ta = np.minimum(t, t1); tb = np.maximum(t - t1, 0.); x1 = x0[0] + x0[1] * ta + u * ta * ta / 2; v1 = x0[1] + u * ta; P = np.stack([x1 + v1 * tb - u * tb * tb / 2, v1 - u * tb], 1); Tr = T - t; Vl = G.Vlayers(P); io = int(np.flatnonzero(uvec == u)[0]); it2 = int(np.flatnonzero(uvec == -u)[0])
    a.plot(TG, Tr, 'k-', lw=2, label='эталон T_ref'); a.plot(TG, Vl[:, io], color='tab:red', label='Q своего слоя u=%+g («продолжать»)' % u); a.plot(TG, Vl[:, it2], color='tab:purple', label='Q слоя u=%+g («переключиться сейчас»)' % -u); a.plot(TG, Vl.min(1), 'k--', lw=1.2, label='V* = min по слоям')
    a.axvline(0, color='gray', lw=.8); a.set_ylim(Tr.min() - .05, Tr.max() + .45); a.set_xlabel('t − t_sw, с'); a.set_title('старт (%g, %g), M=%d' % (x0[0], x0[1], G.M)); a.grid(alpha=.3); a.legend(fontsize=7)
    k = np.argmax(Vl.min(1) - Tr); print('RES старт %s: max (V*−T_ref) = %.3f при t−t_sw = %+.3f; там Q своего − T_ref = %.3f, Q чужого − T_ref = %.3f; при t−t_sw=−.5: %.3f / %.3f' % (x0.tolist(), (Vl.min(1) - Tr)[k], TG[k], Vl[k, io] - Tr[k], Vl[k, it2] - Tr[k], Vl[20, io] - Tr[20], Vl[20, it2] - Tr[20]), flush=True)
fig.tight_layout(); fig.savefig('hump_one_M%d.png' % G.M, dpi=110)
