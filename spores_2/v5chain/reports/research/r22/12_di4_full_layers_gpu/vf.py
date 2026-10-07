"""research-22, эксп. 12в: хвост агента di4 L1600 = нет финиша у цели (клетка ≈ коробка цели, V там плоская). Штатный growN.rollout с финишем стрельбой VF (env), V из эксп. 12."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); r = np.load('di4_ref_60_rho35.npy'); Q = r[:, :4]; TR = r[:, 6]
stop = np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]; inf_ = np.abs(stop).max(1) <= 2.5
for k in (1, 0):
    A.V = np.load('V_sblay%d.npy' % k); t0 = time.time(); T, sw, path = A.rollout(Q); dt = time.time() - t0; ok = np.isfinite(T); q = T / TR; m = ok & inf_
    print('VF %s SBLAY %d | дошли %d/60 (в поле %d/%d) | T/эт в поле: мед. %.3f mean %.3f p90 %.3f max %.2f | T < эталона − 1e-6 у %d | переключений мед. %d max %d | %.0f с на 60 (%.0f мс/запрос)' % (os.environ.get('VF', '0'), k, ok.sum(), m.sum(), inf_.sum(), np.median(q[m]), q[m].mean(), np.percentile(q[m], 90), q[m].max(), (T[m] < TR[m] - 1e-6).sum(), np.median(sw[m]), sw[m].max(), dt, 1e3 * dt / 60), flush=True)
