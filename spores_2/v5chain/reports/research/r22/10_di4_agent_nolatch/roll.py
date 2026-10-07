"""research-22, эксп. 10: агент (growN.rollout) на di4 L400 при V без защёлки и разных PESS. 60 стартов rng(1) [−2, 2]⁴; мерило — T / LB (точная нижняя оценка в коробку ±RHO)."""
import sys, os, time, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
R = float(os.environ['RHO']); name = os.environ['VN']
def tbox(x0, v0, K=161):
    s_ = np.linspace(-R, R, K); tx = np.r_[s_, s_, np.full(K, -R), np.full(K, R)]; tv = np.r_[np.full(K, -R), np.full(K, R), s_, s_]; x = x0[:, None]; v = v0[:, None]; best = np.full(len(x0), np.inf)
    for s in (1., -1.):
        q = s * (tx - x) + (v * v + tv * tv) / 2; vm = s * np.sqrt(np.maximum(q, 0)); t1 = s * (vm - v); t2 = s * (vm - tv); best = np.minimum(best, np.where((q >= 0) & (t1 >= -1e-12) & (t2 >= -1e-12), t1 + t2, np.inf).min(1))
    best[(np.abs(x0) <= R) & (np.abs(v0) <= R)] = 0.; return best
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); A.V = np.load('V_' + name + '.npy')
Q = np.random.default_rng(1).uniform(-2, 2, (60, 4)); LB = np.maximum(tbox(Q[:, 0], Q[:, 2]), tbox(Q[:, 1], Q[:, 3])); V0 = A.vstar(Q); t0 = time.time(); T, sw, path = A.rollout(Q); dt = time.time() - t0; ok = np.isfinite(T)
r = T[ok] / LB[ok]; v0 = V0[ok & (V0 < G.BIG / 2)] / LB[ok & (V0 < G.BIG / 2)]
print('%-13s PESS %s | дошли %d/60 | T/LB mean %.3f мед. %.3f max %.2f | V(старт)/LB мед. %.3f | T/V(старт) мед. %.3f | переключений mean %.1f max %d | rollout %.0f с' % (name, os.environ['PESS'], ok.sum(), r.mean(), np.median(r), r.max(), np.median(v0), np.median(T[ok & (V0 < G.BIG / 2)] / V0[ok & (V0 < G.BIG / 2)]), sw[ok].mean(), sw[ok].max(), dt), flush=True)
np.save('T_' + name + '.npy', T)
