"""research-22, эксп. 01: п.37 на di4 L400 — какой V верен? Сравнение V_ref (все стенсилы, Якоби) и V_new (SBCAUS, SBLAY 1/0)
с точной нижней оценкой LB = max_i Tbox_i (1D ДИ в коробку |x|,|v| <= RHO: перебор точек границы, 1 переключение)."""
import sys, os, pickle, numpy as np
sys.path.insert(0, '.')
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from tqdm import tqdm
R = float(os.environ['RHO'])
def tbox(x0, v0, K=81):
    s_ = np.linspace(-R, R, K); tx = np.r_[s_, s_, np.full(K, -R), np.full(K, R)]; tv = np.r_[np.full(K, -R), np.full(K, R), s_, s_]
    out = np.full(len(x0), np.inf)
    for a in tqdm(range(0, len(x0), 20000), desc='tbox', mininterval=10):
        x = x0[a:a + 20000, None]; v = v0[a:a + 20000, None]; best = np.full(x.shape[0], np.inf)
        for s in (1., -1.):
            q = s * (tx - x) + (v * v + tv * tv) / 2; vm = s * np.sqrt(np.maximum(q, 0)); t1 = s * (vm - v); t2 = s * (vm - tv)
            T = np.where((q >= 0) & (t1 >= -1e-12) & (t2 >= -1e-12), t1 + t2, np.inf); best = np.minimum(best, T.min(1))
        out[a:a + 20000] = best
    out[(np.abs(x0) <= R) & (np.abs(v0) <= R)] = 0.; return out
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); P = A.P
LB = np.maximum(tbox(P[:, 0], P[:, 2]), tbox(P[:, 1], P[:, 3])); cl = np.concatenate([[k] * (len(c.G.reshape(-1, 4))) for k, l in enumerate(A.layers) for c in l])
vr = pickle.load(open(os.environ['VREF'], 'rb')); Vs = {'ref': vr['V'] if isinstance(vr, dict) else vr}
for n_ in sys.argv[1:]: Vs[n_] = pickle.load(open(n_, 'rb'))
far = LB > .5
for k, V in Vs.items():
    f = (V < G.BIG / 2) & far; r = V[f] / LB[f]
    print('%-12s конечных %d (%.3f) | V/LB: mean %.3f med %.3f p5 %.3f p95 %.3f max %.2f | доля V < 0.98 LB: %.3f | V < 0.9 LB: %.3f | по слоям конечных %s' % (k, (V < G.BIG / 2).sum(), (V < G.BIG / 2).mean(), r.mean(), np.median(r), *np.percentile(r, [5, 95]), r.max(), (r < .98).mean(), (r < .9).mean(), [round(float((V[cl == j] < G.BIG / 2).mean()), 3) for j in range(len(A.layers))]), flush=True)
ks = list(Vs)
for k in ks[1:]:
    f = (Vs['ref'] < G.BIG / 2) & (Vs[k] < G.BIG / 2); dv = Vs[k][f] - Vs['ref'][f]; lost = (Vs['ref'] < G.BIG / 2) & ~(Vs[k] < G.BIG / 2)
    print('%-12s − ref: mean %.3f med %.3f p5 %.3f p95 %.3f | потеряно узлов %d, их LB mean %.2f, V_ref/LB mean %.3f' % (k, dv.mean(), np.median(dv), *np.percentile(dv, [5, 95]), lost.sum(), LB[lost].mean() if lost.any() else 0, (Vs['ref'][lost & far] / LB[lost & far]).mean() if (lost & far).any() else 0), flush=True)
np.save('LB.npy', LB)
