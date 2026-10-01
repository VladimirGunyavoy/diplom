"""Данные для research-7 (только вход→выход, без боковых переходов): для клеток слоя — куда попадают выходы.
Для каждой клетки C: точка выхода E=X(τ) и сегмент выхода E±s·V(τ), |s|≤r. Для остальных клеток D того же слоя (и другого слоя — отдельно):
 • вход D = сегмент c_D ± s·n_D, |s|≤r_D (t=0). Зазор = расстояние от E до ближайшего входа (в норм. координатах и в единицах r_D);
 • E в ядре D: (s,t) по модели → доля с t/τ_D ≤ 0.1 («почти вход»), распределение t/τ_D; 
 • покрытие сегмента выхода входами: доля из 11 точек сегмента, лежащих в ядре какой-либо D с t/τ_D ≤ 0.1 (|s_D|≤r_D)."""
import sys, os, json, pickle, time; sys.path.insert(0, '.')
import numpy as np
from src.cells7.systems import di, pend
from src.cells7.cover import cover_layer
from src.cells7.cell import Cell
name, m = sys.argv[1], int(sys.argv[2]); S = di() if name == 'di' else pend()
def get(k):
    fn = f'/tmp/cells_{name}_{m}_{k}_0.pkl'
    if os.path.exists(fn): return [Cell(S, k, c, r=r, tau=tau, tol=1e9) for c, r, tau in pickle.load(open(fn, 'rb'))]
    L = cover_layer(S, k, m=m, seed=0)[0]; pickle.dump([(C.c, C.r, C.tau) for C in L], open(fn, 'wb')); return L
layers = [get(0), get(1)]; out = {'system': name, 'm': m, 'cells': [len(L) for L in layers]}
def seg_dist(S, p, c, n, r):
    d = S.wrap(p - c); s = np.clip(d @ n, -r, r); return float(np.linalg.norm(S.wrap(d - s * n))), float(s)
for k in (0, 1):
    L = layers[k]; res = dict(gap=[], gap_r=[], in_kernel=[], t_frac=[], near_entry=[], seg_cov=[], lateral_in_entry=[])
    for i, C in enumerate(L):
        E = C.X[-1]; g = []
        for j, D in enumerate(L):
            if j == i: continue
            dist, s = seg_dist(S, E, D.c, D.n, D.r); g.append((dist / D.r, dist))
        gr, gd = min(g); res['gap_r'].append(gr); res['gap'].append(gd)
        tf = []; ins_any = False; ne = False
        for j, D in enumerate(L):
            if j == i: continue
            s_, t_, ins = D.locate(E[None], 1.0)
            if ins[0]: ins_any = True; tf.append(float(t_[0] / D.tau)); ne = ne or (t_[0] / D.tau <= 0.1)
        res['in_kernel'].append(ins_any); res['near_entry'].append(ne); res['t_frac'] += tf
        Y = E[None] + np.linspace(-C.r, C.r, 11)[:, None] * C.V[-1][None]; cov = np.zeros(11, bool)
        for j, D in enumerate(L):
            if j == i: continue
            s_, t_, ins = D.locate(Y, 1.0); cov |= ins & (t_ / D.tau <= 0.1)
        res['seg_cov'].append(float(cov.mean()))
    q = lambda a: [float(x) for x in np.quantile(a, [.1, .25, .5, .75, .9])]
    out[f'layer{k}'] = dict(n=len(L), exit_in_other_kernel=float(np.mean(res['in_kernel'])), exit_near_entry_t_le_0p1=float(np.mean(res['near_entry'])), gap_to_nearest_entry_in_r_quantiles_10_25_50_75_90=q(res['gap_r']),
                            gap_norm_quantiles=q(res['gap']), t_frac_of_hits_quantiles=q(res['t_frac']) if res['t_frac'] else None, exit_segment_covered_by_entries_mean=float(np.mean(res['seg_cov'])), exit_segment_covered_by_entries_quantiles=q(res['seg_cov']))
json.dump(out, open(f'reports/exits_{name}.json', 'w'), indent=1); print(json.dumps(out, indent=1))
