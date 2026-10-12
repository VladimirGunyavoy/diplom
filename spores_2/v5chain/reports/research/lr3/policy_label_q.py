"""lr3: метка политики узла ПОСЛЕ Q-решения (qsolve.solve_q): «своё управление», если V узла = Q узла, иначе «переключение».
Вопрос: становится ли метка чистым автодетектором кривой переключения (на старой V она шумела: 34–52% ячеек смешанные)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS
from src.algo import ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas, 'N'); BIG = S.BIG; rho = G.rho; M = G.M
V, Q, it, ts = QS.solve_q(G, 0.); fin = V < BIG / 2; T, info = R.T_ref(G.P, rho, return_details=True); out = np.linalg.norm(G.P, axis=1) > rho; s_true = np.array([i[0] for i in info], float)
sw = fin & (V < Q - 1e-9); u_own = np.array(G.us)[G.layer]; u_lab = np.where(sw, -u_own, u_own); ok = fin & out
d = np.abs(G.P[:, 0] + G.P[:, 1] * np.abs(G.P[:, 1]) / 2) / np.sqrt(1 + G.P[:, 1] ** 2); bad = ok & (u_lab != s_true)
print('RES Q-метка: конечных %d; совпадает с истинным первым управлением %.3f; «переключение» у %.3f узлов; ошибочные: расстояние до кривой мед/p90 %.3f/%.3f (все узлы мед %.3f)' % (ok.sum(), (u_lab[ok] == s_true[ok]).mean(), sw[ok].mean(), np.median(d[bad]) if bad.any() else 0, np.percentile(d[bad], 90) if bad.any() else 0, np.median(d[ok])))
tot = mixed_n = badc = hit = 0
for li, u in enumerate(G.us):
    for ci, c in enumerate(atlas[u]):
        nt = len(c.G); b = G.base[u][ci]; ids = b + np.arange(nt * M).reshape(nt, M); v4 = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]], -1).reshape(-1, 4); v4 = v4[fin[v4].all(1)]
        if not len(v4): continue
        l4 = sw[v4]; mixed = (l4 != l4[:, :1]).any(1); cen = G.P[v4].mean(1); Tc = R.T_ref(cen, rho); under = (T[v4].mean(1) - Tc) < -.03; o_ = np.linalg.norm(cen, axis=1) > rho
        tot += o_.sum(); mixed_n += (mixed & o_).sum(); badc += (under & o_).sum(); hit += (under & mixed & o_).sum()
print('RES ячеек %d; со смешанными метками %d (%.1f%%); «плохих» %d; пойманы %d (%.0f%%)' % (tot, mixed_n, 100 * mixed_n / tot, badc, hit, 100 * hit / max(badc, 1)))
