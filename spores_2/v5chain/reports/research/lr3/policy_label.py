"""lr3: автопоиск излома/кривой переключения по МЕТКЕ ПОЛИТИКИ узла (без знания системы).
Метка узла после solve: откуда взято его V — 'свой поток' (E1/E2/стык с клеткой своего слоя) или 'переключение в слой u2' (E3 через слой).
Ячейка со смешанными метками вершин = ячейка, накрывающая излом. Проверка: (а) совпадение метки с истинным первым управлением (эталон);
(б) ловит ли признак «смешанные метки» те ячейки-доноры, где билинейная интерполяция точной T_ref занижает > tol."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1.], 0, par=False); G = S.make_graph(g, atlas, 'N'); BIG = S.BIG; rho = G.rho; M = G.M
for mode, pen in (('lin', 0.), ('lin', .02)):
    G.mode, G.pen = mode, pen; G.solve(verbose=False); V = G.V; fin = V < BIG / 2; T, info = R.T_ref(G.P, rho, return_details=True); out = np.linalg.norm(G.P, axis=1) > rho
    s_true = np.array([i[0] for i in info], float)                                                   # первое управление эталона
    own = np.full(G.n, BIG); k = G.e1; own[k] = G.cost[k] + V[G.nxt[k]]; own = np.minimum(own, G.V0)
    e3 = [np.where(np.isfinite(x), x + pen, BIG) for x in (G.evalpairs(V, G.n, p) for p in G.e3)]
    lab = G.layer.copy()                                                                             # метка = индекс слоя, чьим управлением идём
    for li2 in range(G.L):
        sw = fin & (G.layer != li2) & (e3[li2] <= V + 1e-9) & (own > V + 1e-9); lab[sw] = li2
    u_lab = np.array(G.us)[lab]; ok = fin & out; switched = lab != G.layer
    print('RES %s pen %.2f: конечных узлов %d; метка = истинному первому управлению: %.3f; узлов с меткой «переключение»: %.3f' % (mode, pen, ok.sum(), (u_lab[ok] == s_true[ok]).mean(), switched[ok].mean()))
    # расстояние ошибочных меток до истинной кривой переключения (в единицах T: |T_ref при u=+1 первом − при u=−1 первом| неизвестно → берём расстояние в плоскости до кривой x = −v|v|/2)
    d = np.abs(G.P[:, 0] + G.P[:, 1] * np.abs(G.P[:, 1]) / 2) / np.sqrt(1 + G.P[:, 1] ** 2); bad = ok & (u_lab != s_true)
    print('RES   ошибочные метки: расстояние до кривой переключения мед/p90 %.3f / %.3f (у всех узлов мед %.3f)' % (np.median(d[bad]), np.percentile(d[bad], 90), np.median(d[ok])))
    # ячейки: смешанные метки вершин против «плохих» ячеек (интерполяция точной T_ref в центре ячейки занижает)
    tot = mixed_n = badc = hit = 0; spread_m = []; spread_o = []
    for li, u in enumerate(G.us):
        for ci, c in enumerate(atlas[u]):
            nt = len(c.G); b = G.base[u][ci]; ids = b + np.arange(nt * M).reshape(nt, M); v4 = np.stack([ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]], -1).reshape(-1, 4)
            f4 = fin[v4].all(1); v4 = v4[f4]
            if not len(v4): continue
            l4 = lab[v4]; mixed = (l4 != l4[:, :1]).any(1); cen = G.P[v4].mean(1); Tc = R.T_ref(cen, rho); under = (T[v4].mean(1) - Tc) < -.03; o_ = np.linalg.norm(cen, axis=1) > rho
            tot += o_.sum(); mixed_n += (mixed & o_).sum(); badc += (under & o_).sum(); hit += (under & mixed & o_).sum(); sp = np.ptp(V[v4], axis=1); spread_m += sp[mixed & o_].tolist(); spread_o += sp[~mixed & o_].tolist()
    print('RES   ячеек %d; со смешанными метками %d (%.1f%%); «плохих» (интерп. точной T занижает > .03) %d; из них пойманы меткой %d (%.0f%%); разброс V по вершинам: смешанные мед %.2f, прочие мед %.2f' % (tot, mixed_n, 100 * mixed_n / tot, badc, hit, 100 * hit / max(badc, 1), np.median(spread_m), np.median(spread_o)))
