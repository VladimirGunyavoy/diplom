"""lr4: как брать значение чужого слоя, когда точку накрывают несколько ячеек: первая по центральности (как в solve8) / min по всем кандидатам (как в v7) / среднее.
Q lin, K подточек нет. Замеры: ошибка узлов, агенты look / commit. env M, ETOL, LAYERS, SEED, OUT. Запуск из v8."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, stuck0 as Z
from src.algo import ref_di_disk as R
BIG = S.BIG


def patch(G, how):
    def evalpairs(V, n, pairs):
        pi, verts, w, rank = pairs; acc = np.full(n, np.inf) if how == 'min' else np.zeros(n); cnt = np.zeros(n)
        for r in range(G.K):
            m = rank == r
            if not m.any(): continue
            p_, v_, w_ = pi[m], verts[m], w[m]; q = V[v_]; ok = (q < BIG / 2).all(1); val = (w_ * np.where(q < BIG / 2, q, 0.)).sum(1); p_, val = p_[ok], val[ok]
            if how == 'min': np.minimum.at(acc, p_, val)
            else: np.add.at(acc, p_, val); np.add.at(cnt, p_, 1)
        return acc if how == 'min' else np.where(cnt > 0, acc / np.maximum(cnt, 1), np.inf)
    G.evalpairs = evalpairs


g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, int(os.environ.get('SEED', '0')), par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
X0 = S.starts(g, 1000, 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); inw = mx <= float(g.XLV[0]); tmax = 3 * Tr + 2; ban = Z.make_ban('none', None, G.L); T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; res = {}
print('RES слои %s M=%d узлов %d кандидатов K=%d стартов в поле %d' % (layers, G.M, G.n, G.K, inw.sum()), flush=True)
for how in ('first', 'min', 'mean'):
    if how != 'first': patch(G, how)
    V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; fin = V < BIG / 2; e = (V - T)[out & fin]
    print('RES донор %-5s solve %.2f с, проходов %d | узлы: |ош| мед %.3f p90 %.3f, <−.05: %.3f, >+.05: %.3f, p1 %.3f' % (how, ts, it, np.median(np.abs(e)), np.percentile(np.abs(e), 90), (e < -.05).mean(), (e > .05).mean(), np.percentile(e, 1)), flush=True)
    for pn, fn in (('look', Z.run_look), ('commit', Z.run_commit)):
        t1 = time.perf_counter(); Tt, sw, Xe, ce = fn(G, X0, tmax, ban); dtm = time.perf_counter() - t1; ok = np.isfinite(Tt); rw = Tt[ok & inw] / Tr[ok & inw]; stuck = ~ok & (np.abs(Xe[:, 1]) < .05) & (np.linalg.norm(Xe, axis=1) > rho) & inw
        res['%s|%s' % (how, pn)] = dict(reached_field=float(ok[inw].mean()), T_med=float(np.median(rw)), T_p90=float(np.percentile(rw, 90)), T_p99=float(np.percentile(rw, 99)), T_max=float(rw.max()), sw_mean=float(sw[inw].mean()), stuck=int(stuck.sum()), sec=dtm)
        print('RES донор %-5s %-6s в поле дошли %.3f (застряло на v≈0: %d) | T/T_ref мед %.3f p90 %.3f p99 %.3f max %.2f | перекл мед %.0f ср %.1f | %.0f с' % (how, pn, ok[inw].mean(), stuck.sum(), np.median(rw), np.percentile(rw, 90), np.percentile(rw, 99), rw.max(), np.median(sw[inw]), sw[inw].mean(), dtm), flush=True)
json.dump(res, open(os.environ.get('OUT', 'donor_k') + '.json', 'w'), ensure_ascii=False, indent=1)
