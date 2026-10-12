"""lr4: убирает ли переключение в подточках строки (Q lin + K, qsolve3) горб настолько, что дешёвый агент look перестаёт дребезжать/застревать?
Q lin K=0/3/7/15: ошибка узлов против T_ref и агенты look / commit (1000 стартов). env M, LAYERS, SEED, N, OUT. Запуск из v8."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, qsolve3 as Q3, stuck0 as Z
from src.algo import ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, int(os.environ.get('SEED', '0')), par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
X0 = S.starts(g, int(os.environ.get('N', '1000')), 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM; inw = mx <= float(g.XLV[0]); tmax = 3 * Tr + 2; ban = Z.make_ban('none', None, G.L)
T = R.T_ref(G.P, rho); out = np.linalg.norm(G.P, axis=1) > rho; res = {}
print('RES слои %s M=%d узлов %d стартов %d, в поле %d' % (layers, G.M, G.n, len(X0), inw.sum()), flush=True)
for k in (0, 3, 7, 15):
    t0 = time.perf_counter(); V, Q, it, ts = QS.solve_q(G, 0.) if k == 0 else Q3.solve_q3(G, Q3.prep_sub(G, k)); tsolve = time.perf_counter() - t0; G.V = Q; fin = V < S.BIG / 2; e = (V - T)[out & fin]
    print('RES Q lin K=%-2d solve+подготовка %.2f с, проходов %d | узлы: |ош| мед %.3f p90 %.3f, <−.05: %.3f, >+.05: %.3f, p1 %.3f, max %.2f' % (k, tsolve, it, np.median(np.abs(e)), np.percentile(np.abs(e), 90), (e < -.05).mean(), (e > .05).mean(), np.percentile(e, 1), e.max()), flush=True)
    for pn, fn in (('look', Z.run_look), ('commit', Z.run_commit)):
        t1 = time.perf_counter(); Tt, sw, Xe, ce = fn(G, X0, tmax, ban); dtm = time.perf_counter() - t1; ok = np.isfinite(Tt); rw = Tt[ok & inw] / Tr[ok & inw]; stuck = ~ok & (np.abs(Xe[:, 1]) < .05) & (np.linalg.norm(Xe, axis=1) > rho) & inw
        res['K%d|%s' % (k, pn)] = dict(reached_field=float(ok[inw].mean()), T_med=float(np.median(rw)), T_p90=float(np.percentile(rw, 90)), T_p99=float(np.percentile(rw, 99)), T_max=float(rw.max()), sw_med=float(np.median(sw[inw])), sw_mean=float(sw[inw].mean()), stuck=int(stuck.sum()), sec=dtm)
        print('RES Q lin K=%-2d %-6s в поле дошли %.3f (застряло на v≈0: %d) | T/T_ref мед %.3f p90 %.3f p99 %.3f max %.2f | перекл мед %.0f ср %.1f | %.0f с' % (k, pn, ok[inw].mean(), stuck.sum(), np.median(rw), np.percentile(rw, 90), np.percentile(rw, 99), rw.max(), np.median(sw[inw]), sw[inw].mean(), dtm), flush=True)
json.dump(res, open(os.environ.get('OUT', 'agents_k') + '.json', 'w'), ensure_ascii=False, indent=1)
