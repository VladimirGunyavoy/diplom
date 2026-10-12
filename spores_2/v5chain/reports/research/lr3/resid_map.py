"""lr3 (laptop-v5chain-research-3, 2026-10-12): карта остатков поперёк потока на готовом атласе v8 (ДИ), алгоритм не меняется.
Для каждого промежутка между соседними линиями эволюции клетки ставится пробная средняя линия (точная траектория ДИ от середины
первой строки, tau — среднее соседних). На ней V считается правилами solve8 по уже посчитанной V атласа (вход в цель точно,
переход E3 в чужие ячейки, цепочка E1 назад по строкам). Остаток = |V_средней − среднее V двух соседних линий| (то, что даёт
интерполяция). Рядом — ошибки против точного эталона T_ref.  Запуск (из v8): python resid_map.py [--layers 1,-1] [--out resid]"""
import os, sys, json, argparse, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--v8', default='.'); ap.add_argument('--layers', default='1,-1'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='resid'); a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.v8)); sys.path.insert(0, os.path.join(os.path.abspath(a.v8), 'src', 'algo'))
import solve8 as S
from src.algo import ref_di_disk as R
g = S.load_growN(); layers = [float(x) for x in a.layers.split(',')]
atlas, tg = S.build_atlas(layers, a.seed, par=False); G = S.make_graph(g, atlas, 'N'); t0 = time.perf_counter(); G.solve(verbose=True); M = G.M; rho = G.rho; BIG = S.BIG
fin = lambda v: np.where(v < BIG / 2, v, np.nan)
rows = []          # по одной записи на промежуток (клетка, j)
pts = []           # все пробные узлы: x, v, остаток, ошибка интерп., ошибка средней, слой
node_err = []
t_probe = 0.
for li, u in enumerate(G.us):
    for ci, c in enumerate(atlas[u]):
        Gc, tau = np.asarray(c.G), np.asarray(c.tau); nt = len(Gc); b = G.base[u][ci]; V = fin(G.V[b:b + nt * M].reshape(nt, M)); gc = G.gcell[u][ci]
        Tn = R.T_ref(Gc.reshape(-1, 2), rho).reshape(nt, M); out_ = np.linalg.norm(Gc, axis=2) > rho; node_err.append((V - Tn)[out_ & np.isfinite(V)])
        ent = np.full((nt, M), np.inf)                                                                   # метка «линия входит в цель на шаге k»
        for j in range(M): ent[:-1, j] = S.entry_time(Gc[:-1, j], u, np.maximum(tau[1:, j] - tau[:-1, j], 1e-12), rho)
        lab = np.isfinite(ent).any(0) | (np.linalg.norm(Gc, axis=2) <= rho).any(0)
        for j in range(M - 1):
            tp = time.perf_counter()
            tm = (tau[:, j] + tau[:, j + 1]) / 2; dt = tm - tm[0]; x0 = (Gc[0, j] + Gc[0, j + 1]) / 2
            P = np.stack([x0[0] + x0[1] * dt + u * dt * dt / 2, x0[1] + u * dt], 1)                      # точная траектория ДИ
            cost = np.diff(tm); e3 = np.full(nt, np.inf)
            for u2 in G.us: e3 = np.minimum(e3, G.evalpairs(G.V, nt, G.locate(u2, P, excl=np.full(nt, gc))))
            e3 = np.where(e3 < BIG / 2, e3, np.inf); en = np.full(nt, np.inf); en[:-1] = S.entry_time(P[:-1], u, np.maximum(cost, 1e-12), rho); inn = np.linalg.norm(P, axis=1) <= rho
            Vm = np.minimum(e3, en); Vm[inn] = 0.
            for k in range(nt - 2, -1, -1): Vm[k] = min(Vm[k], cost[k] + Vm[k + 1])
            t_probe += time.perf_counter() - tp
            Vi = (V[:, j] + V[:, j + 1]) / 2; Tm = R.T_ref(P, rho); ok = np.isfinite(Vm) & np.isfinite(Vi) & ~inn
            res = np.abs(Vm - Vi); ei = Vi - Tm; em = Vm - Tm; w = float(np.linalg.norm(Gc[:, j + 1] - Gc[:, j], axis=1).max())
            labm = bool(np.isfinite(en).any() or inn.any())
            rows.append(dict(layer=u, cell=ci, gap=j, rows=nt, width=w, n_ok=int(ok.sum()), res_max=float(res[ok].max()) if ok.any() else None, res_med=float(np.median(res[ok])) if ok.any() else None,
                             ei_max=float(np.abs(ei[ok]).max()) if ok.any() else None, em_max=float(np.abs(em[ok]).max()) if ok.any() else None, lab=[bool(lab[j]), labm, bool(lab[j + 1])],
                             hole_fixed=int((np.isfinite(Vm) & ~np.isfinite(Vi)).sum()), hole_left=int((~np.isfinite(Vm)).sum())))
            for k in np.flatnonzero(ok): pts.append((P[k, 0], P[k, 1], res[k], ei[k], em[k], u, ci, j))
pts = np.array(pts); ne = np.concatenate(node_err); q = lambda x, p: float(np.percentile(x, p))
rm = np.array([r['res_max'] if r['res_max'] is not None else np.nan for r in rows]); val = np.isfinite(rm); mism = np.array([len(set(r['lab'])) > 1 for r in rows])
summ = dict(layers=layers, M=int(M), cells={str(u): len(atlas[u]) for u in G.us}, nodes=int(G.n), gaps=len(rows), gaps_valid=int(val.sum()), probe_nodes=int(len(pts)), t_solve=G.t_solve, t_probe=t_probe,
            node_err_vs_ref=dict(med_abs=q(np.abs(ne), 50), p90_abs=q(np.abs(ne), 90), max_abs=float(np.abs(ne).max()), under=float((ne < -1e-3).mean()), over=float((ne > 1e-3).mean())),
            resid=dict(med=q(pts[:, 2], 50), p90=q(pts[:, 2], 90), p99=q(pts[:, 2], 99), max=float(pts[:, 2].max())),
            interp_err_vs_ref=dict(med_abs=q(np.abs(pts[:, 3]), 50), p90_abs=q(np.abs(pts[:, 3]), 90), max_abs=float(np.abs(pts[:, 3]).max()), under=float((pts[:, 3] < -1e-3).mean())),
            probe_err_vs_ref=dict(med_abs=q(np.abs(pts[:, 4]), 50), p90_abs=q(np.abs(pts[:, 4]), 90), max_abs=float(np.abs(pts[:, 4]).max()), under=float((pts[:, 4] < -1e-3).mean())),
            gaps_over_tol={str(t): dict(gaps=int((rm[val] > t).sum()), frac=float((rm[val] > t).mean()), probe_nodes_frac=float((pts[:, 2] > t).mean())) for t in (.003, .01, .03, .1, .3)},
            label_mismatch=dict(gaps=int(mism.sum()), res_max_med=float(np.nanmedian(rm[mism])) if mism.any() else None, res_max_med_other=float(np.nanmedian(rm[~mism]))),
            holes=dict(fixed=int(sum(r['hole_fixed'] for r in rows)), left=int(sum(r['hole_left'] for r in rows))))
json.dump(dict(summary=summ, gaps=rows), open(a.out + '.json', 'w'), ensure_ascii=False, indent=1); np.save(a.out + '_pts.npy', pts); print(json.dumps(summ, ensure_ascii=False, indent=1))
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a_, col, ttl, kw in ((ax[0], pts[:, 2], 'остаток |V средней линии − интерполяция|', dict(cmap='viridis', vmin=0, vmax=q(pts[:, 2], 98))), (ax[1], pts[:, 3], 'ошибка интерполяции: V_интерп − T_ref', dict(cmap='coolwarm', vmin=-.3, vmax=.3)), (ax[2], pts[:, 4], 'ошибка пробной линии: V_средней − T_ref', dict(cmap='coolwarm', vmin=-.3, vmax=.3))):
        sc = a_.scatter(pts[:, 0], pts[:, 1], c=col, s=7, **kw); plt.colorbar(sc, ax=a_, shrink=.8); a_.set_title(ttl); a_.set_xlabel('x'); a_.set_ylabel('v'); a_.add_patch(plt.Circle((0, 0), rho, fill=False, color='k')); a_.set_aspect('equal')
        xs = np.linspace(-4.5, 4.5, 400); a_.plot(-xs * np.abs(xs) / 2, xs, 'k--', lw=.6)
    fig.suptitle('v8 ДИ, слои %s, M=%d: пробные средние линии (%d узлов)' % (a.layers, M, len(pts))); fig.tight_layout(); fig.savefig(a.out + '.png', dpi=90)
except Exception as e: print('no plot:', e)
