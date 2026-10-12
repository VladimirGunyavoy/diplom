"""lr3: откуда дыры V (узлы с V=∞, ~29%). Корень дыры — последний узел столбца клетки (дальше по потоку узлов нет): он либо не накрыт ни одной чужой клеткой
(«нет покрытия»: траектория ушла за край выращенного), либо накрыт, но у доноров тоже ∞ («каскад»). Расчёт — по Q (qsolve). env LAYERS."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS
g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1').split(',')]; atlas, _ = S.build_atlas(layers, 0, par=False); G = S.make_graph(g, atlas, 'N'); BIG = S.BIG; M = G.M
V, Q, it, ts = QS.solve_q(G, 0.); inf = V > BIG / 2; XL = float(g.XLV[0]); GMv = float(g.GM); print('RES слои %s: узлов %d, V=∞ у %d (%.3f); поле ±%.1f, рост до ±%.1f' % (layers, G.n, inf.sum(), inf.mean(), XL, XL + GMv))
last = G.nxt < 0; root = inf & last; cover = np.zeros(G.n, bool); cover_fin = np.zeros(G.n, bool)
for li2, u2 in enumerate(G.us):
    pi, verts, w, rank = G.e3[li2]; cover[pi] = True; ok = (Q[verts] < BIG / 2).all(1); cover_fin[pi[ok]] = True
P = G.P; amax = np.abs(P).max(1); inside = amax <= XL; ingrow = amax <= XL + GMv + 1e-9
print('RES корней дыр (последний узел столбца с ∞): %d из %d последних узлов; из них НЕ накрыты чужой клеткой: %d, накрыты, но доноры ∞ (каскад): %d' % (root.sum(), last.sum(), (root & ~cover).sum(), (root & cover).sum()))
r0 = root & ~cover; print('RES   некрытые корни: в поле ±%.1f: %d, в полосе роста: %d, за ней: %d; max(|x|,|v|) мед %.2f' % (XL, (r0 & inside).sum(), (r0 & ~inside & ingrow).sum(), (r0 & ~ingrow).sum(), np.median(amax[r0]) if r0.any() else 0))
# направление выхода: куда смотрит поток в некрытом корне (наружу от поля?)
u = np.array(G.us)[G.layer]; f = np.stack([P[:, 1], u], 1); outward = (np.sign(P[:, 0]) * f[:, 0] > 0) | (np.sign(P[:, 1]) * f[:, 1] > 0); print('RES   у некрытых корней поток направлен наружу (по x или v): %.2f' % outward[r0].mean() if r0.any() else 'RES   -')
print('RES узлы с ∞: в поле ±%.1f — %d (%.3f от узлов в поле), в полосе роста — %d (%.3f)' % (XL, (inf & inside).sum(), (inf & inside).sum() / max(inside.sum(), 1), (inf & ~inside).sum(), (inf & ~inside).sum() / max((~inside).sum(), 1)))
for li, uu in enumerate(G.us): m = G.layer == li; print('RES   слой %s: ∞ у %.3f узлов; в поле %.3f' % (uu, inf[m].mean(), (inf & inside & m).sum() / max((inside & m).sum(), 1)))
X0 = S.starts(g, 2000, 1); G.V = Q; Vs, _ = G.Vstar(X0); nov = ~np.isfinite(Vs); cov = np.zeros(len(X0), bool)
for u2 in G.us: pi, hid, sc = G.idx[u2].query(X0); cov[pi] = True
print('RES стартов 2000: V* нет у %.3f; из них накрыты хоть одной клеткой %.3f; знак x·v у стартов без V*: доля x·v>0 (уходят от цели) %.2f; |x| мед %.2f |v| мед %.2f' % (nov.mean(), cov[nov].mean(), (X0[nov, 0] * X0[nov, 1] > 0).mean(), np.median(np.abs(X0[nov, 0])), np.median(np.abs(X0[nov, 1]))))
np.save('holes_starts.npy', np.c_[X0, nov])
