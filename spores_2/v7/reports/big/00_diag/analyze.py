import json, sys, numpy as np, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
name = sys.argv[1] if len(sys.argv) > 1 else 'pend'
d = json.load(open('data/%s_cells.json' % name)); C = d['cells']
vol = np.array([c['vol'] for c in C]); nodes = np.array([c['nodes'] for c in C]); r = np.array([c['r'] for c in C]); L = np.array([c['L'] for c in C])
o = np.argsort(vol); cv = np.cumsum(vol[o]) / vol.sum(); cn = np.cumsum(nodes[o]) / nodes.sum(); x = np.arange(1, len(o) + 1) / len(o)
out = ['## %s: клеток %d, узлов %d' % (name, len(C), nodes.sum()), '',
 '| величина | min | p10 | медиана | p90 | max |', '|---|---|---|---|---|---|']
for nm, a in (('полуширина r', r), ('длина L, с', L), ('объём w·L', vol), ('узлов/клетку', nodes)):
    out.append('| %s | %.3g | %.3g | %.3g | %.3g | %.3g |' % ((nm,) + tuple(np.percentile(a, [0, 10, 50, 90, 100]))))
out += ['', '| q% самых мелких | доля объёма | доля узлов |', '|---|---|---|']
for q in (10, 25, 50, 75):
    k = int(len(o) * q / 100); out.append('| %d | %.1f%% | %.1f%% |' % (q, 100 * cv[k - 1], 100 * cn[k - 1]))
w = collections.Counter(); 
for c in C:
    for k, v in c['why'].items(): w[(k, v)] += 1
out += ['', 'Причины остановки роста (сторона F/B — вдоль потока, L/R — поперёк):', '', '| сторона | причина | клеток |', '|---|---|---|'] + ['| %s | %s | %d |' % (k, v, n) for (k, v), n in sorted(w.items())]
open('data/%s_diag.md' % name, 'w').write('\n'.join(out)); print('\n'.join(out))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(np.log10(vol), 30); ax[0].set_xlabel('lg объём клетки (w·L)'); ax[0].set_ylabel('клеток')
ax[1].hist(nodes, 30); ax[1].set_xlabel('узлов в клетке')
ax[2].plot(x, cv, label='объём'); ax[2].plot(x, cn, label='узлы'); ax[2].plot([0, 1], [0, 1], 'k:'); ax[2].set_xlabel('доля клеток (от мелких)'); ax[2].legend(); ax[2].set_title('кривая Лоренца')
plt.tight_layout(); plt.savefig('pics/%s_diag.png' % name, dpi=110)
