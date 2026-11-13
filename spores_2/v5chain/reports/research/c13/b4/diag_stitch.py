"""блок 4 (а): ДИ8D NB100k NF3000 — для каждого старта: мин. расстояние (масштабированная метрика стыковки Z/scale) от узлов прямого дерева до обратного, число попаданий и лучший промах кандидатов; гистограмма для дошедших/недошедших.
Что сейчас «попадание»: после стыка (прямой узел i, обратный b из kn=8 ближайших) последовательность слоёв b проигрывается от СОСТОЯНИЯ i (а не от b) и конец должен лечь в окно цели |x|<.35 — допуск стыковки задаёт не радиус, а усиление рассогласования потоком до цели."""
import sys, os, runpy, json, numpy as np
sys.path.insert(0, os.path.expanduser('~/spore_v5/c13/spd6')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ref = json.load(open(os.path.expanduser('~/spore_v5/c13/out_spd6/di8_spd6_nb100k.json')))['per_start']; reached = {r['i']: (r.get('why') == 'ok') for r in ref if not r.get('skip')}
sys.argv = ['spores_di.py', 'di8', '100000', '3000', '3', 'diag', os.path.expanduser('~/spore_v5/c13/out/di8_ref_100_rho35_s9.npy')]
G = runpy.run_path(os.path.expanduser('~/spore_v5/c13/spd6/spores_di.py'), run_name='lib'); S, back, e, tau, rho, miss = [G[k] for k in ('S', 'back', 'e', 'tau', 'rho', 'miss')]
import stitch; rows = []
for i in sorted(reached):
    C, nh, (Pf, Gf, parf, layf) = stitch.cand(S, e.X0[i], back, tau, 3000, rho, miss, 8); d, jb = stitch.fdist(S, back, Pf)
    mm = [c['miss'] for c in C if c['hit'] == 1]; rows.append(dict(i=i, reached=reached[i], dmin=float(d.min()), d10=float(np.percentile(d, 10)), dmed=float(np.median(d)), nh=nh, best_miss=float(min(mm)) if mm else None, nF=len(Pf)))
json.dump(rows, open(os.path.expanduser('~/spore_v5/c13/out_b4/diag_stitch_di8.json'), 'w'), indent=1)
sc = np.asarray(S.scale); print('масштаб осей', sc.tolist(), '; rho (ячейка слияния, в долях масштаба) %.2f; окно цели в масштабированных единицах: поз .35/%.1f=%.3f, ск .35/%.1f=%.3f' % (rho, sc[0], .35 / sc[0], sc[-1], .35 / sc[-1]))
bins = [0, .02, .05, .1, .15, .2, .3, .5, 9]
for lab, flag in (('дошли', True), ('недошли', False)):
    r = [x for x in rows if x['reached'] == flag]
    if not r: continue
    for key in ('dmin', 'dmed'):
        h, _ = np.histogram([x[key] for x in r], bins); print('%s (n=%d) гистограмма %s по бинам %s: %s; медиана %.3f' % (lab, len(r), key, bins[:-1], h.tolist(), np.median([x[key] for x in r])))
    print('%s: попаданий (nh) мед %.0f, доля стартов с nh>0 %.2f; лучший промах (в метрике miss, ед. окна .35) мед %.3f' % (lab, np.median([x['nh'] for x in r]), np.mean([x['nh'] > 0 for x in r]), np.median([x['best_miss'] for x in r if x['best_miss'] is not None])))
