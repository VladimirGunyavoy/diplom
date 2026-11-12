"""Статистика ДИ4D на свежих стартах r37: python3 r37_stats.py dir_with_json starts.npy → строки по 253 keep и 300."""
import json, sys, glob, numpy as np
Q = np.load(sys.argv[2])[:, :4]; x, v = Q[:, :2], Q[:, 2:]; keep = (np.abs(x + v * np.abs(v) / 2).max(1) <= 2.5)
print('keep', keep.sum(), 'из', len(Q))
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    d = json.load(open(f)); T = np.array([np.nan if t is None else t for t in d['T']]); ref = np.array(d['ref']); r = T / ref
    out = [f.split('/')[-1][:-5], 'nodes %d' % d['nodes'], 'iters %d' % d['iters']]
    for nm, m in (('keep', keep), ('all', np.ones(len(Q), bool))):
        ok = m & ~np.isnan(r); q = r[ok]
        out.append('%s: дошли %d/%d mean %.3f мед %.3f p90 %.3f max %.3f' % (nm, ok.sum(), m.sum(), q.mean(), np.median(q), np.quantile(q, .9), q.max()))
    print(' | '.join(out)); print('   потеряны(keep):', [int(i) for i in np.flatnonzero(keep & np.isnan(r))], ' t_build %.0f t_solve %s' % (d['t_build'], d['t_solve'] and round(d['t_solve'])))
