"""Согласие карты грубого поля u* (ближайший узел field.npz, как в проходе 2) с argmin базы на пробах базы: доля совпадений (все пробы / пробы с зазором базы), доля в «полосе» (gap < TAU или ближайший совпал).
Запуск (системный python, scipy): FIELD=… PROBES=… [TAU=.3] python3 agree.py"""
import os, json, numpy as np
from scipy.spatial import cKDTree
f = np.load(os.environ['FIELD']); p = np.load(os.environ['PROBES']); t = cKDTree(f['P']); d, i = t.query(p['P']); arg, gap = f['arg'][i], f['gap'][i]; TAU = float(os.environ.get('TAU', .3)); m = arg == p['arg']; band = gap < TAU
r = dict(n=len(m), agree=float(m.mean()), agree_or_band=float((m | band).mean()), band_share=float(band.mean()), median_dist=float(np.median(d)), p90_dist=float(np.percentile(d, 90))); print(json.dumps(r)); json.dump(r, open(os.environ.get('OUT', '.') + '/agree_tau%g.json' % TAU, 'w'))
