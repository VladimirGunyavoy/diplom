"""R20×100 статик кар: сводка по сидам. python3 agg_car.py data/car_static  (env_A_static_r20s<k>.json, s<k>_p1.log, s<k>_p2.log, starts100.npy)"""
import os, sys, re, json, glob, numpy as np
D = sys.argv[1]; rows = []; allr = []; ntot = 0; ok_tot = 0; tch = 0
for k in range(100):
    f = os.path.join(D, 'env_A_static_r20s%d.json' % k)
    if not os.path.exists(f): continue
    d = json.load(open(f)); res = d['res']; Tref = np.array(d['Tref'], float); ok = [r for r in res if r['ok']]
    rt = np.array([r['T'] / Tref[r['i']] for r in ok if np.isfinite(Tref[r['i']])]); allr += rt.tolist(); ntot += len(res); ok_tot += len(ok); t = sum(r['touch'] for r in res); tch += t
    def sec(fn, pat):
        p = os.path.join(D, fn)
        if not os.path.exists(p): return None
        m = re.findall(pat, open(p, errors='ignore').read().replace('\r', '\n')); return float(m[-1]) if m else None
    nod = sec('s%d_p1.log' % k, r'узлов (\d+) '); tb = sec('s%d_p1.log' % k, r'узлов \d+ (\d+) с'); tp = sec('s%d_p2.log' % k, r'подготовка \(индекс/стенсилы\) ([\d.]+) с'); tq = sec('s%d_p2.log' % k, r'проходы ([\d.]+) с'); ts = round(tp + tq, 1) if tp is not None and tq is not None else None
    rows.append(dict(seed=k, reach=len(ok), n=len(res), touch=t, med=float(np.median(rt)), p90=float(np.percentile(rt, 90)), mx=float(rt.max()), nodes=nod, build=tb, solve=ts, ms=float(np.mean([r['ms'] for r in res]))))
print('| сид | дошли | касаний | мед | p90 | max | узлов | build, с | solve, с | мс/шаг |\n|---|---|---|---|---|---|---|---|---|---|')
for r in rows: print('| %d | %d/%d | %d | %.3f | %.3f | %.3f | %s | %s | %s | %.0f |' % (r['seed'], r['reach'], r['n'], r['touch'], r['med'], r['p90'], r['mx'], r['nodes'], r['build'], r['solve'], r['ms']))
a = np.array(allr); print('\n**Итого %d сидов:** дошли %d/%d, касаний %d, T/T_ref (по всем %d дошедшим) мед %.3f / p90 %.3f / max %.3f; по сидам мед %.3f–%.3f (мед %.3f), p90 %.3f–%.3f' % (len(rows), ok_tot, ntot, tch, len(a), np.median(a), np.percentile(a, 90), a.max(), min(r['med'] for r in rows), max(r['med'] for r in rows), np.median([r['med'] for r in rows]), min(r['p90'] for r in rows), max(r['p90'] for r in rows)))
