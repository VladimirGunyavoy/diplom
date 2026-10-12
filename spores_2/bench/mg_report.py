"""CA4: итог — таблица и total(K) по вариантам A (4 станции, 40 переходов) и B (20 новых целей). python mg_report.py → reports/bench/table_mg.md, total_mg.png.
total(K) = заготовка + Σ(на цель) + K·запрос. Споры: заготовка = слои 270 с + полный solve с SAVEE ~300 с (b9 spd1.md) + разовая загрузка индекса при первом запросе (измерена); на цель — retarget (A: 4 решения V, не 40; B: на каждую)."""
import os, json, numpy as np, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import common
D = common.DATA; OUT = os.path.join(os.path.dirname(D), ); BASE = 270. + 300.
def L(f): return json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else None
def stat(T, ref):
    r = np.array([t / x if t is not None else np.nan for t, x in zip(T, ref)]); ok = np.isfinite(r) & (r <= 3)
    return ok.mean(), (np.median(r[ok]) if ok.any() else np.nan), (np.percentile(r[ok], 90) if ok.any() else np.nan)
rows = []; fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for j, kind in enumerate('AB'):
    ref = L('mg_%s_ref.json' % kind); Tr = [r['Tref'] for r in ref]; n = len(Tr); t_ocp = float(np.median([r['t'] for r in ref]))
    sp = L('mg_%s_spores.json' % kind); sst = L('mg_%s_sst.json' % kind); ppo = sorted(glob.glob(os.path.join(D, 'mg_ppo_*_s*.json')))
    nd = len({tuple(np.round(x['goal'], 6)) for x in sp}); tg = float(np.mean([x['t_goal'] for x in sp if x['t_goal']])); tq = float(np.median([x['t_query'] for x in sp[1:]])); first = sp[0]['t_query']
    ok, med, p90 = stat([x['T_verify'] for x in sp], Tr); K = np.arange(0, n + 1)
    tot_sp = BASE + first + (nd * tg if kind == 'A' else 0) + (K * tg if kind == 'B' else 0) + K * tq
    rows.append((kind, 'споры (retarget b9, без досева цели)', ok, med, p90, BASE + first, tg, tq, tot_sp[-1]))
    ax[j].plot(K, tot_sp, label='споры'); ax[j].plot(K, K * t_ocp, label='OCP IPOPT на переход (%.0f с)' % t_ocp)
    rows.append((kind, 'OCP IPOPT на переход (эталон)', 1., 1., 1., 0., 0., t_ocp, n * t_ocp))
    if sst:
        for bi, B in enumerate([.3, 1., 3., 10., 60.]):
            ok2, m2, p2 = stat([r['runs'][bi].get('T') if r['runs'][bi]['has'] else None for r in sst], Tr); rows.append((kind, 'SST %g с/переход' % B, ok2, m2, p2, 0., 0., B, n * B))
        ax[j].plot(K, K * 60, label='SST 60 с/переход'); ax[j].plot(K, K * 10, '--', label='SST 10 с')
    for f in ppo + sorted(glob.glob(os.path.join(D, 'mg_sac_*_s*.json'))):
        d = json.load(open(f)); nm = 'SAC+HER' if 'mg_sac' in f else 'PPO'; r = d['res'][kind]; ok3, m3, p3 = stat([x['T_verify'] for x in r['per']], Tr); tq3 = float(np.median([x['t_query'] for x in r['per']]))
        rows.append((kind, '%s %.1fM шагов s%d (одна тренировка на все цели)' % (nm, d['steps'] / 1e6, d['seed']), ok3, m3, p3, d['t_train'], 0., tq3, d['t_train'] + n * tq3))
    ax[j].set_xlabel('K переходов'); ax[j].set_ylabel('суммарное время, с'); ax[j].set_title({'A': 'A: 4 станции', 'B': 'B: новые цели'}[kind]); ax[j].legend(fontsize=8); ax[j].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(D), 'total_mg.png'), dpi=110)
with open(os.path.join(os.path.dirname(D), 'table_mg.md'), 'w') as f:
    f.write('| вар | метод | дошли | T/Tref мед | p90 | заготовка, с | на цель, с | запрос, с | total(все K), с |\n|---|---|---|---|---|---|---|---|---|\n')
    for r in rows: f.write('| %s | %s | %.0f%% | %.3f | %.3f | %.0f | %.1f | %.2f | %.0f |\n' % (r[0], r[1], 100 * r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    for kind in 'AB':
        sp = [r for r in rows if r[0] == kind and r[1].startswith('споры')][0]; oc = [r for r in rows if r[0] == kind and r[1].startswith('OCP')][0]
        per = sp[6] * (1 if kind == 'B' else 0) + sp[7]; f.write('\nБезубыточность %s vs OCP: K* = %.0f переходов (заготовка %.0f с / (OCP %.1f − споры %.2f на переход)).\n' % (kind, (sp[5] / (oc[7] - per)) if kind == 'B' else (sp[5] + 4 * sp[6]) / (oc[7] - sp[7]), sp[5], oc[7], per if kind == 'B' else sp[7]))
print(open(os.path.join(os.path.dirname(D), 'table_mg.md')).read())
