"""CA4-6D: table_m3gdg.md + total_m3gdg.png из data/m3gdg_*.json (m3d_goals.py). python m3gdg_report.py"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports', 'bench'); DD = os.path.join(D, 'data'); L = lambda n: json.load(open(os.path.join(DD, n)))
ref, fast, sst, sp = L('m3gdg_ref.json'), L('m3gdg_ocpfast.json'), L('m3gdg_sst.json'), L('m3gdg_spores.json'); K = sp['K']
g = lambda r, k='T': r[k] if r.get(k) else np.nan
Tref = np.array([g(r) for r in ref['res']]); ok = np.isfinite(Tref)
def row(name, T, tb, tq_med, tq_p90, note, per_goal_build=False):
    T = np.array(T, float); r = (T / Tref)[ok & np.isfinite(T)]; reach = int(np.isfinite(T[ok]).sum())
    return dict(name=name, reach='%d/%d' % (reach, ok.sum()), q='%.2f / %.2f / %.2f / %.2f' % tuple(np.nanpercentile(r, [50]).tolist() + [r.mean(), np.percentile(r, 90), r.max()]) if len(r) else '—', tb=tb, tq=tq_med, tq90=tq_p90, note=note, pg=per_goal_build)
R = []
tq = [x['t'] for x in ref['res']]; R.append(row('OCP CasADi/IPOPT + зазор в узлах: multi (8 ветвей×4 затравки) = эталон', Tref, ref['t_build'], np.median(tq), np.percentile(tq, 90), 'T_ref по построению (1.00); заготовка = сборка символьной задачи, один раз'))
tq = [x['t'] for x in fast['res']]; R.append(row('OCP CasADi/IPOPT + зазор: fast (ближайшая ветвь×2 затравки)', [g(r) for r in fast['res']], fast['t_build'], np.median(tq), np.percentile(tq, 90), 'цель — параметр NLP, перестройки нет'))
tq = [x['t_query'] for x in sp['res']]; tbs = [x['t_build'] for x in sp['res']]
R.append(row('**споры** коридор v6 SPD6 (NB24000 NF6000, 1 ядро)', [g(r) for r in sp['res']], float(np.median(tbs)), np.median(tq), np.percentile(tq, 90), 'заготовка (build_back) — НА КАЖДУЮ цель; refine в IPOPT с зазором до дисков (CACLR=1, w29), пакетный поток стыковки BATCHFAST=1, tree_fast с векторным blocked; было 40 с заготовка + 28 с запрос (SLSQP) — m3gdg_spores_slsqp_old.json', True))
import os as _os
if _os.path.exists(_os.path.join(DD, 'm3gdg_spores_v3.json')):
    v3 = L('m3gdg_spores_v3.json'); tq3 = [x['t_query'] for x in v3['res']]; tb3 = [x['t_build'] for x in v3['res']]
    R.append(row('**споры** v3 (NB24000 NF6000, + полудуга из того же вызова потока), прогон w29 11:3x', [g(r) for r in v3['res']], float(np.median(tb3)), np.median(tq3), np.percentile(tq3, 90), 'отдельный прогон K=20 после доп. ускорения tree_fast (полудуга blocked из того же вызова); заготовка НА КАЖДУЮ цель; T те же, что в строке выше', per_goal_build=True))
    for tg, nm in (('b', 'NB24000 NF2000'), ('a', 'NB12000 NF3000'), ('i', 'NB24000 NF2000, NEIGH=0 TRIES=1'), ('j', 'NB12000 NF3000, NEIGH=0 TRIES=1'), ('k', 'NB12000 NF2000, NEIGH=0 TRIES=1'), ('o', '**ИТОГ w29:** NB8000 NF3000, NEIGH=0 TRIES=1'), ('p', 'NB6000 NF3000, NEIGH=0 TRIES=1')):
        fn = 'm3gdg_spores_%s.json' % tg
        if _os.path.exists(_os.path.join(DD, fn)):
            v = L(fn); tq_ = [x['t_query'] for x in v['res']]; tb_ = [x['t_build'] for x in v['res']]
            R.append(row('**споры** v3, урезанные деревья (%s)' % nm, [g(r) for r in v['res']], float(np.median(tb_)), np.median(tq_), np.percentile(tq_, 90), 'свип размера деревьев и числа refine (w29, K=20)', per_goal_build=True))
for bi, B in enumerate((.3, 1, 3, 10)):
    R.append(row('SST (свой, prune .1) бюджет %g с' % B, [g(r['runs'][bi], 'T') for r in sst['res']], 0., B, B, 'бюджет на каждый переход'))
H = '| метод | заготовка, с | запрос мед / p90, с | на переход (заготовка+запрос), с | total(K=%d), с | дошли | T/T_ref мед / ср / p90 / max | примечание |\n|---|---|---|---|---|---|---|---|\n' % K
T = H; tot = {}
for r in R:
    per = r['tb'] + r['tq'] if r['pg'] else r['tq']; total = K * per + (0 if r['pg'] else r['tb']); tot[r['name']] = (0 if r['pg'] else r['tb'], per)
    T += '| %s | %.2f%s | %.2f / %.2f | %.2f | %.0f | %s | %s | %s |\n' % (r['name'], r['tb'], ' (на цель)' if r['pg'] else '', r['tq'], r['tq90'], per, total, r['reach'], r['q'], r['note'])
f_, s_ = tot[R[1]['name']], tot[R[2]['name']]
kstar = 'нет: споры на цель (%.2f с) не дешевле OCP fast (%.2f с) при новой цели на каждый переход' % (s_[1], f_[1]) if s_[1] >= f_[1] else 'K* = %.0f' % ((f_[0]) / (f_[1] - s_[1]))
sA = 4 * R[2]['tb'] + 40 * R[2]['tq']; fA = R[1]['tb'] + 40 * R[1]['tq']
open(os.path.join(D, 'table_m3gdg.md'), 'w').write('# CA4-6D+2 диска: последовательные цели, 3 звена 6D + g .3 + 2 диска (all_systems 8d)\nK = %d переходов между случайными целями (seed 11), старт = прошлая цель в покое, T_ref = OCP multi (где его нет — отмечено). Старт и цели вне дисков (зазор > .1); проверка T: единый проверяльщик с контролем столкновения. Споры — 1 ядро (taskset), OCP/SST — 1 процесс на переход (aida). PPO с целью на входе не делали (обучение > 1 ч).\n\n' % K + T +
    '\n**K\\*** (споры vs OCP fast): %s. Против OCP multi (эталон) споры дешевле при любом K (%.1f с/переход против %.1f).\n\nВариант A «4 станции, K=40» (расчёт по измеренным временам, не прогон): споры 4·%.2f + 40·%.2f = %.0f с против OCP fast %.2f + 40·%.2f = %.0f с.\n' % (kstar, s_[1], R[0]['tq'], R[2]['tb'], R[2]['tq'], sA, R[1]['tb'], R[1]['tq'], fA))
ks = np.arange(0, K + 1); plt.figure(figsize=(6, 4))
for r in R[:3]: tb, per = tot[r['name']]; plt.plot(ks, tb + per * ks, label=r['name'].replace('**', '')[:42])
plt.xlabel('число переходов K'); plt.ylabel('total, с'); plt.legend(fontsize=7); plt.grid(alpha=.3); plt.title('CA4 3зв.+g+диски: total(K)'); plt.tight_layout(); plt.savefig(os.path.join(D, 'total_m3gdg.png'), dpi=130)
# --- проверочный набор (w29): другой сид целей (GSEED=23), K=40, конфиг споров заморожен — НЕ подбирался на этом наборе
if os.path.exists(os.path.join(DD, 'm3gdg_spores_v40.json')):
    rf, fs, sv, ss = L('m3gdg_ref_v40.json'), L('m3gdg_ocpfast_v40.json'), L('m3gdg_spores_v40.json'), L('m3gdg_sst_v40.json')
    T0 = np.array([g(r) for r in rf['res']]); ok0 = np.isfinite(T0) & (T0 > .05)
    def st(T):
        T = np.array(T, float); m = ok0 & np.isfinite(T); r = T[m] / T0[m]
        return '%d/%d' % (m.sum(), ok0.sum()), ('%.2f / %.2f / %.2f / %.2f' % (np.median(r), r.mean(), np.percentile(r, 90), r.max())) if len(r) else '—'
    V = ['\n## Проверочный набор (GSEED=23, K=%d, конфиг споров заморожен: NB8000 NF3000 NEIGH=0 TRIES=1 CACLR=1 BATCHFAST=1; НЕ подбирался на этом наборе)\n' % sv['K'],
         '| метод | заготовка (на цель), с | запрос мед / p90, с | на переход, с | дошли | T/T_ref мед / ср / p90 / max | примечание |\n|---|---|---|---|---|---|---|\n']
    tq = [x['t'] for x in rf['res']]; V.append('| OCP multi (эталон) | %.1f | %.1f / %.1f | %.1f | %s | 1.00 по построению | T_ref; переход %d из %d без эталона/нулевой исключён |\n' % (rf['t_build'], np.median(tq), np.percentile(tq, 90), np.median(tq), *(('%d/%d' % (ok0.sum(), len(ok0)),)), (~ok0).sum(), len(ok0)))
    tq = [x['t'] for x in fs['res']]; a, b = st([g(r) for r in fs['res']]); V.append('| OCP fast | %.1f | %.2f / %.2f | %.2f | %s | %s | |\n' % (fs['t_build'], np.median(tq), np.percentile(tq, 90), np.median(tq), a, b))
    tq = [x['t_query'] for x in sv['res']]; tb = [x['t_build'] for x in sv['res']]; a, b = st([g(r) for r in sv['res']]); V.append('| **споры** (заморож.) | %.2f | %.2f / %.2f | %.2f | %s | %s | no_path: %s |\n' % (np.median(tb), np.median(tq), np.percentile(tq, 90), np.median(tb) + np.mean(tq), a, b, [r['k'] for r in sv['res'] if not r.get('T')]))
    for bi, B in enumerate((.3, 1, 3, 10)):
        a, b = st([g(r['runs'][bi], 'T') for r in ss['res']]); V.append('| SST бюджет %g с | 0 | %g | %g | %s | %s | |\n' % (B, B, B, a, b))
    V.append('\nЧестно: если на проверочном наборе споры хуже, чем на подбиравшемся, — это и есть оценка без подгонки; конфиг не подстраивался.\n')
    open(os.path.join(D, 'table_m3gdg.md'), 'a').write(''.join(V))
print(open(os.path.join(D, 'table_m3gdg.md')).read())
