"""линия A: таблица + total(N) для 6D/8D сред. python make_report_a.py m3d  → reports/bench/table_m3d.md, total_m3d.png (data/<env>_*.json). Свой файл: make_report.py зашит под имена c-линии."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common, envs_6d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = common.DATA; OUT = os.path.join(D, '..'); env = sys.argv[1]; TITLE = {'m3d': 'CA1-6D-a: 3D двузвенный манипулятор, 6D без g', 'm3gd': 'CA1-6D-b: 3 звена 6D + g .3, цель ВНИЗ', 'm8': 'CA1-8D: 4 звена 8D + g .3, цель вниз', 'm3gu': 'CA1-6D-b: 3 звена 6D + g .3, цель ВВЕРХ'}.get(env, env)
def L(p):
    try: return json.load(open(p))
    except Exception: return None
def f(x, n=3): return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else ('%.' + str(n) + 'g') % x
def fb(b): return '—' if not b else ('%.3g МБ' % (b / 1e6) if b >= 1e6 else '%.3g КБ' % (b / 1e3))
def st(s): return '%d%% | %s / %s / %s / %s' % (round(s['reached'] * 100), f(s['ratio_med']), f(s['ratio_mean']), f(s['ratio_p90']), f(s['ratio_max']))
e = common.make(env); R = []   # dict(m, par, build, qm, st, w, hw, note)
FIN = {'m3d': 'NB 12000, NF 1000, K 3, NEIGH 0', 'm3gd': 'NB 12000, NF 3000, K 3', 'm3gu': 'NB 12000, NF 3000, K 3', 'm8': 'NB 60000, NF 15000 (повтор NF 40000 для провалов), K 3, NEIGH 1, CAMS 20, откат SLSQP'}
for tag, nm in (('', 'v6'), ('_wlim', 'v6 + предел |w| по пути (wlim)'), ('_final' if env != 'm8' else '_tf3', 'v6 SPD6 (CasADi/IPOPT-refine + скомпилированный поток, |w|≤3 по пути)')):
    d = L(f'{D}/{env}_spores{"8d" if env == "m8" else "6d"}{tag}.json')
    if d: R.append(dict(m='**споры** коридор ' + nm, par=((FIN[env] + '; ' + ('1 ядро, taskset, OMP=1' if env != 'm8' else '10 процессов') + '; load average aida при прогоне ~6' if tag in ('_final', '_tf3') else ('NB 100000, NF 25000, K 3, эвристика back_heuristic; 1 ядро/запрос' if env == 'm8' else 'NB 12000, NF 3000, K 3; 1 ядро/запрос'))), build=d['t_build'], qm=d['t_query_med'], qp=d['t_query_p90'], s=d, w=d['weight']['bytes'], hw='CPU numpy+math',
                        note='SLSQP-доводка держит |w|≤3 только в узлах' if not tag else ('жёсткий предел |w|≤3 в 16 точках сегмента' if tag == '_wlim' else 'ФИНАЛ: IPOPT-refine (запас .004), gcc-поток; ' + ('wlim' if env != 'm3d' else 'без wlim'))))
if len(R) > 1 and R[-1]['m'].startswith('**споры**') and 'SPD6' in R[-1]['m']: R.insert(0, R.pop())   # финальная строка спор — первой (с ней считаются N*)
if env == 'm8':      # c7: HYB-8D (w30) — отношения к исправленному m8_ref.json (в m8_hyb8d.json лежат к заглушке); цифры из w30 (table_m8.md)
    d = L(f'{D}/m8_hyb8d.json')
    if d: R.append(dict(m='**споры+IPOPT (HYB-8D)** — путь спор `_tf3` → полный OCP fast (свободное T, N50 M4, тёплый старт; при отказе — путь спор)', par='NB 60000, NF 15000 (повтор NF 40000), K 3, wlim, CAMS 20, tree_fast; 10 процессов', build=d['t_build'] + 8., qm=d['t_query_med'], qp=d['t_query_p90'], s={k: d[k] for k in ('reached', 'ratio_med', 'ratio_mean', 'ratio_p90', 'ratio_max')}, w=d['weight']['bytes'], hw='CPU 1 ядро/запрос',
                        note='w30: полировка применена на 18/18 найденных путей; те же 2 провала (13, 16); T_ref — m8_ref.json (лучший найденный ДО HYB; ниже 1 возможно); хвост max 3.23 — старт 19 (путь спор 5.05); отношения пересчитаны по обновлённому m8_ref.json (старт 14: T_ref 3.39)'))
    for tg, nm, par_ in (('all', '**споры+удержание → OCP (HOLD8)** — прямое дерево и стыковка с потоком «удержание на пределе |w|=3», обратное ±1; refine спор НЕ используется: топ-3 попадания + топ-3 промаха → тёплый старт полного OCP (N50, свободное T)', 'NB 60000, NF 1500, kn 8, 3+3 кандидата; 10 процессов; numpy-поток удержания (не скомпилирован)'),
                         ('ctl', '**споры без удержания → OCP (контроль HOLD8)** — то же, но обычный поток ±1 (tree_fast), refine спор не используется: топ-3 попадания + топ-3 промаха → тёплый старт полного OCP', 'NB 60000, NF 1500, kn 8, 3+3 кандидата; 10 процессов; поток ±1 (скомпилирован)')):
        d = L(f'{D}/m8_hold8_{tg}.json')
        if not d: continue
        tt = np.array([r['t_total'] for r in d['res']]); nh = sum(r['hits'] > 0 for r in d['res'])
        R.append(dict(m=nm, par=par_, build=d['t_build'], qm=float(np.median(tt)), qp=float(np.percentile(tt, 90)), s={k: d[k] for k in ('reached', 'ratio_med', 'ratio_mean', 'ratio_p90', 'ratio_max')}, w=0, hw='CPU 1 ядро/запрос',
                        note='w30: попадания в дереве на %d/20 стартах (остальные — OCP из затравки-промаха); T_ref — m8_ref.json; число стартов с попаданием дерева — основной разграничитель вклада потока и OCP' % nh))
for var in ('multi', 'fast'):
    d = L(f'{D}/{env}_ocp6d_{var}.json')
    if d: R.append(dict(m='OCP CasADi/IPOPT на запрос (%s)' % ('мультистарт 8 ветвей×4 затравки = эталон' if var == 'multi' else 'ближайшая ветвь×2 затравки'), par='multiple shooting N50 M4', build=d['t_build'], qm=d['t_query_med'], qp=d['t_query_p90'], s=d, w=0, hw='CPU 1 проц.',
                        note=('сборка символьной задачи — «заготовка»; multi — тот же алгоритм, что дал T_ref, ⇒ 1.000 по построению' if env == 'm3d' else ('T_ref = лучший найденный среди методов; OCP multi поставляет большинство T_ref ⇒ его отношение ≈1 по построению' if env == 'm8' else 'T_ref — лучшее известное из r5 (N40, другой прогон IPOPT)')) if var == 'multi' else ''))
for st_ in (1000000, 5000000):
    P = [L(x) for x in sorted(glob.glob(f'{D}/{env}_ppo_{st_}_s*.json'))]; P = [p for p in P if p]
    if not P: continue
    pars = (9 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3 + 3) + (9 * 64 + 64 + 64 * 64 + 64 + 64 + 1)
    best = min(P, key=lambda s: (-(s['reached']), s['ratio_med'] or 9)); tr = np.mean([p['t_train'] for p in P])
    s = dict(best); s['reached'] = float(np.mean([p['reached'] for p in P])); s['ratio_med'] = float(np.mean([p['ratio_med'] for p in P if p['ratio_med']]))
    for k in ('ratio_mean', 'ratio_p90', 'ratio_max'): s[k] = float(np.mean([p[k] for p in P if p[k]]))
    R.append(dict(m='PPO %dM шагов (%d сида, среднее по сидам; доходимость сидов %s)' % (st_ // 10 ** 6, len(P), '/'.join('%d%%' % round(p['reached'] * 100) for p in P)), par='MlpPolicy 64×64, 16 env, CPU', build=tr, qm=np.mean([p['t_infer_per_start'] for p in P]), qp=None, s=s, w=pars * 4, hw='CPU',
                    note='до 95%% дошёл за %s с (сид 0); вес = %d параметров float32 (+VecNormalize)' % (f(P[0].get('t_train_95'), 3), pars)))
sst = L(f'{D}/{env}_sst.json')
if sst:
    for b, bb in zip(sst['budgets'], sst['budgets']):
        n_nodes = np.median([r['runs'][sst['budgets'].index(bb)]['nodes'] for r in sst['raw'] if r['runs'][sst['budgets'].index(bb)].get('nodes')]) if sst['raw'] else 0
        R.append(dict(m='SST (свой, prune .1) бюджет %g с' % bb['budget'], par='sel .2 / prune .1; одиночный запрос', build=0., qm=bb['budget'], qp=None, s=bb, w=n_nodes * (e.n + e.m + 3) * 8, hw='CPU numpy', note='запрос = бюджет (anytime); дерево строится в запросе'))
H = '| метод | параметры | заготовка, с | запрос мед. / p90, с | total(N=20), с | дошли | T/T_ref мед / ср / p90 / max | вес данных | железо | примечание |\n|---|---|---|---|---|---|---|---|---|---|\n'
T = H
for r in R:
    r['tot20'] = r['build'] + 20 * r['qm']; T += '| %s | %s | %s | %s / %s | %s | %s | %s | %s | %s |\n' % (r['m'], r['par'], f(r['build']), f(r['qm']), f(r['qp']), f(r['tot20']), st(r['s']), fb(r['w']), r['hw'], r['note'])
cr = ''; sp = R[0] if R and R[0]['m'].startswith('**споры**') else None
if sp:
    cr = '\n### Безубыточность N* (total = заготовка + N·запрос), относительно спор (v6)\n'
    for r in R[1:]:
        db = r['build'] - sp['build']; dq = sp['qm'] - r['qm']
        if r['build'] <= sp['build'] and r['qm'] <= sp['qm']: s_ = 'метод дешевле спор при любом N'
        elif r['build'] >= sp['build'] and r['qm'] >= sp['qm']: s_ = 'споры дешевле при любом N (N* = 0)'
        elif dq > 0 and db > 0: s_ = 'метод дешевле спор при N > %.0f' % (db / dq)
        else: s_ = 'споры дешевле метода при N > %.0f' % (db / dq)
        q = r['s']; worse = q['reached'] < sp['s']['reached'] - .02 or (q['ratio_med'] or 9) > (sp['s']['ratio_med'] or 9) * 1.05
        cr += '- %s: %s — качество %s: дошли %d%% (споры %d%%), T/T_ref мед %s (споры %s)\n' % (r['m'], s_, 'ХУЖЕ спор, N* несопоставим' if worse else 'не хуже спор', round(q['reached'] * 100), round(sp['s']['reached'] * 100), f(q['ratio_med']), f(sp['s']['ratio_med']))
Ns = np.logspace(0, 5, 60); fig, ax = plt.subplots(figsize=(9, 6)); cols = plt.cm.tab10.colors
for k, r in enumerate(R):
    if r['m'].startswith('SST') and not any(x in r['m'] for x in ('бюджет 1 с', 'бюджет 60 с')): continue
    ax.loglog(Ns, r['build'] + Ns * r['qm'], color=cols[k % 10], lw=2, label='%s [%d%%, %s]' % (r['m'][:46], round(r['s']['reached'] * 100), f(r['s']['ratio_med'], 3)))
ax.set_xlabel('число запросов N'); ax.set_ylabel('полное время, с = заготовка + N·запрос'); ax.set_title(TITLE + ' — total(N)'); ax.grid(True, which='both', alpha=.3); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{OUT}/total_{env}.png', dpi=110)
ex = int((~e.keep).sum())
hdr = f'# {TITLE}\n20 стартов (поток rng(0), исключено эталоном {ex}); эталон T_ref — OCP CasADi/IPOPT (multiple shooting, 8 ветвей 2π × 4 затравки{"; для 3g — лучшее известное из r5 ref6d_T.json (|w|≤3 по пути); для 8D эталона нет: T_ref = ЛУЧШИЙ НАЙДЕННЫЙ T среди всех методов на этом старте (отношения к нему, не к оптимуму; старт без решения исключён)" if env != "m3d" else ""}); единый проверяльщик `common.verify` (RK4 h .01) с допусками эталона: окно +1e-3, |w|≤3+1e-2. Цель — покой в q=0 (окно |q|<.3, |w|<.6). aida, CPU (запросы по 1 ядру; машина разделена с другими линиями, загрузка 20–25 из 32 ядер — времена запросов спор/OCP 8D завышены).\nПометка: T/T_ref < 1 возможно из-за допусков эталона и кусочно-постоянного u у эталона (N=50).\n\n'
open(f'{OUT}/table_{env}.md', 'w').write(hdr + T + cr + f'\n![total(N)](total_{env}.png)\n'); print(hdr + T + cr)
