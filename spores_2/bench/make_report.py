"""python make_report.py ENV [ENV...] → reports/bench/table_<env>.md + anytime_<env>.png из data/*.json."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = common.DATA; OUT = os.path.join(D, '..')
def L(p):
    try: d = json.load(open(p))
    except Exception: return None
    return d
def f(x, n=4): return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else (('%.' + str(n) + 'g') % x)
class S(str): pass
def stat(s):
    x = S(f"{s['reached']*100:.0f}% | {f(s['ratio_med'])} / {f(s['ratio_mean'])} / {f(s['ratio_p90'])} / {f(s['ratio_max'])}"); x.reached = s['reached']; x.med = s['ratio_med'] or 9.; return x
def fb(b): return '—' if not b else ('%.3g ГБ' % (b / 1e9) if b >= 1e9 else '%.3g МБ' % (b / 1e6) if b >= 1e6 else '%.3g КБ' % (b / 1e3))
def wtxt(n, b): return '—' if not b else f'{n:,} шт., {fb(b)}'.replace(',', ' ') if n else fb(b)
WP = L(f'{D}/ppo_weight.json') or {}
def w_spores(sp):
    w = sp.get('weight'); return (w['elems'], w['bytes']) if w else (sp.get('nodes') or 0, 0)
def w_tree(d, env, bi=-1):
    """узлы дерева на конец запроса (мед. по стартам, последний бюджет) × (n+m+3)·8 байт (состояние, управление, длительность, цена, родитель, индекс)"""
    e = common.make(env); nd = [r['runs'][bi]['nodes'] for r in d['raw'] if r['runs'][bi].get('nodes')]
    if not nd: return (0, 0)
    n = int(np.median(nd)); return (n, n * (e.n + e.m + 3) * 8)
def cross(o, s):
    """N*: при каком числе запросов N итог (build + N·q) метода o и спор сравнивается (в обе стороны). o, s: dict(build, qm)"""
    db = o['build'] - s['build']; dq = s['qm'] - o['qm']                      # o дешевле при N > db/dq, если dq > 0 и db > 0
    if o['build'] <= s['build'] and o['qm'] <= s['qm']: return 'метод дешевле спор всегда'
    if o['build'] >= s['build'] and o['qm'] >= s['qm']: return 'споры дешевле всегда'
    if dq > 0 and db > 0: return 'метод дешевле спор при N > %.0f' % (db / dq)
    if dq < 0 and db < 0: return 'споры дешевле метода при N > %.0f' % (db / dq)
    return '—'
def rows(env):
    R = []; sps = [(L(f'{D}/{env}_spores.json'), ''), (L(f'{D}/{env}_spores_T.json'), '')]; sp0 = None
    for sp, _ in sps:
        if not sp: continue
        b = sp['t_build']; R.append(dict(m='**споры**: ' + sp['method'], par='клеток ' + str(sp['cells']) + ('; запрос по одному, dt агента .01' if env == 'di' else '') + ('; 2/20 (старты 3, 10) — финиш нарушает |ω|≤3 (физический предел), неуспех верный' if env == 'manip' else ''), build=b, qm=sp['t_query_med'], qp=sp['t_query_p90'], tot=b + sp['t_query_sum'], st=stat(sp), nf='не считали', batch=sp.get('t_batch20'), hw=sp['hw'], sp=True, w=w_spores(sp)))
        if sp0 is None: sp0 = R[-1]
        if sp.get('dt05'): d = sp['dt05']; R.append(dict(m='споры, агент с dt .05 (= шаг PPO)', par='', build=b, qm=d['t_query_med'], qp=None, tot=b + d['t_query_sum'], st=stat(d), nf='—', batch=None, hw=sp['hw'], sp=True, w=w_spores(sp)))
    spq = sp0['qm'] if sp0 else None
    for st in (1000000, 5000000):
        P = [L(x) for x in sorted(glob.glob(f'{D}/{env}_ppo_{st}_s*.json'))]; P = [p for p in P if p]
        if not P: continue
        best = min(P, key=lambda s: (s['ratio_med'] or 9)); tr = np.mean([s['t_train'] for s in P]); ti = np.mean([s['t_infer_per_start'] for s in P]); tb = np.mean([s.get('t_infer_batch20') or np.nan for s in P])
        tq = None
        if sp0:
            tgt = None
            for p in P:
                for c in p['curve']:
                    if c['reached'] >= 1.0 and c['ratio_med'] and c['ratio_med'] <= 1.01: tgt = c['t_train'] if tgt is None else min(tgt, c['t_train'])
            tq = tgt
        R.append(dict(m=f'PPO {st//10**6}M шагов ({len(P)} сида: показан лучший; ср. дошли {np.mean([s["reached"] for s in P])*100:.0f}%, мед {f(np.mean([s["ratio_med"] or np.nan for s in P]))})', par='до качества мед ≤1.01 и 100%: ' + (f(tq, 3) + ' с' if tq else 'не достиг'), build=tr, qm=ti, qp=ti, tot=tr + 20 * ti, st=stat(best), nf=f'обучение {best.get("nf_train", 0):.2g}', batch=tb, hw=best['hw'], sp=False, w=(WP.get(env, {}).get('params', 0), WP.get(env, {}).get('bytes', 0) + WP.get(env, {}).get('vecnorm_bytes', 0))))
    for x in sorted(glob.glob(f'{D}/{env}_ppo_5000000_*_s*.json')):
        s = L(x); tag = os.path.basename(x).split('5000000_')[1].rsplit('_s', 1)[0]
        if s: R.append(dict(m=f'PPO 5M, {tag.replace("g0.", "γ .").replace("_pen", " штраф выхода −(T_ep+5)")}, сид {s["seed"]} (1 прогон)', par='до 95%: ' + ('не достиг' if not s.get('t_train_95') else f(s['t_train_95'], 3) + ' с'), build=s['t_train'], qm=s['t_infer_per_start'], qp=s['t_infer_per_start'], tot=s['t_train'] + 20 * s['t_infer_per_start'], st=stat(s), nf=f'обучение {s.get("nf_train", 0):.2g}', batch=s.get('t_infer_batch20'), hw=s['hw'], sp=False, w=(WP.get(env, {}).get('params', 0), WP.get(env, {}).get('bytes', 0) + WP.get(env, {}).get('vecnorm_bytes', 0))))
    for algo, nm in (('rrt', 'RRT (свой)'), ('rrtstar', 'kRRT* (точный локальный планировщик: ДИ — bang-bang, дд — Balkcom–Mason TGT)'), ('sst', 'SST (свой; prune ' )):
        d = L(f'{D}/{env}_{algo}.json')
        if not d: continue
        for b in d['budgets']:
            if algo == 'rrt' and b is not d['budgets'][-1]: continue
            n = len(d['starts']); par = ''
            if algo == 'rrtstar':
                h = [(r['runs'][-1].get('hist') or []) for r in d['raw']]; tf = [x[0][0] for x in h if x]; t1 = [next((t for t, c in x if c <= 1.01 * x[-1][1]), np.nan) for x in h if x]
                if b is d['budgets'][-1]: par = f'до 1-го решения мед {f(np.median(tf),2)} с; до цены ≤1% от итога мед {f(np.nanmedian(t1),2)} с'
            R.append(dict(m=(nm + (str(d.get('prune')) + ')' if algo == 'sst' else '')) + f', бюджет {f(b["budget"],3)} с', par=par, build=0., qm=b['t_wall_sum'] / n, qp=None, tot=b['t_wall_sum'], st=stat(b), nf=f'на запрос {f(b["nf_med"],3)}', batch=None, hw=d['hw'], sp=False, w=w_tree(d, env, d['budgets'].index(b))))
    oc = L(f'{D}/{env}_ocp.json')
    if oc: R.append(dict(m='OCP SLSQP (' + oc['method'] + ')', par='', build=0., qm=oc['t_query_med'], qp=oc['t_query_p90'], tot=oc['t_query_sum'], st=stat(oc), nf=f'на запрос {f(oc["nf_med"],3)}', batch=None, hw=oc['hw'], sp=False, w=(0, 0)))
    oc = L(f'{D}/{env}_ocpc.json')
    if oc: R.append(dict(m='OCP CasADi+IPOPT (' + oc['method'] + ')', par='', build=0., qm=oc['t_query_med'], qp=oc['t_query_p90'], tot=oc['t_query_sum'], st=stat(oc), nf=f'IPOPT-запусков на запрос {f(oc["nf_med"],3)}', batch=None, hw=oc['hw'], sp=False, w=(0, 0)))
    if env == 'dd': R += rows_dd2()
    return R, sp0
def rows_dd2():
    R = []
    for algo, nm in (('sst', 'SST'), ('rrt', 'RRT')):
        d = L(f'{D}/dd2_{algo}.json')
        if not d: continue
        b = d['budgets'][-1]; n = len(d['starts']); R.append(dict(m=f'[цель ±.2] {nm}, бюджет {f(b["budget"],3)} с (T/T_ref к эталону ±.05)', par='чувствительность к размеру цели', build=0., qm=b['t_wall_sum'] / n, qp=None, tot=b['t_wall_sum'], st=stat(b), nf=f'на запрос {f(b["nf_med"],3)}', batch=None, hw=d['hw'], sp=False, w=w_tree(d, 'dd', -1)))
    for x in sorted(glob.glob(f'{D}/dd2_ppo_5000000_s*.json')):
        s = L(x)
        if s: R.append(dict(m=f'[цель ±.2] PPO 5M, сид {s["seed"]} (T/T_ref к эталону ±.05)', par='чувствительность к размеру цели', build=s['t_train'], qm=s['t_infer_per_start'], qp=s['t_infer_per_start'], tot=s['t_train'] + 20 * s['t_infer_per_start'], st=stat(s), nf='', batch=None, hw=s['hw'], sp=False, w=(0, 0)))
    return R
def breakeven(o, s):
    """с какого N споры выгоднее по полному времени (заготовка + N·запрос); без учёта качества — оно в соседних столбцах"""
    if o is s: return '—'
    if s['build'] <= o['build'] and s['qm'] <= o['qm']: return 'при любом N'
    if s['build'] >= o['build'] and s['qm'] >= o['qm']: return 'никогда (метод дешевле при любом N)'
    n = (s['build'] - o['build']) / (o['qm'] - s['qm'])
    return ('N > %.0f' % n) if o['qm'] > s['qm'] else 'N < %.0f' % abs(n)
def table(env):
    R, sp0 = rows(env); lines = [f'### {env}', '', '| метод | примечание | заготовка, с | t_query мед / p90, с | пакет 20, с | t_total(N=20), с | дошли | T/T_ref мед / mean / p90 / max | вызовов f | вес (данные для запроса) | споры выгоднее при N в … | железо |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R:
        ns = '—' if r['sp'] or not sp0 else breakeven(r, sp0)
        if ns != '—' and r['st'].med < 9 and sp0['st'].med < 9:
            dm = (r['st'].med / sp0['st'].med - 1) * 100
            if dm > 1: ns += f' (T хуже: мед +{dm:.0f}%)'
            elif dm < -1: ns += f' (T лучше: мед {dm:.0f}%)'
        if ns != '—' and r['st'].reached < sp0['st'].reached - 1e-9: ns += f' (дошли {r["st"].reached*100:.0f}% против {sp0["st"].reached*100:.0f}%)'
        lines.append(f"| {r['m']} | {r['par']} | {f(r['build'])} | {f(r['qm'])} / {f(r['qp'])} | {f(r['batch'])} | {f(r['tot'])} | {r['st']} | {r['nf']} | {wtxt(*r['w']) if r.get('w') else '—'} | {ns} | {r['hw']} |")
    lines += ['', 'Заготовка: споры = рост+solve, PPO = обучение, RRT/SST/kRRT*/OCP = 0 (дерево/решение строится внутри запроса и входит в t_query). Вес: споры — сумма nbytes массивов атласа (узлы, рёбра, V, геометрия клеток); PPO — параметры/байты state_dict + VecNormalize; SST/RRT — узлы дерева на конец запроса (мед.) × (n+m+3)·8 байт; OCP — нет постоянных данных. «Споры выгоднее при N»: по полному времени заготовка+N·запрос, качество в соседних столбцах.']
    return '\n'.join(lines)
def total_plot(env):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    R, sp0 = rows(env); N = np.logspace(0, 5, 100); fig, ax = plt.subplots(figsize=(9, 6)); fam = {}
    for r in R:
        if r['qm'] is None or r['m'].startswith('[цель'): continue
        k = r['m'].replace('**', '').split(' (')[0].split(',')[0].split(' шаг')[0].split(' бюджет')[0]
        k = 'споры' if k.startswith('споры') else ('PPO' if k.startswith('PPO') else ('SST' if k.startswith('SST') else k))
        fam.setdefault(k, []).append(r)
    best = {}
    for k, rs in fam.items():
        ok = [r for r in rs if r['st'].reached >= .9 and r['st'].med <= 1.3] or [r for r in rs if r['st'].reached >= .9] or rs; best[k] = min(ok, key=lambda r: r['build'] + 1000 * r['qm'])
    for ci, (k, rs) in enumerate(fam.items()):
        for r in rs:
            if r is best[k]: continue
            ax.plot(N, r['build'] + N * r['qm'], c=f'C{ci}', lw=.7, alpha=.4)
        r = best[k]; ax.plot(N, r['build'] + N * r['qm'], c=f'C{ci}', lw=3, label=f"{k}: заготовка {f(r['build'],3)} с, запрос {f(r['qm'],3)} с, дошли {r['st'].reached*100:.0f}%, T/T_ref {r['st'].med:.2f}")
    if 'споры' in best:
        sp = best['споры']
        for k, r in best.items():
            if k == 'споры': continue
            dq = r['qm'] - sp['qm']
            if abs(dq) < 1e-12: continue
            n = (sp['build'] - r['build']) / dq
            if 1 <= n <= 1e5: ax.plot([n], [sp['build'] + n * sp['qm']], 'ko', ms=7); ax.annotate(f'N≈{n:.0f}', (n, sp['build'] + n * sp['qm']), fontsize=8, xytext=(4, -12), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('число запросов N'); ax.set_ylabel('полное время total(N) = заготовка + N·запрос, с'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7, loc='upper left')
    ax.set_title(f'{env}: полное время от числа запросов (жирная — лучший вариант метода; точки — безубыточность со спорами)'); ax.title.set_fontsize(9); fig.tight_layout(); fig.savefig(f'{OUT}/total_{env}.png', dpi=110)
def plot(env):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); col = dict(sst='C0', rrtstar='C1', rrt='C2')
    for algo, nm in (('sst', 'SST'), ('rrtstar', 'kRRT*'), ('rrt', 'RRT')):
        d = L(f'{D}/{env}_{algo}.json')
        if d:
            b = [x['budget'] for x in d['budgets']]; ax[0].plot(b, [x['ratio_med'] or np.nan for x in d['budgets']], 'o-', c=col[algo], label=nm); ax[1].plot(b, [x['reached'] for x in d['budgets']], 'o-', c=col[algo], label=nm)
    for nm, mk in (('spores', 'k*'), ('spores_T', 'k^')):
        sp = L(f'{D}/{env}_{nm}.json')
        if sp: ax[0].plot([sp['t_query_med']], [sp['ratio_med']], mk, ms=11, label='споры%s (запрос; build %.0f с)' % (' по T' if nm.endswith('T') else '', sp['t_build'])); ax[1].plot([sp['t_query_med']], [sp['reached']], mk, ms=11)
    oc = L(f'{D}/{env}_ocp.json')
    if oc: ax[0].plot([oc['t_query_med']], [oc['ratio_med']], 'ms', label='OCP SLSQP'); ax[1].plot([oc['t_query_med']], [oc['reached']], 'ms')
    for st, mk in ((1000000, '^'), (5000000, 'v')):
        P = [L(x) for x in glob.glob(f'{D}/{env}_ppo_{st}_s*.json')]; P = [p for p in P if p]
        if P: t = np.mean([s['t_infer_per_start'] for s in P]); ax[0].plot([t], [np.nanmean([s['ratio_med'] or np.nan for s in P])], mk, c='C3', label='PPO %dM (инференс)' % (st // 10**6)); ax[1].plot([t], [np.mean([s['reached'] for s in P])], mk, c='C3')
    for a, y in zip(ax, ('T/T_ref (медиана)', 'доля дошедших')): a.set_xscale('log'); a.set_xlabel('время на запрос, с'); a.set_ylabel(y); a.grid(alpha=.3)
    ax[0].set_ylim(.99, 2.0); ax[0].legend(fontsize=6); fig.suptitle(f'{env}: качество от времени на запрос'); fig.tight_layout(); fig.savefig(f'{OUT}/anytime_{env}.png', dpi=110)
if __name__ == '__main__':
    for env in sys.argv[1:]:
        t = table(env); open(f'{OUT}/table_{env}.md', 'w').write(t + '\n'); plot(env); total_plot(env); print(t)
