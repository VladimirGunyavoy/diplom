"""python make_report.py ENV [ENV...] → reports/bench/table_<env>.md + anytime_<env>.png из data/*.json."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = common.DATA; OUT = os.path.join(D, '..')
def L(p):
    try: return json.load(open(p))
    except Exception: return None
def f(x, n=4): return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else (('%.' + str(n) + 'g') % x)
def stat(s): return f"{s['reached']*100:.0f}% | {f(s['ratio_med'])} / {f(s['ratio_mean'])} / {f(s['ratio_p90'])} / {f(s['ratio_max'])}"
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
        b = sp['t_build']; R.append(dict(m='**споры**: ' + sp['method'], par='клеток ' + str(sp['cells']) + ('; запрос по одному, dt агента .01' if env == 'di' else ''), build=b, qm=sp['t_query_med'], qp=sp['t_query_p90'], tot=b + sp['t_query_sum'], st=stat(sp), nf='не считали', batch=sp.get('t_batch20'), hw=sp['hw'], sp=True))
        if sp0 is None: sp0 = R[-1]
        if sp.get('dt05'): d = sp['dt05']; R.append(dict(m='споры, агент с dt .05 (= шаг PPO)', par='', build=b, qm=d['t_query_med'], qp=None, tot=b + d['t_query_sum'], st=stat(d), nf='—', batch=None, hw=sp['hw'], sp=True))
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
        R.append(dict(m=f'PPO {st//10**6}M шагов ({len(P)} сида: показан лучший; ср. дошли {np.mean([s["reached"] for s in P])*100:.0f}%, мед {f(np.mean([s["ratio_med"] or np.nan for s in P]))})', par='до качества мед ≤1.01 и 100%: ' + (f(tq, 3) + ' с' if tq else 'не достиг'), build=tr, qm=ti, qp=ti, tot=tr + 20 * ti, st=stat(best), nf=f'обучение {best.get("nf_train", 0):.2g}', batch=tb, hw=best['hw'], sp=False))
    for algo, nm in (('rrt', 'RRT (свой)'), ('rrtstar', 'kRRT* (точный краевой, ДИ)'), ('sst', 'SST (свой; prune ' )):
        d = L(f'{D}/{env}_{algo}.json')
        if not d: continue
        for b in d['budgets']:
            if algo == 'rrt' and b is not d['budgets'][-1]: continue
            n = len(d['starts']); par = ''
            if algo == 'rrtstar':
                h = [(r['runs'][-1].get('hist') or []) for r in d['raw']]; tf = [x[0][0] for x in h if x]; t1 = [next((t for t, c in x if c <= 1.01 * x[-1][1]), np.nan) for x in h if x]
                if b is d['budgets'][-1]: par = f'до 1-го решения мед {f(np.median(tf),2)} с; до цены ≤1% от итога мед {f(np.nanmedian(t1),2)} с'
            R.append(dict(m=(nm + (str(d.get('prune')) + ')' if algo == 'sst' else '')) + f', бюджет {f(b["budget"],3)} с', par=par, build=0., qm=b['t_wall_sum'] / n, qp=None, tot=b['t_wall_sum'], st=stat(b), nf=f'на запрос {f(b["nf_med"],3)}', batch=None, hw=d['hw'], sp=False))
    oc = L(f'{D}/{env}_ocp.json')
    if oc: R.append(dict(m='OCP SLSQP (' + oc['method'] + ')', par='', build=0., qm=oc['t_query_med'], qp=oc['t_query_p90'], tot=oc['t_query_sum'], st=stat(oc), nf=f'на запрос {f(oc["nf_med"],3)}', batch=None, hw=oc['hw'], sp=False))
    return R, sp0
def table(env):
    R, sp0 = rows(env); lines = [f'### {env}', '', '| метод | примечание | t_build, с | t_query мед / p90, с | пакет 20, с | t_total(N=20), с | дошли | T/T_ref мед / mean / p90 / max | вызовов f | N* относительно спор | железо |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R:
        ns = '—' if r['sp'] or not sp0 else cross(r, sp0)
        lines.append(f"| {r['m']} | {r['par']} | {f(r['build'])} | {f(r['qm'])} / {f(r['qp'])} | {f(r['batch'])} | {f(r['tot'])} | {r['st']} | {r['nf']} | {ns} | {r['hw']} |")
    return '\n'.join(lines)
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
        t = table(env); open(f'{OUT}/table_{env}.md', 'w').write(t + '\n'); plot(env); print(t)
