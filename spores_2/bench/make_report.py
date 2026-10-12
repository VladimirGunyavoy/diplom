"""python make_report.py ENV [ENV...] → ../../reports/bench/table_<env>.md (+ anytime_<env>.png); агрегирует data/*.json. compare.md собирается вручную из таблиц."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = common.DATA; OUT = os.path.join(D, '..')
def L(p):
    try: return json.load(open(p))
    except Exception: return None
def f(x, n=4): return '—' if x is None else (('%.' + str(n) + 'g') % x)
def rows(env):
    R = []; sp = L(f'{D}/{env}_spores.json')
    def stat(s): return f"{s['reached']*100:.0f}% | {f(s['ratio_med'])} / {f(s['ratio_mean'])} / {f(s['ratio_p90'])} / {f(s['ratio_max'])}"
    if sp:
        b = sp['t_build']; q = sp['t_query_med']; R.append(dict(m='**споры** (' + sp['method'] + ')', par='cells ' + str(sp['cells']), build=b, qm=q, qp=sp['t_query_p90'], tot=b + sp['t_query_sum'], st=stat(sp), nf='—', hw=sp['hw'] + (' ' + sp['gpu'] if sp.get('gpu') else '')))
    for st in (1000000, 5000000):
        fs = sorted(glob.glob(f'{D}/{env}_ppo_{st}_s*.json')); P = [L(x) for x in fs]
        if P:
            best = min(P, key=lambda s: (s['ratio_med'] or 9)); tr = np.mean([s['t_train'] for s in P]); ti = np.mean([s['t_infer_per_start'] for s in P])
            R.append(dict(m=f'PPO {st//10**6}M шагов (лучший из {len(P)} сидов)', par=f"t_train до 95%: {f(best['t_train_95'])} с; ср. по сидам: reached {np.mean([s['reached'] for s in P])*100:.0f}%, мед {f(np.mean([s['ratio_med'] or np.nan for s in P]))}", build=tr, qm=ti, qp=ti, tot=tr + 20 * ti, st=stat(best), nf='—', hw=best['hw']))
    for algo, nm in (('rrt', 'RRT'), ('rrtstar', 'kRRT* (точный краевой)'), ('sst', 'SST (prune .1)')):
        d = L(f'{D}/{env}_{algo}.json')
        if not d: continue
        for b in d['budgets']:
            if algo == 'rrt' and b is not d['budgets'][-1]: continue
            n = len(d['starts']); R.append(dict(m=f'{nm}, бюджет {f(b["budget"])} с', par='', build=0, qm=b['t_wall_sum'] / n, qp=None, tot=b['t_wall_sum'], st=stat(b), nf=f(b['nf_med']), hw=d['hw']))
    oc = L(f'{D}/{env}_ocp.json')
    if oc: R.append(dict(m='OCP SLSQP (' + oc['method'] + ')', par='', build=0, qm=oc['t_query_med'], qp=oc['t_query_p90'], tot=oc['t_query_sum'], st=stat(oc), nf=f(oc['nf_med']), hw=oc['hw']))
    return R, sp
def table(env):
    R, sp = rows(env); lines = [f'### {env}', '', '| метод | примечание | t_build, с | t_query мед / p90, с | t_total(N=20), с | дошли | T/T_ref мед / mean / p90 / max | вызовов f (мед.) | N* vs споры | железо |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in R:
        ns = '—'
        if sp and 'споры' not in r['m']:
            qs = sp['t_query_med']; dq = r['qm'] - qs
            ns = ('споры дороже на запрос; N* = ∞' if dq <= 0 and r['build'] <= sp['t_build'] else 'N* = %.0f' % (max(sp['t_build'] - r['build'], 0) / dq) if dq > 0 else '—')
        lines.append(f"| {r['m']} | {r['par']} | {f(r['build'])} | {f(r['qm'])} / {f(r['qp'])} | {f(r['tot'])} | {r['st']} | {r['nf']} | {ns} | {r['hw']} |".replace('| ' + str(r['nf']) + ' | ' + ns, '| ' + str(r['nf']) + ' | ' + ns))
    return '\n'.join(lines)
def plot(env):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); col = dict(sst='C0', rrtstar='C1', rrt='C2')
    for algo, nm in (('sst', 'SST'), ('rrtstar', 'kRRT*'), ('rrt', 'RRT')):
        d = L(f'{D}/{env}_{algo}.json')
        if d:
            b = [x['budget'] for x in d['budgets']]; ax[0].plot(b, [x['ratio_med'] or np.nan for x in d['budgets']], 'o-', c=col[algo], label=nm); ax[1].plot(b, [x['reached'] for x in d['budgets']], 'o-', c=col[algo], label=nm)
    sp = L(f'{D}/{env}_spores.json')
    if sp: ax[0].plot([sp['t_query_med']], [sp['ratio_med']], 'k*', ms=12, label='споры (запрос; build %.1f с)' % sp['t_build']); ax[1].plot([sp['t_query_med']], [sp['reached']], 'k*', ms=12)
    oc = L(f'{D}/{env}_ocp.json')
    if oc: ax[0].plot([oc['t_query_med']], [oc['ratio_med']], 'ms', label='OCP SLSQP'); ax[1].plot([oc['t_query_med']], [oc['reached']], 'ms')
    for st, mk in ((1000000, '^'), (5000000, 'v')):
        P = [L(x) for x in glob.glob(f'{D}/{env}_ppo_{st}_s*.json')]
        if P: t = np.mean([s['t_infer_per_start'] for s in P]); ax[0].plot([t], [np.mean([s['ratio_med'] or np.nan for s in P])], mk, c='C3', label='PPO %dM (инференс)' % (st // 10**6)); ax[1].plot([t], [np.mean([s['reached'] for s in P])], mk, c='C3')
    for a, y in zip(ax, ('T/T_ref (медиана)', 'доля дошедших')): a.set_xscale('log'); a.set_xlabel('время на запрос, с'); a.set_ylabel(y); a.grid(alpha=.3)
    ax[0].set_ylim(.99, 2.0); ax[0].legend(fontsize=7); fig.suptitle(f'{env}: качество от времени на запрос'); fig.tight_layout(); fig.savefig(f'{OUT}/anytime_{env}.png', dpi=110)
if __name__ == '__main__':
    for env in sys.argv[1:]:
        open(f'{OUT}/table_{env}.md', 'w').write(table(env) + '\n'); plot(env); print(table(env))
