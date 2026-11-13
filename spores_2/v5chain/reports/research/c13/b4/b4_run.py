"""блок 4: python b4_run.py ENV NB NF K tag STARTS ARM   (ARM = base | local | nof); окружение как spores_di.py; PROCS параллельных запросов (fork после build_back)."""
import sys, os, runpy, time, json, numpy as np, multiprocessing as mp
ENV, NB, NF, K, tag, STARTS, ARM = sys.argv[1:8]
sys.path.insert(0, os.path.expanduser('~/spore_v5/c13/spd6')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['spores_di.py', ENV, NB, NF, K, tag, STARTS]
G = runpy.run_path(os.path.expanduser('~/spore_v5/c13/spd6/spores_di.py'), run_name='lib')
S, back, e, tau, rho, g, miss, flow, US_, common, corridor_query = [G[k] for k in ('S', 'back', 'e', 'tau', 'rho', 'g', 'miss', 'flow', 'US_', 'common', 'corridor_query')]
import stitch; NF = int(NF); K = int(K); t_build = G['t_build']
print('HEADER b4: рука %s ENV %s NB %s NF %d K %d (KK пар = 20); эталон/старты %s; запас цели .004; общий алгоритм (S, flow, g, miss)' % (ARM, ENV, NB, NF, K, STARTS), flush=True)
def work(i):
    if not e.keep[i]: return dict(i=i, skip=True)
    t1 = time.perf_counter(); info = {}
    if ARM == 'base': b = corridor_query(S, flow, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, neigh=0, nbest=1)
    elif ARM == 'fallback':   # база; если пути нет — рука (б)
        b = corridor_query(S, flow, e.X0[i], back, tau, NF, rho, g, miss, K=K, tries=3, kn=8, neigh=0, nbest=1)
        if b is None: b, info = stitch.arm_local(S, flow, e.X0[i], back, tau, NF, rho, g, miss, K=20); info['fb'] = 1
    elif ARM == 'local': b, info = stitch.arm_local(S, flow, e.X0[i], back, tau, NF, rho, g, miss, K=int(os.environ.get('KP', 20)), glob_tries=int(os.environ.get('GT', 2)), macc=int(os.environ.get('MACC', 1)))
    else: b, info = stitch.arm_nof(S, flow, e.X0[i], back, tau, g, K=20, rho=rho)
    dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path', info=info)
    us = [US_[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, info=info)
if __name__ == '__main__':
    NS = int(os.environ.get('NSTART', len(e.X0))); t0 = time.perf_counter()
    with mp.get_context('fork').Pool(int(os.environ.get('PROCS', 4))) as p: res = p.map(work, range(NS), chunksize=1)
    wall = time.perf_counter() - t0; rr = [r for r in res if not r.get('skip')]; T = np.array([np.nan if r['T_verify'] is None else r['T_verify'] for r in rr]); ref = np.array([e.Tref[r['i']] for r in rr])
    ok = np.isfinite(T); ratio = T[ok] / ref[ok]; tq = np.array([r['t_query'] for r in rr]); why = {w: sum(1 for r in rr if r['why'] == w) for w in set(r['why'] for r in rr)}
    s = dict(arm=ARM, env=ENV, NB=NB, NF=NF, K=K, n=len(rr), reach=int(ok.sum()), why=why, slow3=int((ratio > 3).sum()), med=float(np.median(ratio)) if ok.any() else None, p90=float(np.percentile(ratio, 90)) if ok.any() else None, max=float(ratio.max()) if ok.any() else None,
             t_build=t_build, t_query_med=float(np.median(tq)), t_query_p90=float(np.percentile(tq, 90)), wall=wall, per_start=res)
    if ARM != 'base': s['pairs_med'] = float(np.median([r['info'].get('pairs', 0) for r in rr])); s['local_ok_frac'] = float(np.mean([r['info'].get('local_ok', 0) > 0 for r in rr]))
    o = os.path.expanduser('~/spore_v5/c13/out_b4'); os.makedirs(o, exist_ok=True); json.dump(s, open('%s/%s_%s_%s.json' % (o, ENV, ARM, tag), 'w'), indent=1, default=float)
    print({k_: v for k_, v in s.items() if k_ != 'per_start'}, flush=True)
