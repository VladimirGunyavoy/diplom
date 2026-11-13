import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.argv = ['x', 'out/g6d_m3d_100000_0_p.json', '/dev/null']
exec(open(os.path.expanduser('~/spore_v5/d4/refine_graph.py')).read().split("if __name__")[0])
FAIL = [0, 2, 5, 7, 8, 10, 16, 18, 19]
for r in D['res']:
    if r['i'] not in FAIL: continue
    i = r['i']; path = [(idx(u), t) for u, t in zip(r['us'], r['dts'])]
    CA.st.clear(); CA.stat.clear(); CA.stat.update(n=0, t=0., it=0); t0 = time.perf_counter()
    T, seq, d, ok = CA.refine(fl, e.X0[i], path, g, tries=3, wlim=((3, 4, 5), WM))
    from src.atlas6.corridor_nd import merge, endpoint
    sq, d0 = merge(path); xe = endpoint(fl, e.X0[i], sq, d0)
    print(i, 'nseg', len(sq), 'ok', ok, 'status', dict(CA.st), 'stat', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in CA.stat.items()}, 'T0 %.2f' % d0.sum(), 'miss_end %s' % np.round(np.minimum(g(xe), 0), 3), 't %.2f' % (time.perf_counter() - t0), flush=True)
