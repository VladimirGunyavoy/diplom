import re, sys
s = open(sys.argv[1]).read()
s = s.replace("from src.atlas6.corridor_nd import corridor_query", "from src.atlas6.corridor_nd import corridor_query")
i0 = s.index("def work(i):"); i1 = s.index("if __name__")
new = '''import stitch_m8
if os.environ.get('CAREF'): stitch_m8.ORIG = CAR.orig
def work(i):
    e.nf = 0; t1 = time.perf_counter()
    b, info = stitch_m8.arm_local(S, fl, e.X0[i], back, tau, NF, rho, g, miss, K=int(os.environ.get('KP', 20)), wlim=(tuple(range(n, 2 * n)), WM) if WL else None, glob_tries=int(os.environ.get('GT', 3)), macc=int(os.environ.get('MACC', 1))); dt = time.perf_counter() - t1
    if b is None: return dict(i=i, T_agent=None, t_query=dt, T_verify=None, why='no_path', info=info)
    us = [US_[s] for s in b[1]]; Tv, why = common.verify(e, e.X0[i], us, list(b[2])); return dict(i=i, T_agent=float(b[0]), t_query=dt, T_verify=Tv, why=why, nseg=len(us), info=info)
'''
open(sys.argv[2], 'w').write(s[:i0] + new + s[i1:])
