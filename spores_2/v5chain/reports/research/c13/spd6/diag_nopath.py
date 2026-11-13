"""Блок 3 (1): причина no_path у недошедших стартов ДИ6D NB50k: сколько кандидатов-попаданий/промахов даёт дерево и что с доводкой."""
import sys, runpy, numpy as np, time
bad = [int(a) for a in sys.argv[1].split(',')]; NB, NF = sys.argv[2], sys.argv[3]
sys.argv = ['spores_di.py', 'di6', NB, NF, '3', 'dbg', '/home/random/spore_v5/r37/di6_ref_200_rho35_s8.npy']
g = runpy.run_path('spores_di.py', run_name='lib')
import src.atlas6.corridor_nd as C
S, back, e, tau, rho, gg, miss, flow = [g[k] for k in ('S', 'back', 'e', 'tau', 'rho', 'g', 'miss', 'flow')]; NFi = int(NF)
for i in bad:
    x0 = e.X0[i]; t0 = time.time(); Cd, nh = C.candidates(S, x0, back, tau, NFi, rho, miss, 8); dt = time.time() - t0
    mv = [round(float(c[0]), 3) for c in Cd[nh:nh + 3]]
    print('старт %d: T_ref %.3f, |x|max %.2f |v|max %.2f, остановочная точка max %.2f (keep=%s); кандидатов: попаданий %d, промахов %d (%.2f с); лучшие оценки попаданий %s' % (i, e.Tref[i], np.abs(x0[:3]).max(), np.abs(x0[3:]).max(), np.abs(x0[:3] + x0[3:] * np.abs(x0[3:]) / 2).max(), bool(e.keep[i]), nh, len(Cd) - nh, dt, [round(float(c[0]), 3) for c in Cd[:3]]), flush=True)
    b = C.corridor_query(S, flow, x0, back, tau, NFi, rho, gg, miss, K=3, tries=3, kn=8, neigh=0, nbest=1)
    print('   corridor_query: %s' % ('None' if b is None else 'T=%.3f' % b[0]), flush=True)
