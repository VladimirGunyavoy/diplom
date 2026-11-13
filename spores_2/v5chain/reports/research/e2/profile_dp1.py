"""e2: cProfile одного старта dp1 коридора (конфиг (ii): NSW 64, WMARG .05); back — из pickle прогона. Запуск: W43=... NSW=64 WMARG=.05 WL=1 python profile_dp1.py <i> <back.pkl>"""
import sys, os, time, pickle, cProfile, pstats, io
sys.argv = ['corridor_env.py', 'dp1', '_prof'] + sys.argv[1:]
import corridor_env as C
i = int(sys.argv[3]); pk = sys.argv[4]
C.init_w(pk)
pr = cProfile.Profile(); t0 = time.perf_counter(); pr.enable(); r = C.work(i); pr.disable(); dt = time.perf_counter() - t0
print('start', i, 'T_verify', r['T_verify'], 'why', r['why'], 'wall %.1f с' % dt)
for key in ('cumulative', 'tottime'):
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(key).print_stats(14); print('== sort', key); print(s.getvalue()[:3800])
