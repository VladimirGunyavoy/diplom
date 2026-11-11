import sys,os,cProfile,pstats,time,importlib.util
sys.argv=['x','6000','1600','20','_prof']; sys.path.insert(0,os.path.expanduser('~/spore_v5/a37/spores_2/bench'))
spec=importlib.util.spec_from_file_location('sde',os.path.expanduser('~/spore_v5/a37/spores_2/bench/spores_drone_e.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ids=[int(a) for a in os.environ['IDS'].split(',')]
m.work(ids[0])  # прогрев numba
for i in ids:
    pr=cProfile.Profile(); t=time.perf_counter(); pr.enable(); r=m.work(i); pr.disable(); print('=== start',i,'wall %.2f'%(time.perf_counter()-t),'retries',r.get('retries'),r['why'],flush=True)
    pstats.Stats(pr).sort_stats(os.environ.get("SORT","cumulative")).print_stats(int(os.environ.get("NL",14)))
