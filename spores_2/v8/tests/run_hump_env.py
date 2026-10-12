"""hump_k.py с env окна (M=5, SOLVE8_PLUS=1, V8_MAXCELLS=N из argv[1]); cwd = v8. python3 tests/run_hump_env.py N"""
import os, sys, runpy
os.environ['V8_MAXCELLS'] = sys.argv[1]; sys.path.insert(0, os.getcwd())
from src.stepper import live
os.environ.setdefault('KS', '0'); print('env', {k: os.environ.get(k) for k in ('M', 'MAXC', 'SOLVE8_PLUS', 'COMMIT_EVERY', 'XL', 'VL', 'GMX')}, flush=True)
runpy.run_path('reports/laptop_b/hump_k.py', run_name='__main__')
