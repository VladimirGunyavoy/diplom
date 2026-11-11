"""research-30: ДИ 4D через growN как библиотеку (код growN не меняется): сборка с зерном SEED (или LOADL), solve (GPULOCK=1 — замок только на GPU),
прокатка 60 стартов бенча, T по стартам и p90. DUMP — сохранить слои+V (для d1_di4.py). Вывод JSON в OUTJ."""
import os, sys, json, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
t0 = time.time()
if os.environ.get('LOADL'):
    A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(os.environ['LOADL'], 'rb'))['layers']; A.finish(); tb = 0.
else:
    A = G.Atlas(seed=int(os.environ.get('SEED', 0))); tb = time.time() - t0
    if os.environ.get('BONLY'): pickle.dump(dict(layers=A.layers), open(os.environ['BONLY'], 'wb')); json.dump(dict(t_build=tb), open(os.environ['BONLY'] + '.json', 'w')); sys.exit(0)   # сборка без CUDA (системный python), solve — отдельно
t1 = time.time(); A.solve(); ts = time.time() - t1
if os.environ.get('DUMP'): pickle.dump(dict(layers=A.layers, V=A.V), open(os.environ['DUMP'], 'wb'))
Q, ref = G.starts_ref(); t2 = time.time(); T, sw, _ = A.rollout(Q); tq = time.time() - t2
fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]
out = dict(seed=os.environ.get('SEED'), loadl=os.environ.get('LOADL'), DTN=G.DTN, M=G.M, cells=[len(l) for l in A.layers], nodes=int(A.N), t_build=tb, t_solve=ts, t_query60=tq,
           reach=float(np.isfinite(T).mean()), mean=float(r.mean()), med=float(np.median(r)), p90=float(np.percentile(r, 90)), max=float(r.max()), T=[float(x) for x in T])
print(json.dumps({k: v for k, v in out.items() if k != 'T'}), flush=True); json.dump(out, open(os.environ['OUTJ'], 'w'), indent=1)
