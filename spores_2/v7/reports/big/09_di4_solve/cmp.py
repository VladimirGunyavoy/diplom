"""ДИ 4D: сравнение solve (worker-36). MOD=growN|growNj; слои — LOADL (pickle layers), V -> OUT/V_<TAG>.npy, rollout 60 стартов. Запуск из ~/spore_v5/a31/spores_2/v7."""
import os, sys, json, time, pickle, numpy as np
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'cells7')); G = __import__(os.environ.get('MOD', 'growN'))
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
d = pickle.load(open(os.environ['LOADL'], 'rb')); t0 = time.time(); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); tf = time.time() - t0
t1 = time.time(); A.solve(); ts = time.time() - t1; np.save(os.path.join(os.environ['OUT'], 'V_%s.npy' % os.environ['TAG']), A.V.astype(np.float32))
Q, ref = G.starts_ref(); T, sw, _ = A.rollout(Q); fz = np.isfinite(T) & np.isfinite(ref); r = T[fz] / ref[fz]; np.save(os.path.join(os.environ['OUT'], 'T_%s.npy' % os.environ['TAG']), T)
print(json.dumps(dict(tag=os.environ['TAG'], nodes=int(A.N), iters=int(A.n_it), t_finish=round(tf, 1), t_solve=round(ts, 1), reach=round(float(np.isfinite(T).mean()), 3), mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), p90=round(float(np.quantile(r, .9)), 4), max=round(float(r.max()), 3))), flush=True)
