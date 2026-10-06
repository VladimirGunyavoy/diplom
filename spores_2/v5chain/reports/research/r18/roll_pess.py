# research-18: агент на атласе DI 4D (рост от цели 4×400, goalfix ±.35) с пессимистичным заполнением interp (PESS) — дошли/T/T* на 60 стартах эталона.
import os, sys, pickle, json, time, numpy as np
import interp_lib as IF
G = IF.G
G.Atlas.interp = staticmethod(IF.interp_ren)
d = pickle.load(open(os.environ['LOAD'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']
Q, ref = G.starts_ref(); t0 = time.time(); T, sw, _ = A.rollout(Q); fz = np.isfinite(T); r = T[fz] / ref[fz]
print(json.dumps(dict(PESS=IF.PD, VF=G.VF, reach=f"{int(fz.sum())}/60", T_mean=round(float(r.mean()), 4) if fz.any() else None, T_med=round(float(np.median(r)), 4) if fz.any() else None, T_max=round(float(r.max()), 3) if fz.any() else None, sw_med=float(np.median(sw[fz])) if fz.any() else None, sec=round(time.time() - t0))), flush=True)
