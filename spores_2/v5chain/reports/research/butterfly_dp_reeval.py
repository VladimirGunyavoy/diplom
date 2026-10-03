"""research-10: переоценка сохранённого атласа (раунд эллипса) агентом rollout_edges с запасным мини-деревом (ALLN/PLANFB)."""
import sys, os, time, json, numpy as np
sys.path.insert(0, '.')
import butterfly_dp_query as Q
from butterfly_dp_atlas import MN
A = Q.load(sys.argv[1]); t0 = time.time(); T, arcs, wm = Q.rollout_edges(A, 81, 0.)
print(json.dumps(dict(atlas=sys.argv[1][-12:], spores=A.K, V=round(float(A.V[81, MN // 2]), 3), T=round(float(T), 3), T_over_OCP=round(float(T / 5.098), 4), arcs=arcs,
                      plans=getattr(A, 'nplan', 0), wmax=round(float(wm), 2), sec=round(time.time() - t0))), flush=True)
