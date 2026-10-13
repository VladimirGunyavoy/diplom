import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env_lw as W, lw_dyn as L, common
from lw_dyn import C
from legwheel_robot import env as E
e = W.LW(50, ref_file='/nonexistent'); o = json.load(open(os.path.join(common.DATA, 'lw_ocp_d.json')))['per_start']; mx = np.zeros(6); sw = []
for r in o:
    if r['T'] is None: continue
    y = list(L.to_phys(e.X0[r['i']])); U = np.array(r['U']); tr = [np.array(L.from_phys(y))]
    for u in U:
        y = E.rk4_step(y, L.UMAX[0] * u[0], L.UMAX[1] * u[1], C.PHYSICS_DT); tr.append(L.from_phys(y))
    tr = np.array(tr); mx = np.maximum(mx, np.abs(tr).max(0)); sw.append(((np.abs(U) < .99).mean(), len(U)))
print('макс |z| по эталонным траекториям', np.round(mx, 1)); print('доля промежуточных u (|u|<.99) и N:', sw[:15])
print('старты |z| макс', np.abs(e.X0).max(0))
