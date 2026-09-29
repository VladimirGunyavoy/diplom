"""research: атлас дифдрайва (ромб-U) — мельчить (x, y) и θ ВМЕСТЕ при окне цели, закреплённом в физ. единицах (R = .25, Rθ = .26), τ = h.
Сравнение с эталоном min(TGT, TGTGT) на 40 позах (как v6/tests/check_dd_vs_ref.py)."""
import sys, importlib.util, numpy as np, time
sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v6')
import src.atlas6.dd_atlas as dd
sp = importlib.util.spec_from_file_location('ref', '/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/dd_rhombus_ref.py'); ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
rng = np.random.default_rng(3); poses = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(40)]
R = np.array([min(ref.tgt(*p), ref.tgtgt(*p)) for p in poses])
for h, nth in ((0.25, 24), (0.125, 48), (0.25, 48), (0.125, 24)):
    t = time.time(); A = dd.solve_dd(h=h, nth=nth, R_goal=0.25, Rth=2*np.pi/24); d = np.array([dd.V_at(A, p) for p in poses]) / R
    print(f"h={h} nth={nth}: V/эталон mean {d.mean():.3f} max {d.max():.3f} min {d.min():.3f}  ({time.time()-t:.0f} с, {A['iters']} итераций)", flush=True)
