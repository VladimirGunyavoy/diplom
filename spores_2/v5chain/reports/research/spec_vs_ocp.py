"""research hub-research-7 (aida): спектр/обобщённая цена DI — J агента (bang/tri/spec11) против эталона ПМП-OCP (u = sat(α+βt)) по каждому старту."""
import sys, json, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from v7_spore_di import Cover
from v7_spore_spec import solve_U, rollout_U
from ref_di_cost import ocp_ref
rng = np.random.default_rng(1); Q = rng.uniform(-1.5, 1.5, (300, 2))
def ref1(a): return ocp_ref(a[0], a[1], a[2])
if __name__ == '__main__':
    rhos = [float(a) for a in sys.argv[1:]] or [.5, 2.]
    with Pool(30) as pool: R = {rho: np.array(pool.map(ref1, [(x, v, rho) for x, v in Q])) for rho in rhos}
    Cv = Cover(tau=.4, r=.1, seeds=6000); dt = Cv.dt0; out = []
    sets = {'bang': (-1.0, 1.0), 'tri': (-1.0, 0.0, 1.0), 'spec11': tuple(np.linspace(-1, 1, 11))}
    for rho in rhos:
        for nm, U in sets.items():
            solve_U(Cv, U, rho); J, T, sw, tv = rollout_U(Cv, Q, U, rho, dt); ok = np.isfinite(J) & np.isfinite(R[rho]) & (R[rho] > .05); r = J[ok] / R[rho][ok]
            d = dict(rho=rho, U=nm, n=int(ok.sum()), reach=round(float(np.isfinite(J).mean()), 3), ref_found=round(float(np.isfinite(R[rho]).mean()), 3),
                     J_ocp=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), min=round(float(r.min()), 4), max=round(float(r.max()), 3), share_below_1=round(float((r < .999).mean()), 3)),
                     sw_med=float(np.median(sw[ok])), tv_med=round(float(np.median(tv[ok])), 2))
            print(json.dumps(d), flush=True); out.append(d)
    json.dump(out, open('spec_vs_ocp.json', 'w'), indent=1); np.save('spec_vs_ocp_ref.npy', np.array([R[r] for r in rhos]))
