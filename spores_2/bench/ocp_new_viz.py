"""c7: OCP-эталон (export_ref.solve) только на новых стартах → список эпизодов в json. python ocp_new.py car|carfree ID,ID out.json"""
import sys, os, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/c1/vz/viz')); sys.path.insert(0, os.path.expanduser('~/spore_v5/c1/vz/bench'))
import export_ref as XR, export_mobile as EM
from multiprocessing import Pool
if __name__ == '__main__':
    sysn, ids, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]; e = XR.make(sysn)
    with Pool(len(ids)) as p: R = p.map(XR.solve, [(sysn, i, 70) for i in ids])
    eps = []
    for _, i, best in R:
        if best is None: print(sysn, i, 'OCP не сошёлся', flush=True); continue
        Tc, Uv, Tv = best; segs = [(Uv[:, k], Tv / Uv.shape[1]) for k in range(Uv.shape[1])]
        ep = EM.episode(e, i, 'эталон (OCP)', segs, float(e.Tref[i]) if sysn == 'car' else float('nan'), 'нет в бенче (OCP CasADi этой выгрузки; эталон бенча car_ref20 — только T)' if sysn == 'car' else 'OCP CasADi этой выгрузки'); eps.append(ep); print(sysn, i, 'T', ep['T'], flush=True)
    json.dump(eps, open(out, 'w'))
