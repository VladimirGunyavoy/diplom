import sys, time; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('tests/run_faces_pend.py').read().split("# эталон")[0])
for d0 in (.08, .05, .03):
    ln = lines_graded(.02, 1.0, 400, d0=d0); t0=time.time(); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0); V = F.solve(); T,_ = F.rollout(Q)
    print('d0',d0,'lines',len(ln),'probes',len(F.P),'iters',F.iters,'finite V',round(float(np.isfinite(V).mean()),3),'reach',np.isfinite(T).mean(),'sec',round(time.time()-t0),flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fin = np.isfinite(V); plt.figure(figsize=(7,7)); plt.scatter(*F.P[~fin].T, s=1, c='r'); plt.scatter(*F.P[fin].T, s=1, c='g'); plt.savefig('/tmp/pend_inf.png', dpi=70)
