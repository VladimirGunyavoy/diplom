"""c9: причина стоянки в кармане. usage: diag_pocket.py "k:i,..." → diag_pocket.json (по шагам: ok[9], cl[9], Wq[9], Wq0[9] carfree, доля помеченных среди KNN у агента и у концов действий)"""
import os, sys, json
FL = [tuple(int(v) for v in p.split(':')) for p in sys.argv[1].split(',')]
ks = sorted({k for k, _ in FL}); sys.argv = ['x', '500000', '8', 'test', '0', '1', '2']
sys.path.insert(0, os.getcwd()); import numpy as np, car_dyn_c9 as C
D = C.D; S0 = np.load('starts100.npy')[:, :4]; COMP = json.load(open(os.environ['COMPACT'])); out = []
cap = {}; _v = C.vit_gpu
def vg(ag, FLg, nit, tol): cap['F'] = FLg.copy_to_host() if hasattr(FLg, 'copy_to_host') else np.asarray(FLg); return _v(ag, FLg, nit, tol)
C.vit_gpu = vg
for k in ks:
    S = C.mkS(k); back = C.build_back(S, C.TAU, C.NB, C.RB); ag = C.AgentDyn(S, back); H = D.Hazards('random', k); sc = S.scale
    for kk, i in FL:
        if kk != k: continue
        ref = {r['i']: r for r in COMP[str(k)]['ref']}[i]; x = np.array(S0[i], float); ag.prev = None; t = 0.; Pp = H.pos(0.); nsub = int(round(C.TAU / .002)); steps = []
        while t < 20.:
            P0 = H.pos(t); Wv = (P0 - Pp) / C.TAU if t > 0 else np.zeros((3, 2)); Pp = P0; s = ag.act(x, P0, Wv)
            F = cap['F']; KK = min(C.KN * 3, ag.nP)
            def knn(q):
                d, j = ag.tree.query(S.wrap(np.atleast_2d(q)) / sc, k=KK); j = j % ag.nP; return d, j
            d, j = knn(x); fx = float(F[j[0]].mean())
            d9, j9 = knn(ag_Q := np.array([C.cand(x[0], x[1], x[2], x[3], C.ACT[a, 0], C.ACT[a, 1], P0, Wv, C.KAPC, C.QC, C.SMAX, C.TAU, C.HB)[:4] for a in range(9)]))
            fq = F[j9].mean(1); Wq0 = np.min(ag.W0[j9] + C.KV * d9, axis=1)
            steps.append(dict(t=round(t, 2), x=[round(float(v), 3) for v in x], a=int(s), ok=ag.okc.astype(int).tolist(), cl=np.round(ag.cl, 3).tolist(), Wq=np.round(ag.Wq, 3).tolist(), Wq0=np.round(Wq0, 3).tolist(), fx=round(fx, 3), fq=np.round(fq, 2).tolist(), Wx=round(float(ag.Wx), 3), W0x=round(float(ag.W0x), 3), dn=round(float((np.linalg.norm(P0 - x[:2], axis=1) - C.HR).min()), 3)))
            PT = H.pos(t + (np.arange(nsub) + 1) * .002); xn, yn, thn, vn, m, hit = C.execute_dyn(x[0], x[1], x[2], x[3], C.ACT[s, 0], C.ACT[s, 1], .002, nsub, PT); x = np.array([xn, yn, thn, vn])
            if hit >= 0: break
            t += C.TAU
        out.append(dict(k=k, i=i, T_ref=ref['T'], steps=steps)); json.dump(out, open('diag_pocket.json', 'w')); print('done', k, i, flush=True)
