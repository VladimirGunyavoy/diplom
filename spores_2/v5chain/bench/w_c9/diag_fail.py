"""c9: разбор недошедших дерево+V TIGHT. usage: diag_fail.py "k:i,k:i,..." ; пишет json diag_fail.json. Те же параметры/сид дерева=диски k."""
import os, sys, json, time
FL = [tuple(int(v) for v in p.split(':')) for p in sys.argv[1].split(',')]
ks = sorted({k for k, _ in FL}); sys.argv = ['x', '500000', '8', 'test', '0', '1', '2']
sys.path.insert(0, os.getcwd()); import numpy as np, car_dyn_c9 as C
D = C.D; S0 = np.load('starts100.npy')[:, :4]; COMP = json.load(open(os.environ['COMPACT'])); out = []
for k in ks:
    S = C.mkS(k); back = C.build_back(S, C.TAU, C.NB, C.RB); ag = C.AgentDyn(S, back); H = D.Hazards('random', k)
    for kk, i in FL:
        if kk != k: continue
        ref = {r['i']: r for r in COMP[str(k)]['ref']}[i]; x = np.array(S0[i], float); ag.prev = None; t = 0.; Pp = H.pos(0.); nsub = int(round(C.TAU / .002)); tr = []; W0s = None; nt = 0
        while t < 60.:
            P0 = H.pos(t); Wv = (P0 - Pp) / C.TAU if t > 0 else np.zeros((3, 2)); Pp = P0; s = ag.act(x, P0, Wv)
            if W0s is None: W0s = ag.W0x
            dn = float((np.linalg.norm(P0 - x[:2], axis=1) - C.HR).min()); tr.append((t, *x, dn, ag.W0x, ag.Wx, s)); nt = getattr(ag, 'ntie', 0)
            PT = H.pos(t + (np.arange(nsub) + 1) * .002); xn, yn, thn, vn, m, hit = C.execute_dyn(x[0], x[1], x[2], x[3], C.ACT[s, 0], C.ACT[s, 1], .002, nsub, PT); x = np.array([xn, yn, thn, vn])
            if hit >= 0: break
            t += C.TAU
        a = np.array(tr); late = a[a[:, 0] >= 30]
        r = dict(k=k, i=i, T_ref=ref['T'], clr0=ref['clr0'], W0x0=W0s, dnear_min=float(a[:, 5].min()), dnear_med=float(np.median(a[:, 5])), dnear_end=float(a[-1, 5]), reached=hit >= 0,
                 vabs_med_late=float(np.median(np.abs(late[:, 4]))), frac_rest=float((np.abs(a[:, 4]) < .15).mean()), path_late=float(np.hypot(np.diff(late[:, 1]), np.diff(late[:, 2])).sum()), end_xy=[float(a[-1, 1]), float(a[-1, 2])], dgoal_end=float(np.hypot(a[-1, 1] - C.GOAL[0], a[-1, 2] - C.GOAL[1])),
                 W0x_end=float(a[-1, 6]), nact_late=len(set(late[:, 8].astype(int))), tie_total=nt)
        print(r, flush=True); out.append(r); json.dump(out, open('diag_fail.json', 'w'))
