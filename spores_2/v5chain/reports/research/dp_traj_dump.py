"""research-11: траектория агента FORCEPLAN против OCP (только оценка): махи (смены знака w1), энергия, складывание руки |q2|. Запуск: G=2 WIN=1.0 FORCEPLAN=1 WCHK=1 python3 dp_traj_dump.py atlas.npz"""
import numpy as np, sys, os, json
sys.path.insert(0, '.')
from butterfly_dp import flow, wrap, G, MS, S11, S12, S22
import butterfly_dp_query as Q, butterfly_dp_ellipse as E
def energy(z):
    q1, q2, w1, w2 = z.T; th1, th2 = q1, q1 + q2; d1, d2 = w1, w1 + w2
    return .5 * S11 * d1 ** 2 + .5 * S22 * d2 ** 2 + S12 * np.cos(th1 - th2) * d1 * d2 + G * (MS[0] * np.sin(th1) + MS[1] * np.sin(th2))
def summ(name, P, dt):
    w1 = P[:, 2]; sg = np.sign(w1[np.abs(w1) > .15]); sw = int((sg[1:] != sg[:-1]).sum()); En = energy(P); tt = np.arange(len(P)) * dt
    print(json.dumps(dict(path=name, T=round(float(tt[-1]), 2), swings_w1=sw, fold_frac_absq2_gt2=round(float((np.abs(wrap(P[:, 1])) > 2).mean()), 2), wmax=round(float(np.abs(P[:, 2:]).max()), 2),
                          t_E_gt_0=round(float(tt[np.argmax(En > 0)]), 2), E_rate_mean=round(float((En[-1] - En[0]) / tt[-1]), 2), E_at_quarters=np.round(En[[len(P) // 4, len(P) // 2, 3 * len(P) // 4]], 1).tolist())))
a = np.load('refdp_G2_W3_N80.npy'); T = a[0]; u = a[1:].reshape(80, 2); z = np.array([-np.pi / 2, 0, 0, 0.]); P = [z]
for k in range(80):
    for _ in range(10): z = flow(z[:, None], u[k:k + 1, 0], u[k:k + 1, 1], T / 800, n=1)[:, 0]; P.append(z)
summ('OCP', np.array(P), T / 800)
A = Q.load(sys.argv[1]); E.refresh(A); Q.TRAJ.clear(); A.E = Q.node_edges(A); Tm, arcs, wm = Q.rollout_edges(A, 81, 0.); P = []; us = []
for y, u1, u2, tt in Q.TRAJ:
    m = max(1, int(round(tt / .01))); z = y.copy()
    for _ in range(m): P.append(z.copy()); z = flow(z[:, None], np.array([u1]), np.array([u2]), tt / m, n=1)[:, 0]
    us.append((round(tt, 2), round(u1, 2), round(u2, 2)))
print(json.dumps(dict(T=round(float(Tm), 3), arcs=arcs, plan_arcs=len(Q.TRAJ)))); summ('agent(plan arcs only)', np.array(P), .01); print('arcs (t,u1,u2):', us)
np.save(sys.argv[1].replace('.npz', '_traj.npy'), np.array(P))
