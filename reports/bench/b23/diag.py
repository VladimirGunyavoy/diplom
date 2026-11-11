"""b23: трасса стоячего эпизода. runpc.sh diag.py NB tree seed"""
import os, sys
os.environ.setdefault('FM','.1'); os.environ.setdefault('HS','.5')
nb_, ts, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); sys.argv = ['x', nb_, '0.3', '8', 'sel', ts]
import numpy as np, canal_val as V
from canal_val import C, CT
env = C.mkenv(); S = CT.mkS(int(ts)); back = V.build_back(S, C.TAU, int(nb_), .3); ag = V.AgentV(S, back, 8)
t = env.unwrapped.task; env.reset(seed=seed)
for step in range(300):
    d = t.data; p = t.agent.pos; R_ = t.agent.mat; th = np.arctan2(R_[1, 0], R_[0, 0]); th0 = th - d.qpos[2]; c0, s0 = np.cos(th0), np.sin(th0)
    vx = c0 * d.qvel[0] - s0 * d.qvel[1]; vy = s0 * d.qvel[0] + c0 * d.qvel[1]
    a = ag.act(p[0], p[1], th, vx, vy)
    if step % 15 == 0 or step < 4:
        rho, al, vr, vt = C.to_ring(p[0], p[1], th, vx, vy); q = S.wrap(np.array([[rho, al, vr, vt]]))
        dd, j = ag.tree.query(q / S.scale, k=1)
        print(step, 'rho %.2f al %.2f vr %.2f vt %.2f a=%s dNN %.2f V~%.2f' % (rho, al, vr, vt, C.ACT[a], dd[0], ag.V[j[0] % ag.nP]), flush=True)
    env.step(C.ACT[a])
