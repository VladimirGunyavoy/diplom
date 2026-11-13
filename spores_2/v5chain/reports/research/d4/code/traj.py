import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvg_lp as L
T = L.T; env = T.env; d = T.d
IDX = int(os.environ.get("IDX", 10))
ts = T.TS; S = T.mkS(ts)
rng_ = np.random.default_rng(1000 + ts); lo_ = np.where(env.period > 0, -env.period / 2, T.LO); hi_ = np.where(env.period > 0, env.period / 2, T.HI)
P_ = rng_.uniform(lo_, hi_, (2 * T.NB, d)); P_ = P_[T.okv(P_)][:T.NB]
back = (P_, np.zeros(len(P_)), None, None, None, T.cKDTree(T._scaled(S, T._images(S, P_))))
ag = L.AgentLP(S, back); ag.mode = "ip"
x0 = env.X0[IDX]
import common
_rec = {}; _ov = common.verify
def _v(env_, y0, us_, dts_, *a, **k): _rec["us"] = list(us_); _rec["dts"] = list(dts_); return _ov(env_, y0, us_, dts_, *a, **k)
common.verify = _v
Tv, why, m = T.episode(ag, x0); us = _rec["us"]; dts = _rec["dts"]; print("EPISODE", Tv, why, ag.info)
y = np.asarray(x0, float)[None].copy(); tt = 0.; P = [y[0].copy()]; TT = [0.]; exit_i = -1; h = .01
for u, dt in zip(us, dts):
    u = env.clip_u(np.asarray(u, float).reshape(1, -1)); k = max(1, int(np.ceil(dt / h - 1e-9))); hh = dt / k
    for _ in range(k):
        y = env.step(y, u, hh); tt += hh; P.append(y[0].copy()); TT.append(tt)
        if env.out_of_box(y)[0]: exit_i = len(P) - 1; break
    if exit_i >= 0: break
np.savez(os.path.expanduser("~/spore_v5/d4/out/traj_dp1_s4_i%d.npz" % IDX), P=np.array(P), t=np.array(TT), exit_i=exit_i, x0=x0, Tref=env.Tref[IDX], us=np.array(us), dts=np.array(dts))
print("exit", exit_i, P[-1], TT[-1])
