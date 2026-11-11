"""b22 КАНАВА на НАШЕМ графе: обратное дерево v6 (adaptive_nd.build_back) на 4D кольце (ρ, α, v_рад, v_кас), 5D-модель numba, множество дна M (R*=.8, v*=1.2, MARG=.03, λ=1 — не переподбирались);
V̂(q) = min_{K ближайших узлов b} (G_b + κ·|q−b|_scaled); агент — жадный по V̂ с прогнозом .1 с и бонусом λ·награда. Исполнение — SafetyPointCircle2 (их обёртка), 500 шагов.
Запуск: runpc.sh canal_tree.py NB KAP K mode(sel|eval|r20) [TREESEED] ; ENV: MARG LAM TOLR TOLV TOLRAD TOLA"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MARG', '.03')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
import numpy as np, numba as nb
import canal as C
from src.atlas6.adaptive_nd import SysN, build_back
NB, KAP, KN = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); MODE = sys.argv[4]; TS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
R, VS = .8, 1.2; TR, TV, TRAD, TA = [float(os.environ.get(k, d)) for k, d in (('TOLR', .1), ('TOLV', .2), ('TOLRAD', .2), ('TOLA', .6))]; LAM = float(os.environ.get('LAM', 1.)); RMAXW = C.RMAXW; VMX = C.VMX
@nb.njit
def _rf(rho, al, vr, vt, u0, u1, t):
    n = max(1, int(np.ceil(abs(t) / .02))); x, y, th, vx, vy = C.f5(rho, 0., al, vr, vt, u0, u1, t / n, n); return C.to_ring(x, y, th, vx, vy)
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = C.ACT[s]
    if P.ndim == 1: return np.array(_rf(P[0], P[1], P[2], P[3], u[0], u[1], float(t)))
    return np.array([_rf(p[0], p[1], p[2], p[3], u[0], u[1], float(t)) for p in P.reshape(-1, 4)]).reshape(P.shape)
def in_M(P):
    P = np.asarray(P); da = np.abs((P[..., 1] - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)
    return (np.abs(P[..., 0] - R) <= TR) & (np.abs(P[..., 3] - VS) <= TV) & (np.abs(P[..., 2]) <= TRAD) & (da <= TA)
def mkS(seed):
    rng = np.random.default_rng(seed); seeds = []
    while len(seeds) < 300:
        p = np.array([R + rng.uniform(-TR, TR), np.pi / 2 + rng.uniform(-TA, TA), rng.uniform(-TRAD, TRAD), VS + rng.uniform(-TV, TV)])
        if in_M(p) and p[0] <= RMAXW: seeds.append(p)
    return SysN(flow, 9, (.15, .6, .2, .2), in_M, seeds, per=(0, 2 * np.pi, 0, 0), ok=lambda p: abs(p[2]) <= VMX and abs(p[3]) <= VMX, blocked=lambda P: np.asarray(P)[..., 0] > RMAXW)
@nb.njit
def cands(x, y, th, vx, vy, ACTS):
    Q = np.zeros((9, 4)); RW = np.zeros(9)
    for a in range(9):
        xn, yn, thn, vxn, vyn = C.f5(x, y, th, vx, vy, ACTS[a, 0], ACTS[a, 1], C.H, C.NSUB); rho, al, vr, vt = C.to_ring(xn, yn, thn, vxn, vyn)
        Q[a, 0] = rho; Q[a, 1] = al; Q[a, 2] = vr; Q[a, 3] = vt; RW[a] = .1 * vt / (1. + abs(rho - 1.5)) * C.NSUB
    return Q, RW
class Agent:
    def __init__(self, S, back): self.S, self.G, self.tree, self.nP = S, np.asarray(back[1]), back[5], len(back[0])
    def act(self, x, y, th, vx, vy):
        Q, RW = cands(x, y, th, vx, vy, C.ACT); ok = Q[:, 0] <= RMAXW; Qw = self.S.wrap(Q)
        k = min(KN * 3, len(self.tree.data)); d, j = self.tree.query(Qw / self.S.scale, k=k); j = j % self.nP
        V = np.min(self.G[j] + KAP * d, axis=1); sc = np.where(ok, C.TAU + V - LAM * RW, 1e9); return int(np.argmin(sc))
def episode(env, ag, seed, T=500):
    t = env.unwrapped.task; env.reset(seed=seed); ret = 0.; cost = 0; term = False; tms = []; mn = 9.
    for step in range(T):
        d = t.data; p = t.agent.pos; R_ = t.agent.mat; th = np.arctan2(R_[1, 0], R_[0, 0]); th0 = th - d.qpos[2]; c0, s0 = np.cos(th0), np.sin(th0)
        vx = c0 * d.qvel[0] - s0 * d.qvel[1]; vy = s0 * d.qvel[0] + c0 * d.qvel[1]; t1 = time.perf_counter()
        a = ag.act(p[0], p[1], th, vx, vy); tms.append(time.perf_counter() - t1); _, r, c, te, tr, _ = env.step(C.ACT[a]); p = t.agent.pos
        mn = min(mn, C.WALL - max(abs(p[0]), abs(p[1])))
        if max(abs(p[0]), abs(p[1])) > C.WALL: ret += -1.; cost += 1; term = True; break
        ret += float(r)
    return ret, cost, term, float(np.mean(tms)), mn
if __name__ == '__main__':
    env = C.mkenv(); out = {}
    seeds_by = dict(sel=range(100000, 100010), eval=range(200000, 200020), r20=range(200000, 200100))
    for ts in ([TS] if MODE != 'r20' else range(int(os.environ.get('T0', 0)), int(os.environ.get('T1', 20)))):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, C.TAU, NB, float(os.environ.get('RB', .15))); tb = time.perf_counter() - t0; ag = Agent(S, back)
        rs = [episode(env, ag, s) for s in seeds_by[MODE]]; ret = np.array([r[0] for r in rs])
        print('NB=%d(узлов %d) κ=%.2f K=%d ts=%d [%s]: return %.2f±%.2f cost %d обрывов %d t_build %.1f с мс/шаг %.2f Gmax %.2f мин.зазор до стены (мед эп.) %.3f мин %.3f' % (NB, len(back[0]), KAP, KN, ts, MODE, ret.mean(), ret.std(), sum(r[1] for r in rs), sum(r[2] for r in rs), tb, 1e3 * np.mean([r[3] for r in rs]), max(back[1]), np.mean([r[4] for r in rs]), min(r[4] for r in rs)), flush=True)
        out[ts] = dict(ret=ret.tolist(), cost=sum(r[1] for r in rs), term=sum(r[2] for r in rs), t_build=tb, nodes=len(back[0]), ms=1e3 * float(np.mean([r[3] for r in rs])), margin=[r[4] for r in rs])
    json.dump(out, open('canal_tree_%s_%d_%g_%d_%s.json' % (MODE, NB, KAP, KN, os.environ.get('T0', '')), 'w'))
