"""b22 КАНАВА-квадрат: обратное дерево v6 в ПОЛНОМ 5D (x, y, θ, vx, vy) без симметрии; стены — настоящий квадрат |x|,|y| ≤ 1.125−MT; дно M — скруглённый квадрат (полуширина h=1.125−D, радиус скругления RC),
v_кас ≈ V* против часовой, v_норм ≈ 0, курс ≈ по касательной. V̂ = min по K ближайшим узлам (G + κ·dist); агент жадный по V̂ + λ·награда + фильтр безопасности v2 (FM, HS) — как canal_tree.py.
Запуск: runpc.sh canal_tree5.py NB KAP K mode(selT|eval|r20) ; ENV: D RC VS RB FM HS T0 T1"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'v6'))
import numpy as np, numba as nb
import canal as C
from src.atlas6.adaptive_nd import SysN, build_back
NB, KAP, KN = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); MODE = sys.argv[4]
E = lambda k, d: float(os.environ.get(k, d))
D, RC, VS, MT, FM, LAM = E('D', .25), E('RC', .4), E('VS', 1.2), E('MT', .05), E('FM', .1), E('LAM', 1.); NHZ = int(round(E('HS', .5) / .02)); HB = C.WALL - D; WLIM = C.WALL - MT; VMX = C.VMX
TR, TV, TN, TA = .1, .2, .2, .6; SYM = int(os.environ.get('SYM', 1))
@nb.njit
def geom(x, y, h, rc):   # signed distance до скруглённого квадрата (>0 снаружи) и касательная против часовой
    b = h - rc; qx = abs(x) - b; qy = abs(y) - b; sx = 1. if x >= 0 else -1.; sy = 1. if y >= 0 else -1.
    mx = max(qx, 0.); my = max(qy, 0.); nn = np.hypot(mx, my)
    sd = nn + min(max(qx, qy), 0.) - rc
    if nn > 0: nx = sx * mx / nn; ny = sy * my / nn
    elif qx > qy: nx = sx; ny = 0.
    else: nx = 0.; ny = sy
    return sd, -ny, nx
@nb.njit
def canon(x, y, th, vx, vy):   # симметрия C4 квадрата: поворот на −kπ/2 в сектор φ ∈ [−π/4, π/4)
    k = int(np.floor(np.arctan2(y, x) / (np.pi / 2) + .5)) % 4
    for _ in range(k): x, y, vx, vy, th = y, -x, vy, -vx, th - np.pi / 2
    return x, y, (th + np.pi) % (2 * np.pi) - np.pi, vx, vy
@nb.njit
def _fl(x, y, th, vx, vy, u0, u1, t):
    n = max(1, int(np.ceil(abs(t) / .02))); a, b, c, d, e = C.f5(x, y, th, vx, vy, u0, u1, t / n, n)
    if SYM: return canon(a, b, c, d, e)
    return a, b, c, d, e
def flow(P, s, t, dt_max=None):
    P = np.asarray(P, float); u = C.ACT[s]
    if P.ndim == 1: return np.array(_fl(P[0], P[1], P[2], P[3], P[4], u[0], u[1], float(t)))
    return np.array([_fl(p[0], p[1], p[2], p[3], p[4], u[0], u[1], float(t)) for p in P.reshape(-1, 5)]).reshape(P.shape)
def in_M(p):
    sd, tx, ty = geom(p[0], p[1], HB, RC); vt = p[3] * tx + p[4] * ty; vn = -p[3] * ty + p[4] * tx
    dth = abs((p[2] - np.arctan2(ty, tx) + np.pi) % (2 * np.pi) - np.pi)
    return abs(sd) <= TR and abs(vt - VS) <= TV and abs(vn) <= TN and dth <= TA
def mkS(seed):
    rng = np.random.default_rng(seed); seeds = []
    while len(seeds) < 1000:
        p = np.array([rng.uniform(-HB - .2, HB + .2), rng.uniform(-HB - .2, HB + .2), rng.uniform(-np.pi, np.pi), rng.uniform(-VMX, VMX), rng.uniform(-VMX, VMX)])
        if in_M(p) and max(abs(p[0]), abs(p[1])) <= WLIM: seeds.append(np.array(canon(*p)) if SYM else p)
    return SysN(flow, 9, (.15, .15, .6, .2, .2), lambda P: in_M(P), seeds, per=(0, 0, 2 * np.pi, 0, 0), ok=lambda p: abs(p[3]) <= VMX and abs(p[4]) <= VMX, blocked=lambda P: np.maximum(np.abs(np.asarray(P)[..., 0]), np.abs(np.asarray(P)[..., 1])) > WLIM)
@nb.njit
def cands(x, y, th, vx, vy, ACTS):
    Q = np.zeros((9, 5)); RW = np.zeros(9); CL = np.zeros(9)
    for a in range(9):
        a1, b1, c1, d1, e1 = C.f5(x, y, th, vx, vy, ACTS[a, 0], ACTS[a, 1], C.H, C.NSUB); rho = np.hypot(a1, b1); vt = (-b1 * d1 + a1 * e1) / max(rho, 1e-9)
        qa = canon(a1, b1, c1, d1, e1) if SYM else (a1, b1, c1, d1, e1); Q[a, 0] = qa[0]; Q[a, 1] = qa[1]; Q[a, 2] = qa[2]; Q[a, 3] = qa[3]; Q[a, 4] = qa[4]; RW[a] = .1 * vt / (1. + abs(rho - 1.5)) * C.NSUB
        px, py, pth, pvx, pvy = x, y, th, vx, vy; mn = 9.
        for _ in range(NHZ):
            px, py, pth, pvx, pvy = C.f5(px, py, pth, pvx, pvy, ACTS[a, 0], ACTS[a, 1], .02, 1); mn = min(mn, C.WALL - max(abs(px), abs(py)))
        CL[a] = mn
    return Q, RW, CL
class Agent:
    def __init__(self, S, back): self.S, self.G, self.tree, self.nP = S, np.asarray(back[1]), back[5], len(back[0])
    def act(self, x, y, th, vx, vy):
        Q, RW, CL = cands(x, y, th, vx, vy, C.ACT); ok = (np.maximum(np.abs(Q[:, 0]), np.abs(Q[:, 1])) <= WLIM) & (CL >= FM); Qw = self.S.wrap(Q)
        k = min(KN * 3, len(self.tree.data)); d, j = self.tree.query(Qw / self.S.scale, k=k); j = j % self.nP
        V = np.min(self.G[j] + KAP * d, axis=1)
        if not ok.any(): ok = CL >= FM
        sc = np.where(ok, C.TAU + V - LAM * RW, 1e9)
        if not ok.any(): return int(np.argmax(CL))
        return int(np.argmin(sc))
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
    env = C.mkenv(); out = {}; seeds_by = dict(selT=range(100000, 100010), eval=range(200000, 200020), r20=range(200000, 200100))
    for ts in (range(int(os.environ.get('T0', 20)), int(os.environ.get('T1', 30))) if MODE != 'eval' else [0]):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, C.TAU, NB, E('RB', .3)); tb = time.perf_counter() - t0; ag = Agent(S, back)
        rs = [episode(env, ag, s) for s in seeds_by[MODE]]; ret = np.array([r[0] for r in rs])
        print('D=%.2f RC=%.2f V*=%.2f NB=%d(узлов %d) κ=%.2f K=%d ts=%d [%s]: return %.2f±%.2f cost %d обрывов %d t_build %.1f с мс/шаг %.2f Gmax %.2f мин.зазор до стены (мед эп.) %.3f мин %.3f' % (D, RC, VS, NB, len(back[0]), KAP, KN, ts, MODE, ret.mean(), ret.std(), sum(r[1] for r in rs), sum(r[2] for r in rs), tb, 1e3 * np.mean([r[3] for r in rs]), max(back[1]), np.mean([r[4] for r in rs]), min(r[4] for r in rs)), flush=True)
        out[ts] = dict(ret=ret.tolist(), cost=sum(r[1] for r in rs), term=sum(r[2] for r in rs), t_build=tb, nodes=len(back[0]), ms=1e3 * float(np.mean([r[3] for r in rs])), margin=[r[4] for r in rs])
    json.dump(out, open('canal_tree5_%s_%s_%s_%s_%s.json' % (MODE, os.environ.get('D', ''), os.environ.get('RC', ''), os.environ.get('VS', ''), os.environ.get('T0', '')), 'w'))
