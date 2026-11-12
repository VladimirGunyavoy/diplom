"""b26 (B2, research-37): статик-кар, LP-шов + VCAP + переменный τ + узлы дерева/случайные. Запуск: runpc.sh car_lp2.py NB 8 sel T0 [T1]
ENV: RNDN=1 — случайные равномерные узлы вне дисков (иначе дерево); TAUS='.05,.1,.2' — τ рёбер VI (по умолч. .2); VCAP=40; KLP=20; HOLE=none|nn; LIM=0 — жадный по V̂, LIM=300 — A* (HW); агент шаг .2 (TAU)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, numba as nb
import car_astar as ca
cv = ca.cv
from car_astar import *
from lp_core import lp_batch, vit_cap
from scipy.spatial import cKDTree
from src.atlas6.adaptive_nd import _images, _scaled
KLP = int(os.environ.get('KLP', 20)); NITL = int(os.environ.get('NITDL', 3000)); VCAP = float(os.environ.get('VCAP', 40)); HOLE = os.environ.get('HOLE', 'none')
TAUS = np.array([float(v) for v in os.environ.get('TAUS', '.2').split(',')]); RNDN = int(os.environ.get('RNDN', 0)); NNMODE = int(os.environ.get('NNMODE', 0))

@nb.njit(parallel=True)
def edges_v(P, taus):
    n = P.shape[0]; nt = len(taus); L = 9 * nt; E = np.zeros((n, L, 4)); OK = np.zeros((n, L), np.bool_); GT = np.zeros((n, L))
    for i in nb.prange(n):
        for s in range(9):
            for q in range(nt):
                x, y, th, v = P[i, 0], P[i, 1], P[i, 2], P[i, 3]; ok = True; h = taus[q] / 10
                for k in range(10):
                    x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1)
                    if clear(x, y) < MARG or v < VL or v > VH: ok = False
                    if (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: GT[i, s * nt + q] = (k + 1) * h; break
                a = s * nt + q; E[i, a, 0] = x; E[i, a, 1] = y; E[i, a, 2] = th; E[i, a, 3] = v; OK[i, a] = ok
    return E, OK, GT

class AgentLP2(ca.AAgent):
    def __init__(self, S, back):
        self.S, self.tree, self.P = S, back[5], np.asarray(back[0]); self.nP = n = len(self.P)
        t0 = time.perf_counter(); E, OK, GT = edges_v(self.P, TAUS); L2 = OK.shape[1]; nt = len(TAUS)
        Qs = S.wrap(E.reshape(-1, 4)) / S.scale; dd, jj = self.tree.query(Qs, k=KLP, workers=-1)
        J = np.empty((len(Qs), 5), np.int64); L = np.empty((len(Qs), 5)); ins = np.empty(len(Qs), bool); B = 200000
        for a in range(0, len(Qs), B):
            Nb = self.tree.data[jj[a:a + B]]; Jl, Ll, okl = lp_batch(np.ascontiguousarray(Nb), np.ascontiguousarray(Qs[a:a + B]))
            J[a:a + B] = np.take_along_axis(jj[a:a + B], Jl, 1) % n; L[a:a + B] = Ll; ins[a:a + B] = okl
        ins = ins.reshape(-1, L2); goal = GT > 0; t1 = time.perf_counter()
        self.holes = (OK & ~ins & ~goal).sum() / max(1, (OK & ~goal).sum())
        hole = (~ins.reshape(-1)) & ~goal.reshape(-1) & OK.reshape(-1); EXT = np.zeros(len(Qs))
        if HOLE == 'nn': J[hole, 0] = jj[hole, 0] % n; L[hole] = 0.; L[hole, 0] = 1.; EXT = np.where(hole, cv.KV * dd[:, 0], 0.); OKd = OK
        else: OKd = (OK & ins) | (OK & goal)
        tauv = np.repeat(TAUS, 1)[np.tile(np.arange(nt), 9)]
        self.W, it, res = vit_cap(J, L, EXT, OKd, GT, in_goal(self.P), n, tauv, NITL, VCAP); self.tlp = time.perf_counter() - t0
        fin = self.W < VCAP - 1e-6; print('  LP+VCAP: веса+VI %.1f с (веса %.1f), узлов %d, дыр-рёбер %.3f%%, it %d res %.1e, W<VCAP %.1f%% мед %.2f макс %.2f' % (self.tlp, t1 - t0, n, 100 * self.holes, it, res, 100 * fin.mean(), np.median(self.W[fin]), self.W[fin].max()), flush=True)
        self.mode = 'lp'
    vlp = None
    def h(self, Q):
        Qs = self.S.wrap(np.asarray(Q, float)) / self.S.scale; dd, jj = self.tree.query(Qs, k=KLP)
        Jl, Lm, ok = lp_batch(np.ascontiguousarray(self.tree.data[jj]), np.ascontiguousarray(Qs)); J = np.take_along_axis(jj, Jl, 1) % self.nP
        w = (Lm * np.minimum(self.W[J], VCAP)).sum(1); c = self.W[jj % self.nP] + KVA * dd; a = np.argmin(c, axis=1); r = np.arange(len(Qs))
        return np.where(ok, w, c[r, a]), dd[:, 0]
    def vlp_(self, Q): return self.h(Q)[0]

def make_back(S, ts):
    if not RNDN: return build_back(S, TAU, NB, RB)
    rng = np.random.default_rng(1000 + ts); P = np.c_[rng.uniform(-XL, XL, (2 * NB, 2)), rng.uniform(-np.pi, np.pi, 2 * NB), rng.uniform(VL, VH, 2 * NB)]
    P = P[~blocked(P)][:NB]; return (P, np.zeros(len(P)), None, None, None, cKDTree(_scaled(S, _images(S, P))))

if __name__ == '__main__':
    st = get_starts(); out = {}
    from tqdm import tqdm
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = make_back(S, ts); tb = time.perf_counter() - t0
        ag = (ca.AAgent(S, back) if NNMODE else AgentLP2(S, back)); ag.tlp = 0. if NNMODE else ag.tlp; ag.holes = 0. if NNMODE else ag.holes; X0 = np.array([s[1] for s in st], float); tr = np.array([s[2] for s in st]); v0 = ag.h(X0)[0]
        print('ts %d: build %.1f с; V̂(x0)/T_ref мед LP %.3f (p10–p90 %.3f–%.3f)' % (ts, tb, np.median(v0 / tr), np.percentile(v0 / tr, 10), np.percentile(v0 / tr, 90)), flush=True)
        res = []
        for i, x0, tref in tqdm(st, desc='ts%d' % ts, mininterval=10):
            te = time.perf_counter(); T, mn, ms = episode(ag, x0); res.append((i, T, tref, mn, ms, time.perf_counter() - te))
        ok = [r for r in res if r[1] is not None]; ratio = np.array([r[1] / r[2] for r in ok]); touch = sum(r[3] < 0 for r in res)
        print('LP2 RNDN=%d TAUS=%s LIM=%d HW=%g KLP=%d VCAP=%g HOLE=%s NB=%d [%s] ts=%d: дошли %d/%d, касаний %d, T/T_ref мед %.3f p90 %.3f max %.3f, мс/шаг %.1f, зазор мин %.3f, build %.1f с, LP %.1f с, дыр %.3f%%' % (
            RNDN, os.environ.get('TAUS', '.2'), LIM, HW, KLP, VCAP, HOLE, NB, MODE, ts, len(ok), len(res), touch, np.median(ratio), np.percentile(ratio, 90), ratio.max(), 1e3 * np.mean([r[4] for r in res]), min(r[3] for r in res), tb, ag.tlp, 100 * ag.holes), flush=True)
        out[ts] = dict(res=res)
    json.dump(out, open('lp2_%s_%d_%d_%d_%s_%d.json' % (MODE, NB, RNDN, LIM, os.environ.get('TAUS', '.2').replace(',', '-'), TS), 'w'))
