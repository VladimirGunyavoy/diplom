"""b26 (B, research-36 «В-шов»): СТАТИК-КАР с выпуклыми LP-весами в VI вместо «ближайший + KV·d».
Веса один раз: для образа ребра q — k=KLP ближайших узлов (шкала SC, θ периодический), min Σλ|x_j−q|², Σλx_j=q, Σλ=1, λ≥0 (lp_batch из reports/research/r36/tvg_lp.py); вне оболочки — ребра нет.
VI по разреженной матрице (vit_dl); агент: V̂(q)=Σλ W_lp (фолбэк на nn+KV при дыре). Запуск как car_val6: runpc.sh car_lp.py NB 8 sel|test T0 [T1]; LIM=0 — жадный по V̂ (БЕЗ TIE/TIEP/ESC), LIM=300 — A*; HW; KLP.
Печатает: доля рёбер-дыр, V̂(x0)/T_ref мед (nn и LP), build."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, numba as nb
import car_astar as ca
cv = ca.cv
from car_astar import *
KLP = int(os.environ.get('KLP', 20)); NITL = int(os.environ.get('NITDL', 3000))
from lp_core import lp_batch, vit_dl, vit_dx
HOLE = os.environ.get('HOLE', 'nn')

class AgentLP(ca.AAgent):
    def __init__(self, S, back):
        cv.Agent.__init__(self, S, back); self.Wnn = self.W.copy()
        t0 = time.perf_counter(); E, OK, GL = edges(self.P, TAU); n = self.nP
        Qs = S.wrap(E.reshape(-1, 4)) / S.scale; dd, jj = self.tree.query(Qs, k=KLP, workers=-1)
        J = np.empty((len(Qs), 5), np.int64); L = np.empty((len(Qs), 5)); ins = np.empty(len(Qs), bool); B = 200000
        for a in range(0, len(Qs), B):
            Nb = self.tree.data[jj[a:a + B]]; Jl, Ll, okl = lp_batch(np.ascontiguousarray(Nb), np.ascontiguousarray(Qs[a:a + B]))
            J[a:a + B] = np.take_along_axis(jj[a:a + B], Jl, 1) % n; L[a:a + B] = Ll; ins[a:a + B] = okl
        ins = ins.reshape(-1, 9); OKd = (OK & ins) | (OK & GL); t1 = time.perf_counter()
        self.holes = (OK & ~ins & ~GL).sum() / max(1, (OK & ~GL).sum())
        tauv = np.full(9, TAU); GLf = np.where(GL, TAU, 0.)
        if HOLE == 'nn':   # дыры: ближайший узел + KV·d (как раньше), остальные рёбра — LP
            hole = (~ins.reshape(-1)) & ~GL.reshape(-1) & OK.reshape(-1); J[hole, 0] = jj[hole, 0] % n; L[hole] = 0.; L[hole, 0] = 1.
            EXT = np.where(hole, cv.KV * dd[:, 0], 0.); OKd = OK.copy()
            self.Wlp, it, res = vit_dx(J, L, EXT, OKd, GLf, in_goal(self.P), n, tauv, NITL, 1e3)
        else: self.Wlp, it, res = vit_dl(J, L, OKd, GLf, in_goal(self.P), n, tauv, NITL, 1e3)
        self.tlp = time.perf_counter() - t0
        fin = self.Wlp < 1e2; print('  LP: веса+VI %.1f с (веса %.1f), дыр-рёбер %.3f%%, it %d res %.1e, W_lp конечных %.1f%% мед %.2f макс %.2f (nn: мед %.2f)' % (self.tlp, t1 - t0, 100 * self.holes, it, res, 100 * fin.mean(), np.median(self.Wlp[fin]), self.Wlp[fin].max(), np.median(self.Wnn[self.Wnn < 1e2])), flush=True)
        self.W = self.Wlp; self.mode = 'lp'
    def vlp(self, Q):
        Qs = self.S.wrap(np.asarray(Q, float)) / self.S.scale; dd, jj = self.tree.query(Qs, k=KLP)
        Jl, L, ok = lp_batch(np.ascontiguousarray(self.tree.data[jj]), np.ascontiguousarray(Qs)); J = np.take_along_axis(jj, Jl, 1) % self.nP
        w = (L * self.W[J]).sum(1); bad = ~ok | (self.W[J] >= 1e2).any(1)
        dm = dd[:, 0]; a = np.argmin(self.W[jj % self.nP] + KVA * dd, axis=1)
        wn = (self.W[jj % self.nP] + KVA * dd)[np.arange(len(Qs)), a]
        return np.where(bad, wn, w), dm
    def h(self, Q):
        if self.mode == 'lp': return self.vlp(Q)
        return ca.AAgent.h(self, Q)

if __name__ == '__main__':
    st = get_starts(); out = {}
    from tqdm import tqdm
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0
        ag = AgentLP(S, back); X0 = np.array([s[1] for s in st], float); tr = np.array([s[2] for s in st])
        ag.mode = 'nn'; vn = ag.W; ag.W = ag.Wnn; hn = ca.AAgent.h(ag, X0)[0] if False else None
        d_, j_ = ag.tree.query(S.wrap(X0) / S.scale, k=min(KN * 3, ag.nP)); vnn = np.min(ag.Wnn[j_ % ag.nP] + cv.KV * d_, axis=1)
        ag.W = ag.Wlp; ag.mode = 'lp'; vlp_, _ = ag.vlp(X0)
        print('ts %d: build %.1f с; V̂(x0)/T_ref мед: nn+KV %.3f, LP %.3f (p10–p90 LP %.3f–%.3f)' % (ts, tb, np.median(vnn / tr), np.median(vlp_ / tr), np.percentile(vlp_ / tr, 10), np.percentile(vlp_ / tr, 90)), flush=True)
        res = []
        for i, x0, tref in tqdm(st, desc='ts%d' % ts, mininterval=10):
            te = time.perf_counter(); T, mn, ms = episode(ag, x0); res.append((i, T, tref, mn, ms, time.perf_counter() - te))
        ok = [r for r in res if r[1] is not None]; ratio = np.array([r[1] / r[2] for r in ok]); touch = sum(r[3] < 0 for r in res)
        print('LP LIM=%d HW=%g KLP=%d NB=%d [%s] ts=%d: дошли %d/%d, касаний %d, T/T_ref мед %.3f p90 %.3f max %.3f среднее %.3f, мс/шаг %.1f, зазор мин %.3f, build %.1f с, LP %.1f с, дыр %.3f%%' % (
            LIM, HW, KLP, NB, MODE, ts, len(ok), len(res), touch, np.median(ratio), np.percentile(ratio, 90), ratio.max(), ratio.mean(), 1e3 * np.mean([r[4] for r in res]), min(r[3] for r in res), tb, ag.tlp, 100 * ag.holes), flush=True)
        out[ts] = dict(res=res)
    json.dump(out, open('lp_%s_%d_%d_%g_%d.json' % (MODE, NB, LIM, HW, TS), 'w'))
