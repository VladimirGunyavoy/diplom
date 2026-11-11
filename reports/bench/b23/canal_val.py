"""b23 КАНАВА, вариант «V = дисконтированная награда»: то же 4D кольцевое обратное дерево v6 (canal_tree.mkS/build_back, M для узлов), но значения —
V(p)=max_a [ R(p,a) + γ^5 · V̂(flow(p,a,.1)) ], R — сумма награды .1·v_кас/(1+|ρ-1.5|) по 5 подшагам .02 (итерация ценностей, Дейкстра не годится);
V̂(q)=max_{K ближайших b} (V_b − KV·|q−b|_scaled). Действие, выводящее за ρ>RMAXW или |v|>VMX на любом подшаге, запрещено (узел-стена). Агент: argmax R+γ^5 V̂ с фильтром безопасности v2 (как canal_tree).
Запуск: runpc.sh canal_val.py NB KAP K mode(sel|eval|r20|selT) [TREESEED]; ENV: GAM (на шаг .02, .99), KV (1.0), RB, FM=.1, HS=.5, MARG=.03"""
import os, sys, time, json
os.environ.setdefault('FM', '.1'); os.environ.setdefault('HS', '.5')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, numba as nb
import canal_tree as CT
from canal_tree import C, build_back, cands
GAM = float(os.environ.get('GAM', .99)); KV = float(os.environ.get('KV', 1.0)); NIT = int(os.environ.get('NIT', 400)); G5 = GAM ** 5
@nb.njit(parallel=True)
def edges(P, ACTS, rmaxw, vmx):
    n = P.shape[0]; E = np.zeros((n, 9, 4)); RW = np.zeros((n, 9)); OK = np.zeros((n, 9), np.bool_)
    for i in nb.prange(n):
        for a in range(9):
            x, y, th, vx, vy = P[i, 0], 0., P[i, 1], P[i, 2], P[i, 3]; ok = True; rw = 0.; g = 1.
            for s in range(C.NSUB):
                x, y, th, vx, vy = C.f5(x, y, th, vx, vy, ACTS[a, 0], ACTS[a, 1], C.H, 1)
                rho, al, vr, vt = C.to_ring(x, y, th, vx, vy)
                if rho > rmaxw or abs(vr) > vmx or abs(vt) > vmx: ok = False
                rw += g * .1 * vt / (1. + abs(rho - 1.5)); g *= GAM
            E[i, a, 0] = rho; E[i, a, 1] = al; E[i, a, 2] = vr; E[i, a, 3] = vt; RW[i, a] = rw; OK[i, a] = ok
    return E, RW, OK
@nb.njit(parallel=True)
def vit(J, D, RW, OK, nP, g5, kv, nit, vbad):
    V = np.zeros(nP); K = J.shape[1]
    for it in range(nit):
        Vn = np.empty(nP)
        for i in nb.prange(nP):
            best = vbad
            for a in range(9):
                if not OK[i, a]: continue
                m = -1e30
                for k in range(K):
                    v = V[J[i * 9 + a, k]] - kv * D[i * 9 + a, k]
                    if v > m: m = v
                c = RW[i, a] + g5 * m
                if c > best: best = c
            Vn[i] = best
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < 1e-4: break
    return V, it, d
class AgentV:
    def __init__(self, S, back, KN):
        self.S, self.tree, self.nP = S, back[5], len(back[0]); self.P = np.asarray(back[0]); self.KN = KN
        t0 = time.perf_counter(); E, RW, OK = edges(self.P, C.ACT, C.RMAXW, C.VMX); t1 = time.perf_counter()
        Ew = S.wrap(E.reshape(-1, 4)); d, j = self.tree.query(Ew / S.scale, k=min(KN * 3, len(self.tree.data)), workers=-1)
        J = (j % self.nP).astype(np.int32); D = d.astype(np.float64); self.te = t1 - t0; self.tq = time.perf_counter() - t1
        bad = float(os.environ.get('VBAD', -30.))
        self.V, self.it, self.dres = vit(J, D, RW, OK, self.nP, G5, KV, NIT, bad); self.tv = time.perf_counter() - t1 - self.tq
        print('  дерево %d узлов: edges %.1f с, knn %.1f с, vi %.1f с (it %d, res %.1e); V мин/мед/макс %.2f/%.2f/%.2f' % (self.nP, self.te, self.tq, self.tv, self.it, self.dres, self.V.min(), np.median(self.V), self.V.max()), flush=True)
    def act(self, x, y, th, vx, vy):
        Q, RW, CL = cands(x, y, th, vx, vy, C.ACT)   # RW там приближённо (конец×5); считаем точнее ниже
        ok = (Q[:, 0] <= C.RMAXW) & (CL >= CT.FM); Qw = self.S.wrap(Q)
        k = min(self.KN * 3, len(self.tree.data)); d, j = self.tree.query(Qw / self.S.scale, k=k); j = j % self.nP
        Vq = np.max(self.V[j] - KV * d, axis=1)
        if not ok.any(): ok = CL >= CT.FM
        if not ok.any(): return int(np.argmax(CL))
        return int(np.argmax(np.where(ok, RW * 0 + self.rw(x, y, th, vx, vy) + G5 * Vq, -1e9)))
    def rw(self, x, y, th, vx, vy):
        r = np.zeros(9)
        for a in range(9):
            px, py, pth, pvx, pvy = x, y, th, vx, vy; g = 1.
            for s in range(C.NSUB):
                px, py, pth, pvx, pvy = C.f5(px, py, pth, pvx, pvy, C.ACT[a, 0], C.ACT[a, 1], C.H, 1); rho, al, vr, vt = C.to_ring(px, py, pth, pvx, pvy)
                r[a] += g * .1 * vt / (1. + abs(rho - 1.5)); g *= GAM
        return r
if __name__ == '__main__':
    env = C.mkenv(); out = {}; MODE = CT.MODE; NB, KAP, KN = CT.NB, CT.KAP, CT.KN
    seeds_by = dict(sel=range(100000, 100010), eval=range(200000, 200020), r20=range(200000, 200100), selT=range(100000, 100010))
    for ts in ([CT.TS] if MODE not in ('r20', 'selT') else range(int(os.environ.get('T0', 0)), int(os.environ.get('T1', 20)))):
        S = CT.mkS(ts); t0 = time.perf_counter(); back = build_back(S, C.TAU, NB, float(os.environ.get('RB', .3))); tb = time.perf_counter() - t0
        ag = AgentV(S, back, KN); rs = [CT.episode(env, ag, s) for s in seeds_by[MODE]]; ret = np.array([r[0] for r in rs])
        print('GAM=%g KV=%g NB=%d K=%d ts=%d [%s]: return %.2f±%.2f cost %d обрывов %d t_build %.1f с мс/шаг %.2f' % (GAM, KV, NB, KN, ts, MODE, ret.mean(), ret.std(), sum(r[1] for r in rs), sum(r[2] for r in rs), tb, 1e3 * np.mean([r[3] for r in rs])), flush=True)
        out[ts] = dict(ret=ret.tolist(), cost=sum(r[1] for r in rs), term=sum(r[2] for r in rs))
    json.dump(out, open('canal_val_%s_%g_%g_%s.json' % (MODE, GAM, KV, os.environ.get('T0', '')), 'w'))
