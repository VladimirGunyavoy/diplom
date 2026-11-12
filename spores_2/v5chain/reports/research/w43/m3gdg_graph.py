"""w43 [1] m3gdg: граф-схема («кинодинамическая дорожная карта») на 6D манипуляторе + g + 2 диска, цепочка целей c7 (GSEED 11, K переходов).
Построение один раз (узлы RNDN, рёбра 8 слоёв × TAUS, LP-веса KLP), на каждую цель — только VI (VCAP), на переход — жадный агент по V̂ (LP-интерполяция).
Запуск (aida2): cd ~/spore_v5/w43 && NB=100000 KLP=20 VCAP=40 TAUS=.1,.2,.4 python3 m3gdg_graph.py 19 SEED  → out/m3gdg_graph_<NB>_<SEED>.json"""
import os, sys, time, json, resource
os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.expanduser('~/spore_v5/w43/spores_2'); sys.path[:0] = [os.path.join(HERE, 'bench'), os.path.join(HERE, 'v6'), os.path.dirname(os.path.abspath(__file__))]
import numpy as np, numba as nb, common
import manip3g_dyn as G3
from m3gdg_goals import MGoal, OB, transitions, goals, wr, PI
from src.atlas6.adaptive_nd import SysN, _images, _scaled
from scipy.spatial import cKDTree
from lpcore import lp_batch, vit_dl
K = int(sys.argv[1]); SEED = int(sys.argv[2]); NB = int(os.environ.get('NB', 100000)); KLP = int(os.environ.get('KLP', 20)); VCAP = float(os.environ.get('VCAP', 40)); NIT = int(os.environ.get('NIT', 3000))
TAUS = np.array([float(v) for v in os.environ.get('TAUS', '.1,.2,.4').split(',')]); NS = 5; WM = 3.; WBOX = 3.05; CL = float(os.environ.get('CL', .03)); Rq, Rw = .28, .58
ACT = np.array(G3.LAYERS6, float); LA = len(ACT); LT = len(TAUS); L2 = LA * LT; SIDX = np.repeat(np.arange(LA), LT); TAUV = np.tile(TAUS, LA); d = 6
e0 = MGoal(goals(K)[0], goals(K)[1]); A = e0.A; Ai = e0.Ai; Bm = np.linalg.inv(A.T); Sm = e0.S; gw = e0.gw
PER = np.array([2 * PI] * 3 + [0.] * 3); S = SysN(None, LA, np.ones(d), lambda P: np.zeros(len(P), bool), np.zeros((1, d)), per=PER)

@nb.njit(cache=True)
def acc(x, u, A, Ai, Bm, Sm, gw):
    th = np.zeros(3); thd = np.zeros(3)
    for i in range(3):
        for j in range(3): th[i] += A[i, j] * x[j]; thd[i] += A[i, j] * x[3 + j]
    M = np.empty((3, 3)); b = np.zeros(3)
    for i in range(3):
        for j in range(3):
            dd = th[i] - th[j]; M[i, j] = Sm[i, j] * np.cos(dd); b[i] -= Sm[i, j] * np.sin(dd) * thd[j] ** 2
        s = 0.
        for j in range(3): s += Bm[i, j] * u[j]
        b[i] += s - gw[i] * np.cos(th[i])
    for c in range(3):
        for r in range(c + 1, 3):
            f = M[r, c] / M[c, c]
            for k in range(c, 3): M[r, k] -= f * M[c, k]
            b[r] -= f * b[c]
    t = np.zeros(3)
    for r in range(2, -1, -1):
        s = b[r]
        for k in range(r + 1, 3): s -= M[r, k] * t[k]
        t[r] = s / M[r, r]
    out = np.empty(6)
    for i in range(3):
        out[i] = x[3 + i]; s = 0.
        for j in range(3): s += Ai[i, j] * t[j]
        out[3 + i] = s
    return out

@nb.njit(parallel=True, cache=True)
def sub_flow(P, U, TV, SI, NSub, A, Ai, Bm, Sm, gw):
    """P (n,6) × действия (слой SI[a], τ TV[a]) → (n,L2,NSub,6) состояния в конце каждой из NSub подпорций (RK4, шаг ≤ .05)"""
    n = P.shape[0]; La = SI.shape[0]; out = np.empty((n, La, NSub, 6), np.float32)
    for i in nb.prange(n):
        for a in range(La):
            u = U[SI[a]]; x = P[i].copy(); tau = TV[a]; nst = max(1, int(np.ceil(tau / .05 - 1e-9))); per = nst // NSub if nst >= NSub else 1
            ntot = max(nst, NSub); ntot = ((ntot + NSub - 1) // NSub) * NSub; per = ntot // NSub; h = tau / ntot
            for k in range(NSub):
                for _ in range(per):
                    k1 = acc(x, u, A, Ai, Bm, Sm, gw); x2 = x + h / 2 * k1; k2 = acc(x2, u, A, Ai, Bm, Sm, gw); x3 = x + h / 2 * k2; k3 = acc(x3, u, A, Ai, Bm, Sm, gw); x4 = x + h * k3; k4 = acc(x4, u, A, Ai, Bm, Sm, gw)
                    x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                for j in range(6): out[i, a, k, j] = x[j]
    return out

def edges(P):
    """→ Zs (n,L2,NS,6) [углы свёрнуты], OK (n,L2): |w|≤WBOX и зазор ≥ CL на всех подпорциях"""
    Zs = sub_flow(np.ascontiguousarray(P, float), ACT, TAUV, SIDX, NS, A, Ai, Bm, Sm, gw); Zs[..., :3] = ((Zs[..., :3] + np.float32(PI)) % np.float32(2 * PI)) - np.float32(PI)
    ok = (np.abs(Zs[..., 3:]) <= WBOX).all((-1, -2)); n = len(P)
    for a in range(0, n, 20000): ok[a:a + 20000] &= (G3.clearance(Zs[a:a + 20000, :, :, :3].astype(float), OB) >= CL).all(-1) if False else (np.min(G3.clearance(Zs[a:a + 20000, :, :, :3].astype(float), OB), -1) >= CL)
    return Zs, ok

def goal_hits(Zs, qs):
    """GL (n,L2): время первого попадания подпорции в окно цели qs (|q−q*|<Rq, |w|<Rw), 0 — не попадает"""
    Z = Zs.astype(float) if len(Zs) < 5000 else Zs; dq = np.abs(((Z[..., :3] - qs + PI) % (2 * PI)) - PI).max(-1); dw = np.abs(Z[..., 3:]).max(-1); g = (dq < Rq) & (dw < Rw)
    first = np.where(g.any(-1), g.argmax(-1) + 1, 0); return first * (TAUV[None, :] / NS)

def in_goal(P, qs): P = np.asarray(P); dq = np.abs(((P[..., :3] - qs + PI) % (2 * PI)) - PI).max(-1); return (dq < Rq) & (np.abs(P[..., 3:]).max(-1) < Rw)

class Graph:
    def __init__(s):
        t0 = time.perf_counter(); rng = np.random.default_rng(1000 + SEED); lo = np.array([-PI] * 3 + [-WM] * 3); P = rng.uniform(lo, -lo, (3 * NB, d)); P = P[G3.clearance(P[:, :3], OB) >= CL][:NB]; s.P = P; s.nP = len(P)
        s.tree = cKDTree(_scaled(S, _images(S, P))); s.t_nodes = time.perf_counter() - t0; t1 = time.perf_counter()
        s.Zs, s.OK = edges(P); s.t_edges = time.perf_counter() - t1; t2 = time.perf_counter(); E = s.Zs[:, :, -1].astype(float).reshape(-1, d)
        s.J, s.LAM, ins = s.weights(E); s.INS = ins.reshape(-1, L2); s.J = s.J.reshape(s.nP * L2, -1).astype(np.int32); s.LAM = s.LAM.astype(np.float32); s.t_lp = time.perf_counter() - t2
        s.t_build = time.perf_counter() - t0; s.rss_build = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print('граф: узлов %d, дерево %.1f с, рёбра %.1f с, LP-веса %.1f с (вне оболочки %.2f%% допустимых), всего %.1f с, RSS пик %.1f ГБ' % (s.nP, s.t_nodes, s.t_edges, s.t_lp, 100 * (s.OK & ~s.INS).sum() / max(1, s.OK.sum()), s.t_build, s.rss_build), flush=True)
    def weights(s, Q):
        Qs = S.wrap(np.asarray(Q, float)); dd, jj = s.tree.query(Qs, k=KLP, workers=-1); NBh = s.tree.data[jj]; Jl, L, ok = lp_batch(np.ascontiguousarray(NBh), np.ascontiguousarray(Qs))
        return np.take_along_axis(jj, Jl, 1) % s.nP, L, ok
    def vi(s, qs):
        t0 = time.perf_counter(); GL = goal_hits(s.Zs, qs); OKd = (s.OK & s.INS) | (s.OK & (GL > 0)); ING = in_goal(s.P, qs); W, it, res = vit_dl(s.J, s.LAM, OKd, GL, ING, s.nP, TAUV, NIT, 1e3, VCAP); return W, time.perf_counter() - t0, it, res

def nn_v(G, W, Q):
    dd, j = G.tree.query(S.wrap(Q), k=9); j = j % G.nP; return (W[j] + KVN * dd).min(1)
KVN = 1.; DEP = int(os.environ.get('DEP', 1)); DEPPEN = float(os.environ.get('DEPPEN', 0))
def vhat(G, W, Q):
    J, lam, ok = G.weights(Q); w = (lam * W[J]).sum(1); bad = ~ok | (W[J] >= VCAP - 1e-9).any(1)
    if bad.any(): w[bad] = nn_v(G, W, np.asarray(Q)[bad])
    return w

def episode(G, W, x0, qs, tmax=14.):
    x = np.array(x0, float); us = []; dts = []; t = 0.; tm = []
    while t < tmax:
        t1 = time.perf_counter(); Zs, ok = edges(x[None]); Zs = Zs[0]; ok = ok[0]; GL = goal_hits(Zs[None], qs)[0]; c = np.full(L2, 1e9); fin = ok & (GL > 0); c[fin] = GL[fin]
        if fin.any(): a = int(np.argmin(c))
        else:
            m = np.flatnonzero(ok)
            if len(m) == 0: break
            if DEP >= 2:
                E1 = Zs[m, -1].astype(float); Z2, ok2 = edges(E1); GL2 = goal_hits(Z2, qs); n1 = len(m); c2 = np.full((n1, L2), 1e9); f2 = ok2 & (GL2 > 0); c2[f2] = GL2[f2]; rest = ok2 & ~f2
                if rest.any(): c2[rest] = TAUV[None, :].repeat(n1, 0)[rest] + vhat(G, W, Z2[:, :, -1].astype(float)[rest])
                c[m] = TAUV[m] + np.minimum(c2.min(1), vhat(G, W, E1) + DEPPEN)
            else: c[m] = TAUV[m] + vhat(G, W, Zs[m, -1].astype(float))
            a = int(np.argmin(c))
        tm.append(time.perf_counter() - t1); us.append(ACT[SIDX[a]]); dts.append(float(TAUV[a])); t += TAUV[a]
        if fin.any(): break
        x = Zs[a, -1].astype(float)
    return us, dts, tm

if __name__ == '__main__':
    out = dict(K=K, NB=NB, KLP=KLP, VCAP=VCAP, TAUS=TAUS.tolist(), seed=SEED); Tr = transitions(K); ref = json.load(open(os.path.join(common.DATA, 'm3gdg_ref.json')))['res']
    G = Graph(); out.update(nodes=G.nP, t_nodes=G.t_nodes, t_edges=G.t_edges, t_lp=G.t_lp, t_build=G.t_build, rss_build=G.rss_build, stored_mb=(G.P.astype(np.float32).nbytes + G.J.nbytes + G.LAM.nbytes + G.OK.nbytes) / 1e6, res=[])
    nk = int(os.environ.get('NK', K))
    for k, (st, gl) in enumerate(Tr[:nk]):
        e = MGoal(gl, st); qs = np.asarray(gl, float); W, tv, it, res = G.vi(qs); w0 = float(vhat(G, W, np.r_[st, 0., 0., 0.][None])[0] if True else 0)
        t1 = time.perf_counter(); us, dts, tm = episode(G, W, np.r_[st, 0., 0., 0.], qs); tq = time.perf_counter() - t1; Tv, why = common.verify(e, e.X0[0], us, dts); Tr_ = ref[k]['T']
        r = dict(k=k, T=Tv, why=str(why), Tref=Tr_, ratio=None if (Tv is None or not Tr_) else Tv / Tr_, t_vi=tv, vi_it=int(it), t_query=tq, steps=len(us), ms_step=1e3 * float(np.mean(tm)) if tm else None, W0=w0, fin=float((W < VCAP - 1).mean()), rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        out['res'].append(r); print(r, flush=True)
    rr = np.array([x['ratio'] for x in out['res'] if x['ratio'] is not None and x['ratio'] <= 3]); nt = sum(1 for x in out['res'] if x['Tref'])
    if len(rr): print('ИТОГ: дошли %d/%d, T/Tref мед %.3f ср %.3f p90 %.3f max %.3f; VI мед %.1f с, запрос мед %.2f с' % (len(rr), nt, np.median(rr), rr.mean(), np.percentile(rr, 90), rr.max(), np.median([x['t_vi'] for x in out['res']]), np.median([x['t_query'] for x in out['res']])), flush=True)
    os.makedirs('out', exist_ok=True); json.dump(out, open('out/m3gdg_graph_%d_%d%s.json' % (NB, SEED, os.environ.get('TAG', '')), 'w'), default=float)
