"""d3 [3]: граф-схема без подбора на 6D/8D средах с инерцией (m3d, m3gd, m3gu, m8) — как w43/m3gdg_graph.py, но среда = envs_6d (a31, только чтение),
цель — окно среды, старты и T_ref — 20 стартов атласа prelim. RNDN-узлы, 2^m угловых слоёв × TAUS, LP-веса KLP, VCAP, жадный агент DEP2 DEPPEN по V̂ (LP-интерп.).
Метрики: дошли, T/T_ref, НАРУШЕНИЯ (common.verify: why=='box' — |w| вне коробки), build/VI с, мс/шаг, пик RSS, хранимое (МБ).
Запуск (aida, ~/spore_v5/d3): ENVN=m3d NB=100000 SEED=0 [TAUS=.2,.4,.8 KLP=28 VCAP=40 DEP=2 DEPPEN=1] python3 g6d.py → out/g6d_<env>_<NB>_<seed>.json"""
import os, sys, time, json, resource
os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.expanduser('~/spore_v5/d4/spores_2'); sys.path[:0] = [os.path.join(HERE, 'bench'), os.path.join(HERE, 'v6'), os.path.dirname(os.path.abspath(__file__))]
from src.atlas6.adaptive_nd import SysN, _images, _scaled
import numpy as np, common, envs_6d
from scipy.spatial import cKDTree
from lpcore import lp_batch, vit_dl
ENVN = os.environ.get('ENVN', 'm3d'); NB = int(os.environ.get('NB', 100000)); SEED = int(os.environ.get('SEED', 0)); KLP = int(os.environ.get('KLP', 28)); VCAP = float(os.environ.get('VCAP', 40)); NIT = int(os.environ.get('NIT', 3000))
TAUS = np.array([float(v) for v in os.environ.get('TAUS', '.2,.4,.8').split(',')]); NS = 5; DEP = int(os.environ.get('DEP', 2)); DEPPEN = float(os.environ.get('DEPPEN', 1.)); KVN = 1.; CHE = int(os.environ.get('CHE', 20000))
env = common.make(ENVN); d = env.n; nu = env.m; TOLW = env.TOLW
ACT = np.array([[env.umax[i] if (s >> i) & 1 else env.umin[i] for i in range(nu)] for s in range(2 ** nu)], float); LA = len(ACT); LT = len(TAUS); L2 = LA * LT; SIDX = np.repeat(np.arange(LA), LT); TAUV = np.tile(TAUS, LA)
PER = np.asarray(env.period, float); WB = 3. + TOLW; S = SysN(None, LA, np.ones(d), lambda P: np.zeros(len(P), bool), np.zeros((1, d)), per=PER)
wrap_ang = lambda Z: np.where(PER > 0, ((Z + PER / 2) % np.where(PER > 0, PER, 1.)) - PER / 2, Z)

def flow(P, u, tau):
    """RK4 шаг ≤ .05, возврат (n,NS,d): состояния в конце NS подпорций"""
    ntot = max(NS, int(np.ceil(tau / .05 - 1e-9))); ntot = ((ntot + NS - 1) // NS) * NS; per = ntot // NS; h = tau / ntot; x = np.asarray(P, float).copy(); out = np.empty((len(P), NS, d)); u = np.asarray(u, float)
    for k in range(NS):
        for _ in range(per):
            k1 = env.f(x, u); k2 = env.f(x + h / 2 * k1, u); k3 = env.f(x + h / 2 * k2, u); k4 = env.f(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[:, k] = x
    return out

def edges(P):
    """→ Zs (n,L2,NS,d) [углы свёрнуты], OK (n,L2): |w| ≤ 3+TOLW на всех подпорциях (как verify), GL (n,L2): время первого попадания в окно цели"""
    P = np.asarray(P, float); n = len(P); Zs = np.empty((n, L2, NS, d), np.float32); OK = np.empty((n, L2), bool); GL = np.zeros((n, L2))
    for a in range(L2):
        Z = flow(P, ACT[SIDX[a]], TAUV[a]); ok = ~env.out_of_box(Z).any(-1); g = env.in_goal(Z); first = np.where(g.any(-1), g.argmax(-1) + 1, 0)
        Zs[:, a] = wrap_ang(Z); OK[:, a] = ok; GL[:, a] = np.where(first > 0, first * TAUV[a] / NS, 0.)
        # попадание засчитываем только до выхода из коробки (первое попадание раньше выхода)
        bad = env.out_of_box(Z); fb = np.where(bad.any(-1), bad.argmax(-1) + 1, 99); GL[:, a] = np.where((first > 0) & (first <= fb), GL[:, a], 0.); OK[:, a] = np.where(GL[:, a] > 0, True, OK[:, a])
    return Zs, OK, GL

class Graph:
    def __init__(s):
        t0 = time.perf_counter(); rng = np.random.default_rng(1000 + SEED); lo = np.where(PER > 0, -PER / 2, env.lo); P = rng.uniform(lo, -lo, (NB, d)); s.P = P; s.nP = NB
        s.tree = cKDTree(_scaled(S, _images(S, P))); s.t_nodes = time.perf_counter() - t0; t1 = time.perf_counter(); Zs = []; OK = []; GL = []
        for a0 in range(0, NB, CHE): z, o, g = edges(P[a0:a0 + CHE]); Zs.append(z[:, :, -1].copy()); OK.append(o); GL.append(g)
        s.OK = np.concatenate(OK); s.GL = np.concatenate(GL); E = np.concatenate(Zs).astype(float).reshape(-1, d); del Zs; s.t_edges = time.perf_counter() - t1; t2 = time.perf_counter()
        J, LAM, ins = s.weights(E); s.J = J.reshape(s.nP * L2, -1).astype(np.int32); s.LAM = LAM.astype(np.float32); s.INS = ins.reshape(-1, L2); s.t_lp = time.perf_counter() - t2
        s.t_build = time.perf_counter() - t0; s.rss_build = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        print('граф %s: узлов %d, дерево %.1f с, рёбра %.1f с, LP %.1f с (вне оболочки %.2f%% допустимых), всего %.1f с, RSS пик %.1f ГБ' % (ENVN, s.nP, s.t_nodes, s.t_edges, s.t_lp, 100 * (s.OK & ~s.INS).sum() / max(1, s.OK.sum()), s.t_build, s.rss_build), flush=True)
    def weights(s, Q):
        Qs = S.wrap(np.asarray(Q, float)); n = len(Qs); CH = 300000; Js = []; Ls = []; Os = []
        for a0 in range(0, n, CH):
            q = Qs[a0:a0 + CH]; dd, jj = s.tree.query(q, k=KLP, workers=int(os.environ.get('KDW', -1))); Jl, L, ok = lp_batch(np.ascontiguousarray(s.tree.data[jj]), np.ascontiguousarray(q)); Js.append(np.take_along_axis(jj, Jl, 1) % s.nP); Ls.append(L); Os.append(ok)
        return np.concatenate(Js), np.concatenate(Ls), np.concatenate(Os)
    def vi(s):
        t0 = time.perf_counter(); OKd = (s.OK & s.INS) | (s.OK & (s.GL > 0)); ING = env.in_goal(s.P); W, it, res = vit_dl(s.J, s.LAM, OKd, s.GL, ING, s.nP, TAUV, NIT, 1e3, VCAP); return W, time.perf_counter() - t0, it, res

def nn_v(G, W, Q): dd, j = G.tree.query(S.wrap(Q), k=9); j = j % G.nP; return (W[j] + KVN * dd).min(1)
def vhat(G, W, Q):
    J, lam, ok = G.weights(Q); w = (lam * W[J]).sum(1); bad = ~ok | (W[J] >= VCAP - 1e-9).any(1)
    if bad.any(): w[bad] = nn_v(G, W, np.asarray(Q)[bad])
    return w

def episode(G, W, x0, tmax):
    x = np.array(x0, float); us = []; dts = []; t = 0.; tm = []; stop = ''
    while t < tmax:
        t1 = time.perf_counter(); Zs, ok, GL = edges(x[None]); Zs = Zs[0]; ok = ok[0]; GL = GL[0]; c = np.full(L2, 1e9); fin = ok & (GL > 0); c[fin] = GL[fin]
        if fin.any(): a = int(np.argmin(c))
        else:
            m = np.flatnonzero(ok)
            if len(m) == 0: stop = 'dead_end'; break
            E1 = Zs[m, -1].astype(float)
            if DEP >= 2:
                Z2, ok2, GL2 = edges(E1); n1 = len(m); c2 = np.full((n1, L2), 1e9); f2 = ok2 & (GL2 > 0); c2[f2] = GL2[f2]; rest = ok2 & ~f2
                if rest.any(): c2[rest] = TAUV[None, :].repeat(n1, 0)[rest] + vhat(G, W, Z2[:, :, -1].astype(float)[rest])
                c[m] = TAUV[m] + np.minimum(c2.min(1), vhat(G, W, E1) + DEPPEN)
            else: c[m] = TAUV[m] + vhat(G, W, E1)
            a = int(np.argmin(c))
        tm.append(time.perf_counter() - t1); us.append(ACT[SIDX[a]]); dts.append(float(TAUV[a])); t += TAUV[a]
        if fin.any(): break
        x = Zs[a, -1].astype(float)
    return us, dts, tm, stop

if __name__ == '__main__':
    out = dict(env=ENVN, NB=NB, KLP=KLP, VCAP=VCAP, TAUS=TAUS.tolist(), seed=SEED, DEP=DEP, DEPPEN=DEPPEN); G = Graph(); W, tv, it, res = G.vi()
    out.update(t_nodes=G.t_nodes, t_edges=G.t_edges, t_lp=G.t_lp, t_build=G.t_build, t_vi=tv, vi_it=int(it), rss_build=G.rss_build, stored_mb=(G.P.astype(np.float32).nbytes + G.J.nbytes + G.LAM.nbytes + G.OK.nbytes + W.nbytes) / 1e6, res=[]); print('VI %.1f с, %d ит, res %.1e, конечных %.1f%%' % (tv, it, res, 100 * (W < VCAP - 1).mean()), flush=True)
    ns = int(os.environ.get('NSTART', len(env.X0)))
    for i in range(ns):
        if not env.keep[i]: continue
        t1 = time.perf_counter(); us, dts, tm, stop = episode(G, W, env.X0[i], 3 * env.Tref[i] + 1.); tq = time.perf_counter() - t1; Tv, why = common.verify(env, env.X0[i], us, dts)
        r = dict(i=i, T=Tv, why=str(why), Tref=float(env.Tref[i]), ratio=None if Tv is None else Tv / float(env.Tref[i]), t_query=tq, steps=len(us), stop=stop, ms_step=1e3 * float(np.mean(tm)) if tm else None, W0=float(vhat(G, W, env.X0[i][None])[0]))
        out['res'].append(dict(r, us=[[float(v) for v in u] for u in us], dts=[float(t) for t in dts])); print(r, flush=True)
    rr = np.array([x['ratio'] for x in out['res'] if x['ratio'] is not None and x['ratio'] <= 3]); nt = len(out['res'])
    if len(rr): print('ИТОГ %s: дошли %d/%d, T/Tref мед %.3f ср %.3f p90 %.3f max %.3f; запрос мед %.2f с' % (ENVN, len(rr), nt, np.median(rr), rr.mean(), np.percentile(rr, 90), rr.max(), np.median([x['t_query'] for x in out['res']])), flush=True)
    out['violations'] = sum(1 for x in out['res'] if x['why'] == 'box'); print('нарушений (verify: выход |w| из коробки): %d из %d; хранимое %.0f МБ, RSS пик %.1f ГБ' % (out['violations'], nt, out['stored_mb'], G.rss_build), flush=True)
    os.makedirs(os.path.expanduser('~/spore_v5/d4/out'), exist_ok=True); json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/g6d_%s_%d_%d%s.json' % (ENVN, NB, SEED, os.environ.get('TAG', ''))), 'w'), default=float)
