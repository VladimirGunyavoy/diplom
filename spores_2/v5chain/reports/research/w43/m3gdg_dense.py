"""w43 [1]-б (решение research-39): посев с концентрацией у цели. Общие равномерные NU узлов + на каждую цель обратное дерево из окна цели (build_back, NT узлов, τ TT) → граф = U ∪ дерево,
рёбра дерева считаются на цель, остальное общее; LP-веса и VI на цель. Протокол c7 (K переходов, «дошли» по common.verify).
Запуск (aida2, ядра: NUMBA_NUM_THREADS=10 KDW=10): cd ~/spore_v5/w43 && NU=150000 NT=50000 TT=.3 python3 m3gdg_dense.py 19 SEED → out/m3gdg_dense_<NU>_<NT>_<SEED>.json"""
import os, sys, time, json, resource
K = int(sys.argv[1]); SEED = int(sys.argv[2]); sys.argv = ['m3gdg_graph.py', str(K), str(SEED)]; NU = int(os.environ.get('NU', 150000)); NT = int(os.environ.get('NT', 50000)); TT = float(os.environ.get('TT', .3)); os.environ['NB'] = str(NU)
import numpy as np
import m3gdg_graph as M
from m3gdg_graph import *
import tree_fast
from m3gdg_graph import _scaled, _images
from src.atlas6.adaptive_nd import SysN as SN, build_back
G3_ = M.G3; US_ = np.array(G3_.LAYERS6); flow_all = tree_fast.make_flow_all('m3gd', US_); fl = lambda P, s, t: G3_.flow(P, s, t, dt_max=.05, g=.3)
OBS = lambda P: G3_.clearance(np.asarray(P)[..., :3], OB)

def tree_nodes(qs, seed):
    ing = lambda P: M.in_goal(P, qs); rng = np.random.default_rng(seed); seeds = []
    for _ in range(32): v = rng.uniform(-1, 1, 6); v = v / np.max(np.abs(v)) * 0.99; seeds.append(np.r_[qs, 0, 0, 0] + v * np.array([.3, .3, .3, .6, .6, .6]))
    St = SN(fl, 8, (PI, PI, PI, 3., 3., 3.), ing, seeds, per=(2 * PI, 2 * PI, 2 * PI, 0, 0, 0), ok=lambda p: np.max(np.abs(p[3:])) <= 3., blocked=lambda P: OBS(P) < 0); St.flow_all = flow_all
    back = build_back(St, TT, NT, .15); P = np.asarray(back[0], float); P[:, :3] = (P[:, :3] + PI) % (2 * PI) - PI; return P[(OBS(P) >= CL) & (np.abs(P[:, 3:]) <= 3.).all(1)]

if __name__ == '__main__':
    out = dict(K=K, NU=NU, NT=NT, TT=TT, seed=SEED, KLP=KLP, VCAP=VCAP, TAUS=TAUS.tolist(), res=[]); Tr = transitions(K); ref = json.load(open(os.path.join(common.DATA, 'm3gdg_ref.json')))['res']
    t0 = time.perf_counter(); rng = np.random.default_rng(1000 + SEED); lo = np.array([-PI] * 3 + [-3.] * 3); P0 = rng.uniform(lo, -lo, (3 * NU, 6)); P0 = P0[OBS(P0) >= CL][:NU]; Zs0, OK0 = edges(P0); tu = time.perf_counter() - t0; out['t_uniform'] = tu; print('равномерные %d узлов: %.1f с' % (len(P0), tu), flush=True)
    for k, (st, gl) in enumerate(Tr):
        e = MGoal(gl, st); qs = np.asarray(gl, float); t1 = time.perf_counter(); Pt = tree_nodes(qs, SEED * 100 + k); t2 = time.perf_counter(); Zt, Ot = edges(Pt)
        G = Graph.__new__(Graph); G.P = np.vstack([P0, Pt]); G.nP = len(G.P); G.Zs = np.concatenate([Zs0, Zt]); G.OK = np.concatenate([OK0, Ot]); G.tree = cKDTree(_scaled(S, _images(S, G.P))); t3 = time.perf_counter()
        J, LAM, ins = G.weights(G.Zs[:, :, -1].astype(float).reshape(-1, 6)); G.INS = ins.reshape(-1, L2); G.J = J.astype(np.int32); G.LAM = LAM.astype(np.float32); t4 = time.perf_counter()
        W, tv, it, res = G.vi(qs); w0 = float(vhat(G, W, np.r_[st, 0., 0., 0.][None])[0]); t5 = time.perf_counter(); us, dts, tm = episode(G, W, np.r_[st, 0., 0., 0.], qs); tq = time.perf_counter() - t5
        Tv, why = common.verify(e, e.X0[0], us, dts); Tr_ = ref[k]['T']
        r = dict(k=k, T=Tv, why=str(why), Tref=Tr_, ratio=None if (Tv is None or not Tr_) else Tv / Tr_, W0=w0, fin=float((W < VCAP - 1).mean()), t_tree=t2 - t1, t_edges=time.perf_counter() - t2 - (t4 - t3) - tv - tq - (t5 - t4 - tv) if False else None, t_graph=t4 - t2, t_vi=tv, t_goal_total=t5 - t1, t_query=tq, steps=len(us), stop=M.M_STOP[0], n_tree=len(Pt), rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        out['res'].append(r); print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()}, flush=True)
        del G, J, LAM
    nn_ = NU + NT; out['stored_mb'] = (nn_ * 6 * 4 + nn_ * L2 * 7 * 8 + nn_ * L2) / 1e6; out['violations'] = sum(1 for x in out['res'] if x['Tref'] and x['why'] == 'box'); out['rss_peak'] = max(x['rss'] for x in out['res'])
    print('нарушений (verify): %d; хранимое ≈ %.0f МБ (узлы float32 + J int32 + LAM float32 на 200k×24 рёбер на цель); RSS пик %.1f ГБ' % (out['violations'], out['stored_mb'], out['rss_peak']), flush=True)
    R = [x for x in out['res'] if x['Tref']]; ok = [x for x in R if x['T'] is not None and x['why'] == 'ok']; ra = np.array([x['T'] / x['Tref'] for x in ok])
    if len(ra): print('ИТОГ: дошли %d/%d, T/Tref мед %.3f p90 %.3f max %.3f; W0/Tref мед %.1f; fin мед %.2f; на цель %.0f с' % (len(ok), len(R), np.median(ra), np.percentile(ra, 90), ra.max(), np.median([x['W0'] / x['Tref'] for x in R]), np.median([x['fin'] for x in R]), np.median([x['t_goal_total'] for x in R])), flush=True)
    os.makedirs('out', exist_ok=True); json.dump(out, open('out/m3gdg_dense_%d_%d_%d%s.json' % (NU, NT, SEED, os.environ.get('TAG', '')), 'w'), default=float)
