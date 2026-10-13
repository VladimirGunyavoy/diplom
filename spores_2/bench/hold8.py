"""HOLD8 (w30, research-27): 8D (m8) с потоком «удержание на пределе |w|=3» в прямом дереве и стыковке; обратное дерево — обычный ±1 (tree_fast). Без refine спор:
кандидаты (попадания + лучшие промахи) → тёплый старт полного OCP (N50, свободное T) с эффективными u потока удержания; проверка единым common.verify.
Алгоритм hold_u — reports/research/r27/hold_flow.py. python hold8.py m8 NB NF KTOP STARTS [tag]  (aida: PYTHONPATH=$HOME/spore_v5/r5/pylib TREEFAST=1)"""
import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, common, envs_6d
sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
from src.atlas6.adaptive_nd import SysN, build_back, back_heuristic
from src.atlas6 import corridor_nd as CN
import multiprocessing as mp
NB, NF, KTOP = [int(a) for a in sys.argv[2:5]]; STARTS = [int(v) for v in sys.argv[5].split(',')]; tag = sys.argv[6] if len(sys.argv) > 6 else ''
e = common.make('m8'); n = 4; tau = .4; Rq, Rw, WM = .3, .6, 3.; rho = .15; C = np.array(e.C3, float); H = .01
US_ = np.array([[1.0 if (s >> i) & 1 else -1.0 for i in range(n)] for s in range(2 ** n)])
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
ing = lambda P: (np.max(np.abs(wr(P[..., :n] - C[:n])), -1) < Rq) & (np.max(np.abs(P[..., n:]), -1) < Rw)
miss = lambda X: np.maximum(np.max(np.abs(wr(X[..., :n] - C[:n])), -1) - Rq, 0) + np.maximum(np.max(np.abs(X[..., n:]), -1) - Rw, 0)
rng = np.random.default_rng(0); seeds = []; R8 = np.array([Rq] * n + [Rw] * n)
for _ in range(32):
    v = rng.uniform(-1, 1, 2 * n); v = v / np.max(np.abs(v)) * 0.99; seeds.append(C + v * R8)
import manip3g_dyn as G3
fl_plain = lambda P, s, t: G3.flow(P, s, t, dt_max=.02, n=n, g=e.G)
S = SysN(fl_plain, 2 ** n, (np.pi,) * n + (WM,) * n, ing, seeds, per=(2 * np.pi,) * n + (0,) * n, ok=lambda p: np.max(np.abs(p[n:])) <= WM)
import tree_fast; tree_fast.install(); fa_plain = tree_fast.make_flow_all('m8', US_); S.flow_all = fa_plain
def rk4(Y, U, h):
    k1 = e.f(Y, U); k2 = e.f(Y + h / 2 * k1, U); k3 = e.f(Y + h / 2 * k2, U); k4 = e.f(Y + h * k3, U); return Y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
HOLD = int(os.environ.get('HOLD', 1))      # 0 = контроль: обычный поток ±1 (дерево, стыковка, replay), всё остальное то же
def hold_u(Y, U):
    if not HOLD: return U.copy()
    B = len(Y); a0 = e.f(Y, np.zeros((B, n)))[:, n:]; J = np.stack([e.f(Y, np.tile(np.eye(n)[j], (B, 1)))[:, n:] - a0 for j in range(n)], -1)
    Ue = U.copy(); Sm = np.zeros((B, n), bool); w = Y[:, n:]
    for _ in range(n):
        qdd = a0 + np.einsum('bij,bj->bi', J, Ue); wn = w + H * qdd; new = (np.abs(wn) > WM) & ~Sm; tg = (WM * np.sign(wn) - w) / H
        if not new.any(): break
        Sm |= new
        for b in np.where(new.any(1))[0]:
            s = Sm[b]; fr = ~s; us = np.linalg.solve(J[b][np.ix_(s, s)], tg[b][s] - (a0[b][s] + J[b][np.ix_(s, fr)] @ U[b][fr])); Ue[b, s] = np.clip(us, -1, 1)
    return Ue
import hold8_c as HC; HC.init(G3.MS, G3.L, e.G)
def hold_flow(Y, U, t, rec=None):
    if HOLD and int(os.environ.get('HOLDC', 1)):      # C-версия (hold8_c.c): совпадает с numpy до 1e-15, ×15 быстрее
        Yo, Yh, R = HC.hold_flow_c(Y, U, t, WM, H, rec=True)
        if rec is not None: rec.extend(R[0])
        return Yo, Yh
    """t>0: вперёд с удержанием; возвращает (Y_конец, Y_середина). rec — список для записи эффективных u (только B=1)."""
    k = max(1, int(round(t / H))); Y = np.array(Y, float); half = Y
    for i in range(k):
        Ue = hold_u(Y, U)
        if rec is not None: rec.append(Ue[0].copy())
        Y = rk4(Y, Ue, t / k)
        if i == k // 2 - 1: half = Y
    return Y, half
def fa_hold(P, t):
    if t < 0: return fa_plain(P, t)
    Y, half = hold_flow(np.tile(np.asarray(P, float), (len(US_), 1)), US_, t); fa_hold.half = half; return Y
def fl_hold(P, s, t):
    P = np.asarray(P, float)
    if t < 0: return fl_plain(P, s, t)
    sh = P.shape; Y = P.reshape(-1, 2 * n); Y, _ = hold_flow(Y, np.tile(US_[s], (len(Y), 1)), t); return Y.reshape(sh)
def replay(x0, path):
    """путь [(слой, dt)] с удержанием → (список u по подшагам H, длительности, конечное состояние)"""
    Y = np.array(x0, float)[None]; rec = []
    for s, dt in path: Y, _ = hold_flow(Y, US_[s][None], dt, rec)
    return np.array(rec), Y[0]
import ocp_m8_lib as OL; HS = OL.make_solver(50, 4)
def polish(x0, rec, N=50):
    S_, lbx, ubx, lbg, ubg = HS; T = len(rec) * H; ws = float(os.environ.get('WSHRINK', .02))
    if ws: lbx = lbx.copy(); ubx = ubx.copy(); wi = np.arange(8 * (N + 1)).reshape(N + 1, 8)[:, n:].ravel(); lbx[wi] = np.maximum(lbx[wi], -(WM - ws)); ubx[wi] = np.minimum(ubx[wi], WM - ws)      # запас по |w| между узлами (проверяльщик RK4 h .01 допускает 3+TOLW)
    edges = np.linspace(0, len(rec), N + 1).astype(int); Ug = np.array([rec[edges[i]:max(edges[i + 1], edges[i] + 1)].mean(0) for i in range(N)])
    x = np.array(x0, float); X = [x.copy()]
    for i in range(N): x = rk4_n(x, Ug[i], T / N); X.append(x.copy())
    X = np.array(X); tgt = C[:n] + 2 * np.pi * np.round((X[-1][:n] - C[:n]) / (2 * np.pi)); P = np.concatenate([x0, tgt])
    r = S_(x0=np.concatenate([X.ravel(), Ug.ravel(), [T]]), p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S_.stats()['return_status']
    polish.diag = dict(guess_wmax=round(float(np.abs(X[:, n:]).max()), 2), guess_miss=round(float(miss(X[-1])), 3), T0=round(T, 2), iters=S_.stats().get('iter_count'))
    if st not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): return None, st
    wv = np.array(r['x']).ravel(); return (wv[8 * (N + 1):8 * (N + 1) + 4 * N].reshape(N, 4), wv[-1]), st
def rk4_n(x, u, T, m=10):
    Y = x[None]; U = u[None]
    for _ in range(m): Y = rk4(Y, U, T / m)
    return Y[0]
t0 = time.perf_counter(); back = build_back(S, tau, NB, rho); HF = back_heuristic(S, back, wh=None); t_build = time.perf_counter() - t0; print('build_back NB %d: %.1f с' % (NB, t_build), flush=True)
if HOLD: S.flow_all = fa_hold; S.flow = fl_hold; S.ok = lambda p: np.max(np.abs(p[n:])) <= WM + .02
def work(i):
    t1 = time.perf_counter(); nfs = [int(v) for v in os.environ['NFSEQ'].split(',')] if os.environ.get('NFSEQ') else [NF]; RESMAX = float(os.environ.get('RESMAX', 1.0)); used = []
    for nf in nfs:      # 3в: повтор с бо́льшим NF, пока нет попадания и остаточный промах лучшего кандидата-промаха > RESMAX
        Cn, nh = CN.candidates(S, e.X0[i], back, tau, nf, rho, miss, 8, hfun=HF); used.append(nf)
        ms_ = [c[1] - sum(d for _, d in c[2]) for c in Cn[nh:] if c[2]]; resid = min(ms_) if ms_ else np.inf
        if nh > 0 or resid <= RESMAX: break
    tc = time.perf_counter() - t1
    seen = set(); picks = []
    for part, lab in ((Cn[:nh], 'hit'), (Cn[nh:], 'miss')):
        c = 0
        for h, sc, p in part:
            if not p or sum(d for _, d in p) < float(os.environ.get('MINLEN', 0)): continue
            key = tuple(s for s, _ in p)
            if key in seen: continue
            seen.add(key); picks.append((lab, sc, p)); c += 1
            if c >= KTOP: break
    best = None; log = []
    for lab, sc, p in picks:
        rec, xe = replay(e.X0[i], p); T0 = len(rec) * H; Tv0 = None
        if lab == 'hit': Tv0, _ = common.verify(e, e.X0[i], list(rec), [H] * len(rec))
        t2 = time.perf_counter(); h, st = polish(e.X0[i], rec); Th = None
        dg = dict(polish.diag)
        if h is not None:
            Th, why = common.verify(e, e.X0[i], [u for u in h[0]], [h[1] / len(h[0])] * len(h[0]))
            if Th is None:
                x = np.array(e.X0[i], float); wm = 0.
                for u in h[0]:
                    x = rk4_n(x, u, h[1] / len(h[0]), 20); wm = max(wm, np.abs(x[n:]).max())
                dg.update(why=why, T_ocp=round(float(h[1]), 3), end_miss=round(float(miss(x)), 4), wmax=round(float(wm), 3))
        log.append((lab, round(sc, 3), round(T0, 2), Tv0 and round(Tv0, 3), Th and round(Th, 3), st[:12], round(time.perf_counter() - t2, 1), dg))
        for T in (Tv0, Th):
            if T is not None and (best is None or T < best): best = T
    return dict(i=i, T=best, ncand=len(Cn), hits=nh, nf_used=used, resid=float(resid), t_cand=tc, t_total=time.perf_counter() - t1, log=log)
if __name__ == '__main__':
    with mp.get_context('fork').Pool(min(int(os.environ.get('PROCS', 4)), len(STARTS))) as pool: res = pool.map(work, STARTS, chunksize=1)
    ref = json.load(open(os.path.join(common.DATA, 'm8_ref.json')))
    for r in res: print('старт %d: T %s (T_ref %.3f) cand %d hits %d t_cand %.0f t_total %.0f\n   %s' % (r['i'], r['T'] and round(r['T'], 3), ref[r['i']], r['ncand'], r['hits'], r['t_cand'], r['t_total'], r['log']), flush=True)
    json.dump(dict(res=res, t_build=t_build, NB=NB, NF=NF, KTOP=KTOP), open(os.path.join(common.DATA, 'm8_hold8%s.json' % tag), 'w'), indent=1, default=float)
