"""BIG: общие функции стенда маятника u .3 (строка «по T», VARIANT=56). Импорт из bench/ на aida: sys.path.insert(0,'.'); import big_lib.
make_atlas() -> Atlas; sub(A, keep_idx) -> Atlas из подмножества клеток (finish+solve отдельно); evalA(A) -> dict (дошли, T/T_ref)."""
import os, sys, copy, time
os.environ.setdefault('VARIANT', '56');
sys.path.insert(0, os.getcwd())
import spores_pend as SP
import numpy as np, common
G = SP.G; e = common.make('pend'); Q = e.X0
def make_atlas(): return G.Atlas()
def cell_vol(c): return 2 * c.r * c.G.shape[0] * G.DTN
def sub(A, keep):
    keep = set(int(i) for i in keep); A2 = copy.copy(A); A2.layers = []; ci = 0
    for l in A.layers:
        A2.layers.append([c for k, c in enumerate(l, ci) if k in keep]); ci += len(l)
    if hasattr(A2, 'qx'): delattr(A2, 'qx')
    A2.finish(); return A2
def evalA(A, per=False):
    Ts = []; fin = []
    for i in range(len(Q)):
        T, rec = SP.rollout_rec(A, Q[i:i+1]); us = [[r[0][0]] for r in rec if r[2][0]]; dts = [float(r[1][0]) for r in rec if r[2][0]]
        Tv, why = common.verify(e, Q[i], us, dts); Ts.append(Tv)
    s = common.summarize(e, Ts); s['T'] = [None if (t is None or not np.isfinite(t)) else float(t) for t in Ts]; return s
def solve(A, thr=.5):
    """solve + авто-страховка (слово research-30): если доля узлов с V<BIG < thr — пересчёт с GOALH=2 (затравка V у цели), пометка в лог и в A.goalh_used."""
    t = time.perf_counter(); A.solve(); A.goalh_used = False; fin = float((A.V < G.BIG / 2).mean())
    if fin < thr and G.GOALH <= 0:
        print('GOALH-СТРАХОВКА СРАБОТАЛА: V<BIG %.1f%% < %.0f%% -> пересчёт с GOALH=2' % (100 * fin, 100 * thr), flush=True)
        G.GOALH = 2.; A.V = np.full(A.N, G.BIG); A.V[A.goal] = 0.; A.solve(); G.GOALH = 0.; A.goalh_used = True
    return time.perf_counter() - t
def eval100(A):
    """100 внутренних стартов (starts_ref: rng(0), 100 точек, отфильтрованы по эталону pend_ref_T) — как status.json r18: T агента (rollout) / T_ref. mean/med/p90/max, дошли."""
    Qs, ref = G.starts_ref(); T, sw, path = A.rollout(Qs); ok = np.isfinite(T); r = T[ok] / ref[ok]
    return dict(n=len(Qs), reached=float(ok.mean()), mean=float(r.mean()) if ok.any() else None, med=float(np.median(r)) if ok.any() else None, p90=float(np.percentile(r, 90)) if ok.any() else None, max=float(r.max()) if ok.any() else None), path
