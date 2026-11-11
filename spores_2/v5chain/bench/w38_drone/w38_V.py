"""w38: рецепт круга на дроне. Обратное дерево (узлы p, Gb) -> V=время на графе узлов (итерация ценностей, рёбра: 4 слоя × шаг H, Эйлер .02)
-> жадный агент по V̂ (min_k V[j]+KV*d) + фильтр безопасности по модели (зазор вдоль шага >= PAD+MARG) -> план проверяется verify (их Эйлер).
Запуск: SEED=.. STARTSEED=.. NSTARTS=100 [env дрона как dr.sh] python w38_V.py NB tag   (импортирует spores_drone_e)"""
import os, sys, time, json
nbb, tag = sys.argv[1], sys.argv[2]
sys.argv = ['spores_drone_e.py', nbb, '1600', '20']
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import spores_drone_e as D
from src.atlas6.adaptive_nd import tree_query
S, back, X0 = D.S, D.back, D.X0
H = float(os.environ.get('HV', .1)); KV = float(os.environ.get('KV', 1.0)); KN = int(os.environ.get('KNV', 8)); MARG = float(os.environ.get('MARG', .02)); NIT = int(os.environ.get('NIT', 300))
Pb, Gb, tree = np.asarray(back[0]), np.asarray(back[1]), back[5]; nP = len(Pb)
def ceil_ok(P): return ~((P[..., 1] > D.YT - .005) & (np.abs(P[..., 0]) > D.HW - .0) & (P[..., 1] > D.YT - .005)) | (np.abs(P[..., 0]) <= D.HW)
def okv(P): return ceil_ok(P) & (np.abs(P[..., 0]) <= D.XM) & (P[..., 1] >= 0) & (P[..., 1] <= D.YG + .1) & (np.abs(P[..., 3]) <= D.VM) & (np.abs(P[..., 4]) <= D.VM) & (np.abs(P[..., 5]) <= D.WM)
t0 = time.perf_counter(); Q = np.zeros((nP, 4, 6)); VAL = np.zeros((nP, 4), bool)
for a in range(4):
    mid = D.flow(Pb, a, H / 2); q = D.flow(Pb, a, H); Q[:, a] = q
    VAL[:, a] = ~D.blocked(mid[..., :2]) & ~D.blocked(q[..., :2]) & okv(q) & ceil_ok(mid)
GQ = D.ing(Q); Qw = S.wrap(Q.reshape(-1, 6))
d, j = tree.query(Qw / S.scale, k=min(KN * 3, len(tree.data)), workers=-1); J = (j % nP).astype(np.int64); Dd = d
t1 = time.perf_counter(); ING = D.ing(Pb); V = Gb.copy(); V[ING] = 0.
for it in range(NIT):
    Vq = np.min(V[J] + KV * Dd, axis=1).reshape(nP, 4); Vq = np.where(GQ, 0., Vq); c = np.where(VAL, H + Vq, 1e3); Vn = np.minimum(c.min(1), 60.); Vn[ING] = 0.
    dl = np.abs(Vn - V).max(); V = Vn
    if dl < 1e-3: break
t2 = time.perf_counter(); print('V: рёбра %.1f с, vi %.1f с (it %d, d %.1e); V мед %.2f макс(<60) %.2f; доля узлов V>=60: %.3f' % (t1 - t0, t2 - t1, it, dl, np.median(V), V[V < 60].max(), (V >= 60).mean()), flush=True)
UP = D.US_
DEP = int(os.environ.get('DEP', 3)); NS = int(round(H / .02)); PADA = D.PAD + MARG
def step_all(X, a):   # X (n,6) -> Xn (n,6), ok (n,) : шаг H слоем a, зазор в NS Эйлер-точках
    P = X; ok = np.ones(len(X), bool)
    for _ in range(NS):
        P = D.flow(P, a, .02); ok &= (D.clr(P[:, :2]) - PADA >= 0) & okv(P)
    return P, ok
def agent(x0, tmax=40.):
    x = np.array(x0, float); seq = []; dts = []; T = 0.; lat = []; why = 'timeout'
    while T < tmax:
        if D.ing(x): why = 'goal'; break
        ta = time.perf_counter(); X = x[None]; root = np.array([-1]); alive = np.ones(1, bool); best = {}
        for lv in range(DEP):
            Xs, Rs = [], []
            for a in range(4):
                Xn, ok = step_all(X, a); r = np.full(len(X), a) if lv == 0 else root
                g_ = D.ing(Xn) & ok
                for ri in np.unique(r[g_]): best[ri] = min(best.get(ri, 1e9), (lv + 1) * H)
                keep = ok & ~g_
                if keep.any(): Xs.append(Xn[keep]); Rs.append(r[keep])
            if not Xs: break
            X = np.vstack(Xs); root = np.concatenate(Rs)
        else:
            Vh = np.min(V[(tree.query(S.wrap(X) / S.scale, k=KN * 3)[1]) % nP] + KV * tree.query(S.wrap(X) / S.scale, k=KN * 3)[0], axis=1) + DEP * H
            for ri in np.unique(root): best[ri] = min(best.get(ri, 1e9), float(Vh[root == ri].min()))
        lat.append(time.perf_counter() - ta)
        if not best: why = 'stuck'; break
        a = min(best, key=best.get); x = D.flow(x, a, H) if NS == 0 else step_all(x[None], a)[0][0]
        if seq and seq[-1] == a: dts[-1] += H
        else: seq.append(a); dts.append(H)
        T += H
    return seq, dts, why, lat
res = []; ST = int(os.environ.get('SEED', 0))
for i in range(len(X0)):
    _d, _j = tree.query(S.wrap(X0[i][None]) / S.scale, k=KN * 3); V0 = float(np.min(V[_j[0] % nP] + KV * _d[0]))
    if os.environ.get('VONLY'): res.append(dict(i=i, V0=V0, T=None, why='vonly', touch=False, ms=0., ms_max=0., nseg=0)); continue
    seq, dts, why, lat = agent(X0[i]); tref = 0.
    if os.environ.get('REFINE') and seq and why == 'goal':
        from src.atlas6.corridor_nd import refine
        t_r = time.perf_counter(); gx = lambda x: np.array([(D.HW - D.GM) ** 2 - x[0] ** 2, x[1] - D.YG - D.GM])
        T_, sq_, dd_, ok_ = refine(D.flow, X0[i], [(int(q), float(t)) for q, t in zip(seq, dts)], gx, tries=int(os.environ.get('RTRIES', 1)), clear=lambda P: D.clr(P) - D.PAD, wlim=None); tref = time.perf_counter() - t_r
        if ok_:
            e = np.round(np.cumsum(dd_) / .02) * .02; dq = np.diff(np.concatenate([[0.], e])); k_ = dq > 1e-9; sq2 = [q for q, kk in zip(sq_, k_) if kk]; dq = dq[k_]
            Tv2, w2, t2_ = D.verify(X0[i], sq2, list(dq))
            if Tv2 is not None and not t2_: seq, dts = sq2, list(dq)
    Tv, why2, tch = (None, why, False) if not seq else D.verify(X0[i], seq, dts)
    if D.HARD and tch: Tv = None; why2 = 'touch'
    res.append(dict(i=i, V0=V0, T=Tv, why=why2, touch=bool(tch), ms=1e3 * float(np.mean(lat)) if lat else 0., ms_max=1e3 * float(np.max(lat)) if lat else 0., ms_ref=1e3 * tref, nseg=len(seq)))
ok = [r for r in res if r['T'] is not None]; print(json.dumps(dict(tag=tag, n=len(res), reached=len(ok), touches=sum(r['touch'] for r in res), T_med=float(np.median([r['T'] for r in ok])) if ok else None, ms=float(np.mean([r['ms'] for r in res])), why={w: sum(r['why'] == w for r in res) for w in set(r['why'] for r in res)})), flush=True)
json.dump(dict(per_start=res, V_info=dict(it=it)), open(os.path.join(HERE, 'data', 'w38V_%s.json' % tag), 'w'))
