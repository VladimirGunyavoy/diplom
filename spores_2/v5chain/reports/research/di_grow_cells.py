"""research-13 (слово пользователя 2026-10-05): ДИ, три ОТДЕЛЬНЫХ атласа u ∈ {−1, 0, +1}. Спора = нормальный отрезок (M = 5 узлов), заметаемый потоком
своего управления вперёд и назад («бабочка» = клетка). Новая спора — только в непокрытую точку своего атласа; отрезок расширяется до соседей (≤ RMAX),
клетка растёт по времени, пока наложение на ядра других клеток того же атласа ≤ OV (или до края коробки / TMAX). Цена и агент — лучшая из реализованных
схем (research-7, `v7_spore_cells.md`): V*(узел) = min_k [Δt + V*(φ_k(узел, Δt))], V*(точка) = min по клеткам любого атласа (ядро + гало) билинейно;
агент каждые Δt берёт argmin_k. Запуск: [TMAX=.5] [RMAX=.1] [OV=.2] python3 di_grow_cells.py [png]"""
import numpy as np, sys, os, json, time
sys.path.insert(0, '.')
L, RHO, BIG = 2.5, float(os.environ.get('RHO', .1)), 1e3; US = (-1., 0., 1.); M = 5
DTN = float(os.environ.get('DTN', .06)); RMAX = float(os.environ.get('RMAX', .1)); TMAX = float(os.environ.get('TMAX', 1e9)); OV = float(os.environ.get('OV', .2)); HALO = .1; OVL = float(os.environ.get('OVL', 0.)); ORDER = int(os.environ.get('ORDER', 0)); VMIN = float(os.environ.get('VMIN', .03)); RMIN = float(os.environ.get('RMIN', .01)); NFAIL = int(os.environ.get('NFAIL', 400))
def flow(y, u, t): return np.stack([y[..., 0] + y[..., 1] * t + u * t * t / 2, y[..., 1] + u * t], -1)
def ingoal(y): return (np.abs(y[..., 0]) <= RHO + 1e-9) & (np.abs(y[..., 1]) <= RHO + 1e-9)
def inbox(y): return (np.abs(y[..., 0]) <= L) & (np.abs(y[..., 1]) <= L)
def normal(c, u): n = np.array([-u, c[1]]) if u else np.array([0., 1.]); return n / np.linalg.norm(n)
class Cell:
    def __init__(s, c, u, r): s.c, s.u, s.r = np.array(c, float), u, r; s.n = normal(s.c, u); s.nb = 0; s.nf = 0
    def locate(s, Y):
        """(s, t) точки Y в координатах клетки: точка = flow(c + s·n, u, t). ДИ — формулой (инвариант x − v²/2u)."""
        c, n, u = s.c, s.n, s.u
        with np.errstate(all='ignore'):
            if u == 0: sv = Y[..., 1] - c[1]; t = (Y[..., 0] - c[0]) / Y[..., 1]; return sv, t
            a = -n[1] ** 2 / (2 * u); b = n[0] - c[1] * n[1] / u; cc = c[0] - c[1] ** 2 / (2 * u) - (Y[..., 0] - Y[..., 1] ** 2 / (2 * u))
            if abs(a) < 1e-12: sv = -cc / b
            else: d = np.sqrt(b * b - 4 * a * cc); s1, s2 = (-b + d) / (2 * a), (-b - d) / (2 * a); sv = np.where(np.abs(s1) < np.abs(s2), s1, s2)
            return sv, (Y[..., 1] - c[1] - sv * n[1]) / u
    def inside(s, Y, halo=0.):
        sv, t = s.locate(Y); return np.nan_to_num(np.abs(sv), nan=9.) <= s.r * (1 + halo) + 1e-9, sv, t
    def incell(s, Y, halo=0.):
        m, sv, t = s.inside(Y, halo); return m & (np.nan_to_num(t, nan=9e9) >= -s.nb * DTN - 1e-9) & (np.nan_to_num(t, nan=9e9) <= s.nf * DTN + 1e-9), sv, t
    def nodes(s):
        sn = np.linspace(-(1 + HALO) * s.r, (1 + HALO) * s.r, M); tt = np.arange(-s.nb, s.nf + 1) * DTN
        return flow((s.c + sn[None, :, None] * s.n)[0][None] + 0 * tt[:, None, None], s.u, tt[:, None]), sn, tt   # (nt, M, 2)
def setbb(c):
    """ограничивающий прямоугольник клетки (с гало) — для пространственного индекса."""
    tt = np.arange(-c.nb, c.nf + 1) * DTN; e = np.concatenate([flow(c.c + sg * (1 + HALO) * c.r * c.n, c.u, tt) for sg in (-1., 0., 1.)]); c.bb = (e[:, 0].min() - 1e-6, e[:, 0].max() + 1e-6, e[:, 1].min() - 1e-6, e[:, 1].max() + 1e-6)
def inbb(c, Y): return (Y[:, 0] >= c.bb[0]) & (Y[:, 0] <= c.bb[1]) & (Y[:, 1] >= c.bb[2]) & (Y[:, 1] <= c.bb[3])
def covered(cells, Y):
    m = np.zeros(len(Y), bool)
    for c in cells:
        q = np.flatnonzero(inbb(c, Y))
        if len(q): m[q] |= c.incell(Y[q])[0]
    return m
class Index:
    """сетка ячеек HB: в ячейке — клетки, чей прямоугольник её задевает; покрытие точки проверяется только по ним."""
    HB = .25
    def __init__(s): s.bins = {}
    def add(s, c):
        setbb(c); i0, i1, j0, j1 = [int(np.floor(v / s.HB)) for v in c.bb]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1): s.bins.setdefault((i, j), []).append(c)
    def covered(s, Y):
        Y = np.atleast_2d(Y); m = np.zeros(len(Y), bool); key = np.floor(Y / s.HB).astype(int)
        for k in set(map(tuple, key)):
            q = np.flatnonzero((key[:, 0] == k[0]) & (key[:, 1] == k[1]))
            for c in s.bins.get(k, ()):
                if not m[q].all(): m[q] |= c.incell(Y[q])[0]
        return m
def build_layer(u, rng, log=None):
    cells = []; fails = 0; idx = Index()
    queue = []                                                                                # ORDER=1: кандидаты встык к готовым клеткам (вдоль потока и сбоку)
    while fails < NFAIL:
        p = queue.pop(0) if queue else rng.uniform(-L, L, 2)
        if not inbox(p): continue
        if (u == 0 and abs(p[1]) < VMIN + RMIN) or idx.covered(p[None])[0]: fails += 0 if ORDER and queue else 1; continue
        n = normal(p, u); ext = []
        for sg in (1., -1.):                                                                  # расширяем отрезок до соседа / края / RMAX (в сумме ≤ 2·RMAX)
            ss = sg * np.arange(1, int(2 * RMAX / .01) + 1) * .01; P = p + ss[:, None] * n; bad = ~inbox(P) | idx.covered(P)
            if u == 0: bad |= (np.abs(P[:, 1]) < VMIN) | (np.sign(P[:, 1]) != np.sign(p[1]))
            k = int(np.argmax(bad)) if bad.any() else -1; e = (abs(ss[k - 1]) if k > 0 else 0.) if k >= 0 else abs(ss[-1])
            if k >= 0 and (not inbox(P[k]) or idx.covered(P[k][None])[0]): e += OVL                # боковое пересечение: заходим в соседа (или за границу) на OVL
            ext.append(e)
        lo, hi = -ext[1], ext[0]
        if hi - lo > 2 * RMAX: mid = np.clip(0., lo + RMAX, hi - RMAX); lo, hi = mid - RMAX, mid + RMAX
        if hi - lo < 2 * RMIN: fails += 0 if ORDER and queue else 1; continue
        fails = 0; c = Cell(p + n * (lo + hi) / 2, u, (hi - lo) / 2); sc = np.linspace(-c.r, c.r, M); seg = c.c + sc[:, None] * c.n; tot = M; ov = int(idx.covered(seg).sum())
        for sg in (1, -1):                                                                    # рост по времени: вперёд, затем назад
            i = 0
            while (i + 1) * DTN <= TMAX + 1e-9:
                sl = flow(seg, u, sg * (i + 1) * DTN); inb = inbox(sl)
                if not inb.any(): break                                                      # растём, пока срез целиком не вышел из коробки
                cv = idx.covered(sl) | ~inb; o = int(cv.sum())                                # узлы за коробкой покрывать не нужно
                if (ov + o) / max(tot + M, 10 * M) > OV: break                               # знаменатель не меньше 10 срезов: иначе маленькая клетка в кармане умирает на первом же срезе
                ov += o; tot += M; i += 1
                if cv.all(): break                                                           # полностью покрытый срез оставляем как гало и останавливаемся
            if sg > 0: c.nf = i
            else: c.nb = i
        if c.nb + c.nf == 0: fails += 0 if ORDER and queue else 1; continue               # клетка нулевой длительности ничего не покрывает — не добавляем
        cells.append(c); idx.add(c)
        if ORDER:
            tf, tb_ = (c.nf + .5) * DTN + min(TMAX, 3.) * .9, -(c.nb + .5) * DTN - min(TMAX, 3.) * .9
            queue += [flow(c.c, u, tf), flow(c.c, u, tb_), c.c + 1.9 * c.r * c.n, c.c - 1.9 * c.r * c.n, flow(c.c + 1.9 * c.r * c.n, u, tf), flow(c.c - 1.9 * c.r * c.n, u, tf), flow(c.c + 1.9 * c.r * c.n, u, tb_), flow(c.c - 1.9 * c.r * c.n, u, tb_)]
        if log: log(u, cells)
    return cells
def contact_stats(cells):
    """доля клеток, у которых с каждой из 4 сторон (2 бока, 2 торца) ≥ 80% точек сразу за краем лежат в ядре соседа или за коробкой."""
    idx = Index(); [idx.add(c) for c in cells]; ok = []; sides = []
    for c in cells:
        tt = np.linspace(-c.nb, c.nf, 7) * DTN; sc = np.linspace(-c.r, c.r, M); fr = []
        for P in (flow(c.c + (c.r + .015) * c.n, c.u, tt), flow(c.c - (c.r + .015) * c.n, c.u, tt), flow(c.c + sc[:, None] * c.n, c.u, (c.nf + .5) * DTN), flow(c.c + sc[:, None] * c.n, c.u, -(c.nb + .5) * DTN)):
            fr.append(float((idx.covered(P) | ~inbox(P)).mean()))
        sides.append(fr); ok.append(min(fr) >= .8)
    sides = np.array(sides) if sides else np.zeros((0, 4)); return dict(all4=round(float(np.mean(ok)), 3) if ok else None, side=round(float((sides[:, :2] >= .8).all(1).mean()), 3) if ok else None, end=round(float((sides[:, 2:] >= .8).all(1).mean()), 3) if ok else None)
class Atlas:
    def __init__(s, seed=0, log=None):
        rng = np.random.default_rng(seed); s.layers = [build_layer(u, rng, log) for u in US]; s.cells = [c for l in s.layers for c in l]
        s.off = []; N = 0; P = []
        for c in s.cells: nd, sn, tt = c.nodes(); c.sn, c.tt, c.o = sn, tt, N; s.off.append(N); N += nd.shape[0] * M; P.append(nd.reshape(-1, 2))
        s.P = np.concatenate(P); s.N = N; s.goal = ingoal(s.P); s.V = np.full(N, BIG); s.V[s.goal] = 0.
    def stencils(s, Y):
        """для точек Y: список (iy, idx(·,4), w(·,4)) по всем клеткам, содержащим точку (ядро + гало)."""
        I, IDX, W = [], [], []
        for c in s.cells:
            q = np.flatnonzero(inbb(c, Y))
            if not len(q): continue
            m, sv, t = c.incell(Y[q], HALO); ii = q[m]; sv, t = sv[m], t[m]
            if not len(ii): continue
            fs = (sv + (1 + HALO) * c.r) / (2 * (1 + HALO) * c.r) * (M - 1); ft = t / DTN + c.nb; nt = len(c.tt)
            j = np.clip(np.floor(fs).astype(int), 0, M - 2); a = np.clip(fs - j, 0, 1); k = np.clip(np.floor(ft + 1e-9).astype(int), 0, max(nt - 2, 0)); b = np.clip(ft - k, 0, 1) if nt > 1 else np.zeros(len(ii))
            k1 = np.minimum(k + 1, nt - 1); I.append(ii); IDX.append(c.o + np.stack([k * M + j, k * M + j + 1, k1 * M + j, k1 * M + j + 1], 1)); W.append(np.stack([(1 - b) * (1 - a), (1 - b) * a, b * (1 - a), b * a], 1))
        if not I: return np.zeros(0, int), np.zeros((0, 4), int), np.zeros((0, 4))
        return np.concatenate(I), np.concatenate(IDX), np.concatenate(W)
    def vstar(s, Y):
        I, IDX, W = s.stencils(Y); out = np.full(len(Y), BIG)
        if len(I): v = (W * s.V[IDX]).sum(1); v[((W > 1e-6) & (s.V[IDX] >= BIG / 2)).any(1)] = BIG; np.minimum.at(out, I, v)
        out[ingoal(Y)] = 0.; return out
    def tgoal(s, Y, u, n=8):
        tg = np.full(len(Y), np.inf)
        for i in range(n, 0, -1): tg = np.where(ingoal(flow(Y, u, DTN * i / n)), DTN * i / n, tg)
        return tg
    def solve(s, it=5000):
        E = []; Vg = np.full(s.N, np.inf)
        for u in US:
            Vg = np.minimum(Vg, s.tgoal(s.P, u)); I, IDX, W = s.stencils(flow(s.P, u, DTN)); E.append((I, IDX, W))
        I = np.concatenate([e[0] for e in E]); IDX = np.concatenate([e[1] for e in E]); W = np.concatenate([e[2] for e in E]); o = np.argsort(I, kind='stable'); I, IDX, W = I[o], IDX[o], W[o]
        st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; V = s.V.copy(); V = np.minimum(V, Vg)
        for n in range(it):
            val = DTN + (W * V[IDX]).sum(1); val[((W > 1e-6) & (V[IDX] >= BIG / 2)).any(1)] = BIG; new = V.copy(); new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st)); new[s.goal] = 0.
            d = np.max(np.abs(new - V)); V = new
            if d < 1e-9: break
        s.V = V; s.n_it = n; s.edges = len(I); return s
    def rollout(s, Q, tmax=20.):
        Y = np.array(Q, float); n = len(Y); T = np.zeros(n); done = ingoal(Y); sw = np.zeros(n, int); pu = np.full(n, np.nan); path = [Y.copy()]
        for _ in range(int(tmax / DTN)):
            if done.all(): break
            J = np.stack([np.minimum(s.tgoal(Y, u), DTN + s.vstar(flow(Y, u, DTN))) for u in US], 1); k = J.argmin(1); u = np.array(US)[k]; tg = np.stack([s.tgoal(Y, uu) for uu in US], 1)[np.arange(n), k]
            stuck = J.min(1) >= BIG / 2; h = np.where(np.isfinite(tg), tg, DTN); act = ~done & ~stuck
            sw += act & np.isfinite(pu) & (u != pu); pu = np.where(act, u, pu); Yn = flow(Y, u, h); Y = np.where(act[:, None], Yn, Y); T += np.where(act, h, 0.); T[~done & stuck] = np.inf; done |= ingoal(Y) | stuck; path.append(Y.copy())
        T[~ingoal(Y)] = np.inf; return T, sw, np.array(path)
if __name__ == '__main__':
    from v7_faces_di import tstar_box
    t0 = time.time(); A = Atlas(); tb = time.time() - t0; A.solve(); rng = np.random.default_rng(1)
    X = rng.uniform(-L, L, (20000, 2)); cov = [float(covered(l, X).mean()) for l in A.layers]; cnt = np.zeros(len(X), int)
    for l in A.layers:
        for c in l: cnt += c.incell(X)[0]
    Q = rng.uniform(-1.5, 1.5, (200, 2)); Ts = tstar_box(Q[:, 0], Q[:, 1], RHO); ok = Ts > .05; T, sw, path = A.rollout(Q[ok]); f = np.isfinite(T); r = T[f] / Ts[ok][f]; Vs = A.vstar(Q[ok]); fv = Vs < BIG / 2
    dur = [(c.nb + c.nf) * DTN for c in A.cells]
    print(json.dumps(dict(TMAX=None if TMAX > 1e8 else TMAX, RMAX=RMAX, OV=OV, M=M, DTN=DTN, cells=[len(l) for l in A.layers], cells_total=len(A.cells), nodes=int(A.N), edges=int(A.edges), iters=int(A.n_it),
                          cover_core_by_layer=[round(v, 3) for v in cov], cores_per_point_same_layer_sum=round(float(cnt.mean()), 2), cell_duration_med_max=[round(float(np.median(dur)), 2), round(float(np.max(dur)), 2)],
                          V_big_nodes=round(float((A.V >= BIG / 2).mean()), 3), V_over_Tstar_med=round(float(np.median(Vs[fv] / Ts[ok][fv])), 3), reach=round(float(f.mean()), 3),
                          T_over_Tstar=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3), min=round(float(r.min()), 3)), sw_med=float(np.median(sw[f])), sec_build=round(tb, 1), sec=round(time.time() - t0, 1))), flush=True)
    stats = dict(TMAX=None if TMAX > 1e8 else TMAX, RMAX=RMAX, OV=OV, M=M, DTN=DTN, cells=[len(l) for l in A.layers], nodes=int(A.N), cover=[round(v, 3) for v in cov], reach=round(float(f.mean()), 3), V_over_T=round(float(np.median(Vs[fv] / Ts[ok][fv])), 3),
                 T_mean=round(float(r.mean()), 4), T_med=round(float(np.median(r)), 4), T_max=round(float(r.max()), 3), sw_med=float(np.median(sw[f])), dur_med=round(float(np.median(dur)), 2), dur_max=round(float(np.max(dur)), 2), sec=round(time.time() - t0, 1))
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):                                   # данные для интерактивной страницы
        g = np.linspace(-L, L, 101); GX, GV = np.meshgrid(g, g); VV = A.vstar(np.c_[GX.ravel(), GV.ravel()]); gs = np.linspace(-1.5, 1.5, 13); S = np.array([(x, v) for x in gs for v in gs]); S = S[~ingoal(S)]
        T2, _, p2 = A.rollout(S); Ts2 = tstar_box(S[:, 0], S[:, 1], RHO); paths = []
        for i in range(len(S)):
            pp = p2[:, i]; n_ = int(np.argmax(np.all(np.abs(pp[1:] - pp[:-1]) < 1e-12, axis=1))) + 1 if np.any(np.all(np.abs(pp[1:] - pp[:-1]) < 1e-12, axis=1)) else len(pp)
            paths.append(dict(p=[[round(float(a_), 3), round(float(b_), 3)] for a_, b_ in pp[:n_]], T=None if not np.isfinite(T2[i]) else round(float(T2[i]), 3), Ts=round(float(Ts2[i]), 3)))
        json.dump(dict(stats=stats, cells=[dict(c=[round(float(v), 4) for v in c.c], n=[round(float(v), 4) for v in c.n], r=round(float(c.r), 4), u=int(c.u), nb=int(c.nb), nf=int(c.nf)) for c in A.cells],
                       V=[None if v >= BIG / 2 else round(float(v), 2) for v in VV], paths=paths), open(sys.argv[1], 'w')); print('saved', sys.argv[1]); sys.exit()
    if len(sys.argv) > 1:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        INK, MUT = '#1f2328', '#6b7280'; fig, ax = plt.subplots(2, 2, figsize=(12.5, 12.5), dpi=130); fig.patch.set_facecolor('white'); ax = ax.ravel(); vv = np.linspace(-L, L, 200)
        for a, l, u in zip(ax, A.layers, US):
            for c in l:
                tt = np.arange(-c.nb, c.nf + 1) * DTN; e1 = flow(c.c + c.r * c.n, u, tt[:, None]); e2 = flow(c.c - c.r * c.n, u, tt[:, None]); pg = np.r_[e1, e2[::-1]]
                a.fill(pg[:, 0], pg[:, 1], fc='#2a78d6', alpha=.16, ec='#1f5fbf', lw=.7); sg = c.c + np.linspace(-c.r, c.r, M)[:, None] * c.n; a.plot(sg[:, 0], sg[:, 1], '-', color=INK, lw=1.3); a.plot(sg[:, 0], sg[:, 1], 'o', color=INK, ms=1.8)
            a.set_title('атлас u = %+d: %d спор, покрыто ядрами %.1f%%' % (u, len(l), 100 * cov[US.index(u)]), color=INK, fontsize=11, loc='left')
        a = ax[3]; g = np.linspace(-L, L, 161); GX, GV = np.meshgrid(g, g); VV = A.vstar(np.c_[GX.ravel(), GV.ravel()]).reshape(GX.shape); VV = np.where(VV >= BIG / 2, np.nan, VV)
        im = a.pcolormesh(GX, GV, VV, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', plt.cm.Blues_r(np.linspace(0, .8, 64))), shading='auto'); cb = fig.colorbar(im, ax=a, fraction=.046, pad=.02); cb.set_label('V — время до цели, с', color=INK); cb.outline.set_visible(False)
        a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1, ls=(0, (4, 3)))
        for q in ([-1.3, -.6], [1.4, 1.0], [-.4, 1.3]):
            T1, _, p1 = A.rollout(np.array([q])); a.plot(p1[:, 0, 0], p1[:, 0, 1], color='#d9480f', lw=2); a.plot(*q, 'o', color='#d9480f', ms=6, mec='white', mew=1.2)
            a.annotate('T %.2f (точное %.2f)' % (T1[0], float(tstar_box(np.array([q[0]]), np.array([q[1]]), RHO)[0])), q, (q[0] + .08, q[1] + .16 if q[1] > 0 else q[1] - .26), color=INK, fontsize=9, bbox=dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6))
        a.set_title('склейка: цена V (минимум по трём атласам) и пути агента; пунктир — точная кривая переключения', color=INK, fontsize=10, loc='left')
        for a in ax:
            a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_xlabel('положение x', color=INK); a.set_ylabel('скорость v', color=INK); a.tick_params(colors=MUT, labelsize=8); a.add_patch(plt.Rectangle((-RHO, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='#b42318', lw=1.4))
            for sp in a.spines.values(): sp.set_color('#d0d5dd')
        fig.tight_layout(); fig.savefig(sys.argv[1]); print('saved', sys.argv[1])
