"""c14 блок 8 п.2: прямое дерево m8 (tree_fast._tree) с опережающим пакетным ростом. Порядок раскрытия прежний (как tree_vec для ДИ): при промахе кеша достаём B верхних записей кучи,
потоки всех L слоёв для B точек — ОДНИМ numba-вызовом (S.flow_rows, строка = точка×слой, те же 20 шагов RK4), эвристика hfun для всех детей — одним запросом к дереву;
записи возвращаем в кучу. Фильтры (seen по ячейке, ok), ключи и cnt — как в оригинале, на момент настоящего push."""
import heapq, os
import numpy as np
B = int(os.environ.get('VECTREE', 64))
def make(orig):
    def _tree(S, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
        if getattr(S, 'flow_rows', None) is None or S.has_obst or B <= 1: return orig(S, tau, starts, N, rho, sw_w, fwd, hist_on, hfun)
        inv = 1.0 / (rho * np.asarray(S.scale, float)[:S.d]); tt = tau if fwd else -tau; L = S.L
        key = lambda p, s: (s,) + tuple(np.floor(np.asarray(p)[:S.d] * inv).astype(np.int64).tolist())
        seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0; cache = {}; sidx = np.arange(L)
        def kids(Pn):
            m = len(Pn); X = np.repeat(np.asarray(Pn, float), L, axis=0); Sx = np.tile(sidx, m)
            E = S.wrap(S.flow_rows(X, Sx, tt)); hv = np.zeros(len(E))
            if hfun is not None:       # эвристика только для детей, которые ещё могут лечь в кучу (seen только растёт; ok не зависит от состояния)
                KK = np.floor(E[:, :S.d] * inv).astype(np.int64).tolist(); need = [r for r in range(len(E)) if ((r % L,) + tuple(KK[r])) not in seen and S.ok(E[r])]
                if need: hv[need] = np.asarray(hfun(E[need]), float)
            return [(E[r * L:(r + 1) * L], hv[r * L:(r + 1) * L]) for r in range(m)]
        def add(p, s, g, pa, ls):
            k = key(p, s)
            if k in seen: return -1
            seen.add(k); P.append(p); G.append(g); par.append(pa); lay.append(ls); return len(P) - 1
        def children(i):
            k = P[i].tobytes()
            if k not in cache:
                top = [heapq.heappop(pq) for _ in range(min(B - 1, len(pq)))]
                pts = [P[i]] + [t[3] for t in top]; ks = [k] + [t[3].tobytes() for t in top]
                for t in top: heapq.heappush(pq, t)
                new = []; sk = set()
                for j, kk in enumerate(ks):
                    if kk not in cache and kk not in sk: new.append(j); sk.add(kk)
                for j, ch in zip(new, kids([pts[j] for j in new])): cache[ks[j]] = ch
            return cache.pop(k)
        def push(i, ns, ls):
            nonlocal cnt
            E, hv = children(i); KK = np.floor(E[:, :S.d] * inv).astype(np.int64).tolist()
            for s in range(L):
                if ((s,) + tuple(KK[s])) in seen: continue
                e = E[s]
                if not S.ok(e): continue
                n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
                heapq.heappush(pq, (sw_w * n2 + G[i] + tau + hv[s], cnt, s, e.copy(), G[i] + tau, n2, i))
        for st in starts:
            i = add(S.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1)
        while pq and len(P) < N:
            _, _, s, e, g, n2, pa = heapq.heappop(pq)
            i = add(e, s, g, pa, s)
            if i >= 0: push(i, n2, s)
        return P, G, par, lay
    return _tree
def install():
    from src.atlas6 import adaptive_nd, corridor_nd
    o = adaptive_nd._tree; t = make(o); adaptive_nd._tree = t; corridor_nd._tree = t
