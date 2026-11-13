"""c13/c14 блок 6 п.2: _tree с опережающим пакетным ростом. Порядок раскрытия узлов ТОТ ЖЕ (куча/ключи/дедуп не тронуты): при промахе кеша достаём B верхних
записей кучи, считаем детей для всех одним вызовом flow на слой (B точек), возвращаем записи в кучу (записи неизменны, cnt уникален → порядок прежний),
кладём детей в кеш по байтам точки. Только для S без push_batch (ДИ, общий flow(P,s,t))."""
import heapq, os
import numpy as np
B = int(os.environ.get('VEC', 64))


def make(orig):
    def _tree(S, tau, starts, N, rho, sw_w, fwd, hist_on=True, hfun=None):
        if getattr(S, 'push_batch', None) is not None or B <= 1:
            return orig(S, tau, starts, N, rho, sw_w, fwd, hist_on, hfun)
        sc = rho * S.scale; tt = tau if fwd else -tau
        def key(p, s): return (s,) + tuple(int(np.floor(p[k] / (rho * S.scale[k]))) for k in range(S.d))
        seen = set(); P = []; G = []; par = []; lay = []; pq = []; cnt = 0; cache = {}
        def kids(Pn):
            res = [[] for _ in range(len(Pn))]
            for s in range(S.L):
                E = S.wrap(S.flow(Pn, s, tt)); Mid = S.flow(Pn, s, tt / 2)
                bl = S.blocked(E) | S.blocked(Mid)
                okv = S.ok_batch(E) if hasattr(S, 'ok_batch') else None
                for r in range(len(Pn)):
                    if (okv[r] if okv is not None else S.ok(E[r])) and not bl[r]: res[r].append((s, E[r].copy()))
            return res
        def add(p, s, g, pa, ls):
            k = key(p, s)
            if k in seen: return -1
            seen.add(k); P.append(p); G.append(g); par.append(pa); lay.append(ls); return len(P) - 1
        def push(i, ns, ls, ch):
            nonlocal cnt
            hh = hfun(np.array([e for _, e in ch])) if (hfun is not None and ch) else [0.0] * len(ch)
            for (s, e), hv in zip(ch, hh):
                n2 = ns + (1 if (ls >= 0 and s != ls) else 0); cnt += 1
                heapq.heappush(pq, (sw_w * n2 + G[i] + tau + hv, cnt, s, e, G[i] + tau, n2, i))
        def children(i):
            k = P[i].tobytes()
            if k not in cache:
                top = [heapq.heappop(pq) for _ in range(min(B - 1, len(pq)))]
                pts = [P[i]] + [t[3] for t in top]; ks = [k] + [t[3].tobytes() for t in top]
                for t in top: heapq.heappush(pq, t)
                new = []; seen_k = set()
                for j, kk in enumerate(ks):
                    if kk not in cache and kk not in seen_k: new.append(j); seen_k.add(kk)
                for j, ch in zip(new, kids(np.array([pts[j] for j in new]))): cache[ks[j]] = ch
            return cache.pop(k)
        for st in starts:
            i = add(S.wrap(st), -1, 0.0, -1, -1); push(i, 0, -1, children(i))
        while pq and len(P) < N:
            _, _, s, e, g, n2, pa = heapq.heappop(pq)
            i = add(e, s, g, pa, s)
            if i >= 0: push(i, n2, s, children(i))
        return P, G, par, lay
    return _tree


def install(*mods):
    for m in mods:
        if not hasattr(m, '_orig_tree'): m._orig_tree = m._tree
        m._tree = make(m._orig_tree)
