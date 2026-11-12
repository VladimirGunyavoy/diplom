"""b26 ДИНАМИК-КАР: A* агент (слово пользователя 12.11). Запуск как car_dyn6.py (`runpc.sh car_dyn_astar.py NB 8 test|blind|sel TS K0 K1`, те же ENV) + ASTAR-ENV как в car_astar.py (LIM, BAT, GC, HW, KVA).
Первый шаг — как в жадном (прогноз дисков + торможение); дальше A* по 9 примитивам×τ: g=время, h=HW·V̂ (V̂ по W этого шага — с пометкой облаком), зазор каждого ребра — по ПРОГНОЗУ дисков на абсолютное время g..g+τ (центр p0+w t, облако κσ(t)) ≥ FM.
WARMA=1 — тёплый старт: состояния прошлого плана (сдвинутые на шаг, проверенные по свежему прогнозу) кладутся в открытый список с их g (верхняя граница)."""
import heapq, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_dyn6 as cd
from car_dyn6 import np, nb, time, rk, ACT, TAU, FM, HR, XL, GOAL, GR, VL, VH, AM, HB, KN, cand
LIM = int(os.environ.get('LIM', 300)); BAT = int(os.environ.get('BAT', 8)); GC = float(os.environ.get('GC', .5)); HW = float(os.environ.get('HW', 1.5)); KVA = float(os.environ.get('KVA', .3)); PREB = int(os.environ.get('PREB', 1)); HYB = int(os.environ.get('HYB', 0)); WARMA = int(os.environ.get('WARMA', 0)); HZN = float(os.environ.get('HZN', .4))
CELL = np.array(cd.SC) * GC; PER = int(round(2 * np.pi / CELL[2]))

@nb.njit
def expand_dyn(X, G, P0, Wv, kap, q, smax, tau, fm, hz):
    n = X.shape[0]; out = np.empty((n * 9, 4)); okk = np.ones(n * 9, np.bool_); hit = np.full(n * 9, -1.)
    for i in range(n):
        for s in range(9):
            x, y, th, v = X[i, 0], X[i, 1], X[i, 2], X[i, 3]; h = tau / 10
            for k in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1); t = G[i] + (k + 1) * h
                if v < VL or v > VH or XL - max(abs(x), abs(y)) < fm: okk[i * 9 + s] = False
                sg = min(np.sqrt(q * t ** 3 / 3.), smax)
                for d in range(3):
                    if t > hz: break
                    cx = min(max(P0[d, 0] + Wv[d, 0] * t, -(XL - HR)), XL - HR); cy = min(max(P0[d, 1] + Wv[d, 1] * t, -(XL - HR)), XL - HR)
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - HR - kap * sg < fm: okk[i * 9 + s] = False
                if (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: hit[i * 9 + s] = (k + 1) * h; break
            out[i * 9 + s, 0] = x; out[i * 9 + s, 1] = y; out[i * 9 + s, 2] = th; out[i * 9 + s, 3] = v
    return out, okk, hit
def key(q):
    c = np.floor(q / CELL).astype(np.int64); return (int(c[0]), int(c[1]), int(c[2] % PER), int(c[3]))

BASE_ACT = cd.AgentDyn.act
class AgentDynA(cd.AgentDyn):
    def hh(self, Q):
        d, j = self.tree.query(self.S.wrap(Q) / self.S.scale, k=min(KN * 3, self.nP)); c = self.W[j % self.nP] + KVA * d; return c.min(axis=1)
    def act(self, x, P0, Wv):
        if HYB:
            ne = getattr(self, 'nesc', 0); a = BASE_ACT(self, x, P0, Wv)
            if getattr(self, 'nesc', 0) == ne: return a
            self.nastar = getattr(self, 'nastar', 0) + 1
        return self.act_astar(x, P0, Wv)
    def act_astar(self, x, P0, Wv):
        self.xcur = np.asarray(x, float)[:2]; self.plan(x, P0, Wv); self.nexp = 0
        sc = np.full(9, 1e9); cl = np.zeros(9); Q = np.zeros((9, 4))
        for s in range(9):
            xe, ye, the, ve, mn, hit = cand(x[0], x[1], x[2], x[3], ACT[s, 0], ACT[s, 1], P0, Wv, cd.KAPC, cd.QC, cd.SMAX, TAU, HB); cl[s] = mn; Q[s] = (xe, ye, the, ve)
            if hit >= 0 and mn >= FM: sc[s] = hit
        ok = (cl >= FM) & (Q[:, 3] >= VL) & (Q[:, 3] <= VH)
        d0, j0 = self.tree.query(self.S.wrap(np.asarray(x, float)[None]) / self.S.scale, k=min(KN * 3, self.nP)); self.Wx = float(np.min(self.W[j0 % self.nP] + cd.KV * d0))
        if (sc < 1e9).any(): self.plan_acts = []; return int(np.argmin(sc))
        if not ok.any(): self.plan_acts = []; return int(np.argmax(cl))
        if PREB:   # b26: корни A* — состояние ПОСЛЕ шага τ (до торможения); cand возвращает состояние после торможения (V̂ жадного — в точке остановки)
            for s_ in range(9): Q[s_] = rk(x[0], x[1], x[2], x[3], ACT[s_, 0], ACT[s_, 1], TAU / 10, 10)
        hq = self.hh(Q); heap = []; best = {}; cnt = 0; leaf = {}
        for s in range(9):
            if not ok[s]: continue
            kk = key(Q[s]); best[kk] = TAU; heapq.heappush(heap, (TAU + HW * hq[s], cnt, TAU, s, Q[s], (s,))); cnt += 1
        self.warm_used = 0
        self.warm_n = 0
        pa = getattr(self, 'plan_acts', None)
        if WARMA and pa and len(pa) > 1:   # тёплый старт: прошлый план без 1-го действия, по свежему прогнозу
            cur = np.array([x], float); g = 0.; first = pa[1]
            for a in pa[1:]:
                out, okk, hit = expand_dyn(np.ascontiguousarray(cur), np.array([g]), np.ascontiguousarray(P0), np.ascontiguousarray(Wv), cd.KAPC, cd.QC, cd.SMAX, TAU, FM, HZN)
                if not okk[a] or hit[a] >= 0 or not ok[first]: break
                g += TAU; cur = out[a:a + 1]; kk = key(cur[0])
                if best.get(kk, 1e18) <= g: continue
                best[kk] = g; self.warm_n += 1; heapq.heappush(heap, (g + HW * float(self.hh(cur)[0]), cnt, g, first, cur[0].copy(), tuple(pa[1:1 + self.warm_n]))); cnt += 1
        while heap and self.nexp < LIM:
            batch = []
            while heap and len(batch) < BAT: batch.append(heapq.heappop(heap))
            self.nexp += len(batch); X = np.ascontiguousarray(np.array([b[4] for b in batch])); G = np.array([b[2] for b in batch])
            out, okk, hit = expand_dyn(X, G, np.ascontiguousarray(P0), np.ascontiguousarray(Wv), cd.KAPC, cd.QC, cd.SMAX, TAU, FM, HZN)
            hm = (hit >= 0) & okk
            if hm.any():
                gg = np.where(hm, np.repeat(G, 9) + hit, 1e9); i = int(np.argmin(gg)); self.plan_acts = list(batch[i // 9][5]) + [i % 9]; return int(batch[i // 9][3])
            idx = np.where(okk)[0]
            if len(idx) == 0: continue
            hh = self.hh(out[idx])
            for m, i in enumerate(idx):
                b = batch[i // 9]; g = b[2] + TAU; kk = key(out[i])
                if best.get(kk, 1e18) <= g: continue
                best[kk] = g; heapq.heappush(heap, (g + HW * hh[m], cnt, g, b[3], out[i], b[5] + (int(idx[m] % 9),))); cnt += 1
        self.nit = getattr(self, 'nit', 0); self.nw = getattr(self, 'nw', 0) + (self.warm_n > 0); self.ns = getattr(self, 'ns', 0) + 1
        if not heap: return int(np.argmin(np.where(ok, hq, 1e9)))
        it_ = min(heap); self.plan_acts = list(it_[5]); return int(it_[3])

_ep = cd.episode
def _episode(ag, *a, **k): ag.plan_acts = []; return _ep(ag, *a, **k)
cd.episode = _episode
if __name__ == '__main__':
    cd.AgentDyn = AgentDynA
    src = open(cd.__file__).read(); body = src.split("if __name__ == '__main__':\n", 1)[1]
    body = '\n'.join(l[4:] if l.startswith('    ') else l for l in body.split('\n'))
    g = cd.__dict__; g['__name__'] = '__main__'; exec(compile(body.replace("if MODE == 'tail': tail_main(); sys.exit(0)", ''), 'dynmain', 'exec'), g)
