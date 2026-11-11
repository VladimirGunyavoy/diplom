"""b26 СТАТИК-КАР: A* агент на графе обратного дерева (слово пользователя 12.11). Запуск как car_val6.py (тот же argv/ENV), плюс:
ASTAR=1 (по умолч.), LIM (лимит раскрытий, 300), BAT (раскрытий за итерацию, 8), EPS (радиус «дошли до узла» в масштабе SC, .15), GC (ячейка закрытого списка = GC*SC, .5), HW (вес h, 1.0)
Идея: из состояния x — A* по 9 примитивам×τ(.2); g=время, h=V̂(q)=min_k W_k+KV·d; закрытый список по хэш-решётке; ребро — зазор ≥ FM (первый шаг + торможение как в жадном);
цель — цель среды или узел дерева (d≤EPS) → g+W; лимит — лучший по g+h. Исполняется 1-й шаг, перепланирование каждые τ."""
import heapq, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from car_val6 import *
import car_val6 as cv
LIM = int(os.environ.get('LIM', 300)); BAT = int(os.environ.get('BAT', 8)); EPS = float(os.environ.get('EPS', .15)); GC = float(os.environ.get('GC', .5)); HW = float(os.environ.get('HW', 1.0)); KVA = float(os.environ.get('KVA', KV))
HYB = int(os.environ.get('HYB', 0))
CELL = np.array(SC) * GC

@nb.njit
def expand_c(X, tau, fm):
    n = X.shape[0]; out = np.empty((n * 9, 4)); okk = np.ones(n * 9, np.bool_); hit = np.full(n * 9, -1.)
    for i in range(n):
        for s in range(9):
            x, y, th, v = X[i, 0], X[i, 1], X[i, 2], X[i, 3]; h = tau / 10
            for k in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1)
                if clear(x, y) < fm or v < VL or v > VH: okk[i * 9 + s] = False
                if (x - GOAL[0]) ** 2 + (y - GOAL[1]) ** 2 <= (GR - .05) ** 2: hit[i * 9 + s] = (k + 1) * h; break
            out[i * 9 + s, 0] = x; out[i * 9 + s, 1] = y; out[i * 9 + s, 2] = th; out[i * 9 + s, 3] = v
    return out, okk, hit

def key(q):
    c = np.floor(q / CELL).astype(np.int64)
    return (int(c[0]), int(c[1]), int(c[2] % int(round(2 * np.pi / CELL[2]))), int(c[3]))

BaseAgent = cv.Agent
class AAgent(cv.Agent):
    def h(self, Q):
        d, j = self.tree.query(self.S.wrap(Q) / self.S.scale, k=min(KN * 3, self.nP)); j = j % self.nP
        c = self.W[j] + KVA * d; a = np.argmin(c, axis=1); r = np.arange(len(Q))
        return c[r, a], d[r, a]
    def act(self, x0):
        if HYB:
            ne = getattr(self, "nesc", 0); a = BaseAgent.act(self, x0)
            if getattr(self, 'nesc', 0) == ne: return a
            self.nastar = getattr(self, 'nastar', 0) + 1
        return self.act_astar(x0)
    def act_astar(self, x0):
        self.xcur = np.asarray(x0, float)[:2]; self.nexp = 0
        sc = np.full(9, 1e9); cl = np.zeros(9); Q = np.zeros((9, 4))
        for s in range(9):
            x, y, th, v = x0; mn = 9.; hit = None; h = TAU / 10
            for k in range(10):
                x, y, th, v = rk(x, y, th, v, ACT[s, 0], ACT[s, 1], h, 1); mn = min(mn, clear(x, y))
                if hit is None and np.hypot(x - GOAL[0], y - GOAL[1]) <= GR - .05: hit = (k + 1) * h
            if HB > 0:
                bx, by, bth, bv = x, y, th, v
                for _ in range(int(HB / .05)): bx, by, bth, bv = rk(bx, by, bth, bv, -AM if bv > 0 else AM, 0., .05, 1); mn = min(mn, clear(bx, by))
            cl[s] = mn; Q[s] = (x, y, th, v)
            if hit is not None and mn >= FM: sc[s] = hit
        ok = (cl >= FM) & (Q[:, 3] >= VL) & (Q[:, 3] <= VH)
        if (sc < 1e9).any(): return int(np.argmin(sc))
        if not ok.any(): return int(np.argmax(cl))
        hq, dq = self.h(Q)
        # открытый список: (f, cnt, g, first, q, term)
        heap = []; best = {}; cnt = 0; bestleaf = (1e18, None)
        for s in range(9):
            if not ok[s]: continue
            g = TAU; f = g + HW * hq[s]; kk = key(Q[s]); best[kk] = g
            heapq.heappush(heap, (f, cnt, g, s, Q[s], dq[s] <= EPS)); cnt += 1
        while heap and self.nexp < LIM:
            batch = []
            while heap and len(batch) < BAT:
                it = heapq.heappop(heap)
                if it[5]: return int(it[3])           # дошли до узла дерева — берём
                batch.append(it)
            self.nexp += len(batch)
            X = np.array([b[4] for b in batch]); out, okk, hit = expand_c(np.ascontiguousarray(X), TAU, FM)
            # цель среды достижима из кого-то
            hm = hit >= 0
            if hm.any():
                gg = np.array([batch[i // 9][2] for i in range(len(out))]) + np.where(hm, hit, 1e9)
                i = int(np.argmin(gg)); return int(batch[i // 9][3])
            idx = np.where(okk)[0]
            if len(idx) == 0: continue
            hh, dd = self.h(out[idx])
            for m, i in enumerate(idx):
                b = batch[i // 9]; g = b[2] + TAU; kk = key(out[i])
                if best.get(kk, 1e18) <= g: continue
                best[kk] = g; f = g + HW * hh[m]
                heapq.heappush(heap, (f, cnt, g, b[3], out[i], dd[m] <= EPS)); cnt += 1
        if not heap: return int(np.argmin(np.where(ok, hq, 1e9)))
        return int(min(heap)[3])

if __name__ == '__main__':
    cv.Agent = AAgent
    st = get_starts(); out = {}
    for ts in range(TS, int(sys.argv[5]) if len(sys.argv) > 5 else TS + 1):
        S = mkS(ts); t0 = time.perf_counter(); back = build_back(S, TAU, NB, RB); tb = time.perf_counter() - t0; print('ts %d: build %.1f с, узлов %d' % (ts, tb, len(back[0])), flush=True)
        ag = AAgent(S, back); res = []; nex = []
        from tqdm import tqdm
        for i, x0, tref in tqdm(st, desc='ts%d' % ts, mininterval=10):
            te = time.perf_counter(); T, mn, ms = episode(ag, x0); res.append((i, T, tref, mn, ms, time.perf_counter() - te))
        ok = [r for r in res if r[1] is not None]; ratio = np.array([r[1] / r[2] for r in ok]); touch = sum(r[3] < 0 for r in res)
        print('nastar %d nesc %d' % (getattr(ag, 'nastar', 0), getattr(ag, 'nesc', 0)), end=' '); print('ASTAR HYB=%d' % HYB + ' LIM=%d BAT=%d EPS=%g GC=%g HW=%g NB=%d KV=%g [%s] ts=%d: дошли %d/%d, касаний %d, T/T_ref мед %.3f p90 %.3f max %.3f среднее %.3f, мс/шаг %.1f, мс/эпизод %.0f' % (
            LIM, BAT, EPS, GC, HW, NB, KV, MODE, ts, len(ok), len(res), touch, np.median(ratio), np.percentile(ratio, 90), ratio.max(), ratio.mean(), 1e3 * np.mean([r[4] for r in res]), 1e3 * np.mean([r[5] for r in res])), flush=True)
        out[ts] = dict(res=res, build=tb, nodes=len(back[0]))
    json.dump(out, open('astar_%s_%d_%d_%g_%g_%d.json' % (MODE, NB, LIM, EPS, GC, TS), 'w'))
