"""v7 п.3 по рекомендации hub-research-7 (`knowledge/research/v7_faces.md`): клетки всех слоёв делят ОБЩИЕ грани — сеть сечений x=lx[i], v=lv[j] (тензорная, шаг любой —
градуированная/адаптивная). Клетка = (проба на грани, слой): поток слоя до ПЕРВОГО пересечения любой линии сети (rk4 с поиском события). V живёт на гранях:
пробы вдоль каждой линии (m на ячейку), линейно между пробами. V(p) = min_k [t_k(p) + V(выход_k(p))] — итерация значений; боковых переходов нет, агент
решает только на гранях и летит точно. 2D системы `Sys2` (нормированные координаты, x может быть периодичен: период 2 — шов x=±1 — линия сети). Цель — ячейка вокруг goal."""
import numpy as np

BIG = 1e3


def lines_uniform(n, periodic):
    """n ячеек на [-1,1], цель (0) в центре ячейки: линии ±h/2·(2i+1) и границы ±1. periodic: нужно нечётное число ячеек между шовными… используем h=2/n, n нечётное."""
    h = 2.0 / n; k = np.arange(-(n // 2) , n // 2 + 1) * h if n % 2 == 1 else None
    if n % 2 == 1: ln = np.arange(n + 1) * h - 1.0 + 0.0; return np.sort(np.unique(np.round(np.concatenate([[-1.0], (np.arange(n // 2 + 1) * 2 + 1) * h / 2, -(np.arange(n // 2 + 1) * 2 + 1) * h / 2, [1.0]]), 12)))
    return np.sort(np.unique(np.round(np.concatenate([[-1.0], (np.arange(n // 2) + .5) * h, -(np.arange(n // 2) + .5) * h, [1.0]]), 12)))


def lines_graded(a, ratio, n_side, periodic=False, d0=None):
    """Градуированные линии: ±a, ±a(1+d), ... шаг растёт в ratio раз (мельче у цели), до границы ±1 (граница — линия)."""
    pos = [a]; d = 2 * a if d0 is None else d0
    while pos[-1] + d < 1.0 - 1e-9 and len(pos) < n_side: pos.append(pos[-1] + d); d *= ratio
    pos = np.array(pos); pos = pos[pos < 1.0 - 1e-9]
    return np.sort(np.concatenate([[-1.0, 1.0], pos, -pos]))


class Faces:
    def __init__(self, S, lx, lv, m=4, periodic_x=False, dtfac=0.08, tmax=6.0, front=False, bigfin=False):
        self.front = front; self.bigfin = bigfin
        self.S, self.lx, self.lv, self.m, self.per = S, np.asarray(lx, float), np.asarray(lv, float), m, periodic_x
        self.dtfac, self.tmax = dtfac, tmax
        sub = lambda ln: np.unique(np.round(np.concatenate([ln[:-1, None] + np.diff(ln)[:, None] * np.arange(m)[None, :] / m]).ravel().tolist() + [ln[-1]], 13))
        self.PX, self.PV = sub(self.lx), sub(self.lv)
        self.n1 = len(self.lv) * len(self.PX); self.n0 = len(self.lx) * len(self.PV)
        X1, V1 = np.meshgrid(self.PX, self.lv); X0, V0 = np.meshgrid(self.lx, self.PV, indexing='ij')
        self.P = np.concatenate([np.stack([X1.ravel(), V1.ravel()], 1), np.stack([X0.ravel(), V0.ravel()], 1)])
        self.kind = np.concatenate([np.ones(self.n1, int), np.zeros(self.n0, int)])         # 1: v-линия (позиция по x), 0: x-линия (позиция по v)
        c = (self.lx[:-1] + self.lx[1:]) / 2; self.ax = float(np.min(np.abs(self.lx[np.abs(self.lx) < 1 - 1e-9])) if np.any(np.abs(self.lx) < 1 - 1e-9) else 1)
        self.av = float(np.min(np.abs(self.lv[np.abs(self.lv) < 1 - 1e-9])))

    def goal_mask(self, x, v): return (np.abs(x) <= self.ax + 1e-9) & (np.abs(v) <= self.av + 1e-9)

    def _cell(self, x, v):
        ix = np.clip(np.searchsorted(self.lx, x, side='right') - 1, 0, len(self.lx) - 2); iv = np.clip(np.searchsorted(self.lv, v, side='right') - 1, 0, len(self.lv) - 2); return ix, iv

    def hit(self, Y, k):
        """Первое пересечение линии сети потоком слоя k из точек Y (N,2). Возвращает t, Yexit (N,2), kind (1: v-линия, 0: x-линия), out (вышел из области / не дошёл за tmax)."""
        S = self.S; Y = np.array(Y, float); N = len(Y); lx, lv = self.lx, self.lv
        if self.per: Y[:, 0] = np.where(Y[:, 0] >= 1 - 1e-11, Y[:, 0] - 2.0, Y[:, 0])
        for c, ln_ in ((0, lx), (1, lv)):                                            # привязка к линии сети (округление 1e-16 ломало ячейку и зеркальную симметрию)
            j = np.clip(np.searchsorted(ln_, Y[:, c]), 1, len(ln_) - 1); jn = np.where(np.abs(ln_[j - 1] - Y[:, c]) < np.abs(ln_[j] - Y[:, c]), j - 1, j)
            Y[:, c] = np.where(np.abs(ln_[jn] - Y[:, c]) < 1e-11, ln_[jn], Y[:, c])
        f0 = S.f(Y, k); eps = 1e-10
        if self.per: Y[:, 0] = np.where((Y[:, 0] <= -1 + 1e-11) & (f0[:, 0] < 0), Y[:, 0] + 2.0, Y[:, 0])
        # начальная ячейка: точка на линии — ячейка по направлению скорости
        def cell0(y, fc, ln):
            r = np.searchsorted(ln, y - 1e-12, side='right') - 1; onl = np.abs(y - ln[np.clip(np.searchsorted(ln, y - 1e-12), 0, len(ln) - 1)]) < 1e-11
            idx = np.searchsorted(ln, y, side='left'); isl = (idx < len(ln)) & (np.abs(ln[np.minimum(idx, len(ln) - 1)] - y) < 1e-11)
            base = np.where(isl, np.where(fc > 0, idx, idx - 1), np.searchsorted(ln, y, side='right') - 1)
            return np.clip(base, 0, len(ln) - 2)
        ix = cell0(Y[:, 0], f0[:, 0], lx); iv = cell0(Y[:, 1], f0[:, 1], lv)
        speed = np.maximum(np.max(np.abs(f0), 1), 1e-3); hmin = min(np.min(np.diff(lx)), np.min(np.diff(lv)))
        dt = self.dtfac * hmin / np.maximum(speed, 0.3)                             # шаг на точку
        t = np.zeros(N); y = Y.copy(); done = np.zeros(N, bool); out = np.zeros(N, bool); Yex = Y.copy(); kind = np.ones(N, int)
        # граница области: линии ±1 в lx/lv — выход через них (x не периодичен) = out; при периодичности x шов — обычная линия
        for step in range(int(self.tmax / np.min(dt)) + 1):
            act = np.nonzero(~done)[0]
            if not len(act): break
            h = dt[act]; ya = y[act]
            k1 = S.f(ya, k); k2 = S.f(ya + h[:, None] / 2 * k1, k); k3 = S.f(ya + h[:, None] / 2 * k2, k); k4 = S.f(ya + h[:, None] * k3, k); yn = ya + h[:, None] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            nx, nv = self._cell(yn[:, 0], yn[:, 1]) if False else (None, None)
            # новые индексы без клипа (за границей — ±)
            jx = np.searchsorted(lx, yn[:, 0], side='right') - 1; jv = np.searchsorted(lv, yn[:, 1], side='right') - 1
            cx = (jx != ix[act]); cv = (jv != iv[act])
            hit = cx | cv
            if hit.any():
                ah = act[hit]; hh = h[hit]; yh = ya[hit]; ynh = yn[hit]
                # доля шага до линии для каждой пересёкшей координаты
                sx = np.full(len(ah), np.inf); sv = np.full(len(ah), np.inf)
                cxh, cvh = cx[hit], cv[hit]
                tgx = np.where(jx[hit] > ix[ah], lx[np.clip(ix[ah] + 1, 0, len(lx) - 1)], lx[ix[ah]]); tgv = np.where(jv[hit] > iv[ah], lv[np.clip(iv[ah] + 1, 0, len(lv) - 1)], lv[iv[ah]])
                dx = ynh[:, 0] - yh[:, 0]; dv = ynh[:, 1] - yh[:, 1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    sx = np.where(cxh, np.clip((tgx - yh[:, 0]) / dx, 0, 1), np.inf); sv = np.where(cvh, np.clip((tgv - yh[:, 1]) / dv, 0, 1), np.inf)
                useX = sx <= sv; s = np.where(useX, sx, sv); s = np.where(np.isfinite(s), s, 1.0)
                # точная доводка: частичный шаг rk4 длиной s·h (затем снап координаты на линию)
                hp = s * hh; a1 = S.f(yh, k); a2 = S.f(yh + hp[:, None] / 2 * a1, k); a3 = S.f(yh + hp[:, None] / 2 * a2, k); a4 = S.f(yh + hp[:, None] * a3, k)
                yp = yh + hp[:, None] / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
                yp[useX, 0] = tgx[useX]; yp[~useX, 1] = tgv[~useX]
                tt = t[ah] + hp; done[ah] = True; Yex[ah] = yp; t[ah] = tt; kind[ah] = np.where(useX, 0, 1)
                # граница области: выход через ±1 по неперiodическому x или по v
                outb = (~useX) & ((np.abs(tgv) >= 1 - 1e-12)) | (useX & (np.abs(tgx) >= 1 - 1e-12) & (not self.per))
                out[ah] = outb
                if self.per:
                    sw = useX & (tgx >= 1 - 1e-12); Yex[ah[sw], 0] = -1.0
            keep = ~hit
            ak = act[keep]; y[ak] = yn[keep]; t[ak] += h[keep]
            if self.per: pass
        out |= ~done; t[~done] = np.inf
        return t, Yex, kind, out

    def _interp(self, e, kind):
        """Индексы 2 проб и вес для точки выхода на линии (kind 1: v-линия, позиция по x; 0: x-линия, позиция по v)."""
        lx, lv, PX, PV = self.lx, self.lv, self.PX, self.PV; N = len(e)
        li = np.where(kind == 1, np.clip(np.rint(np.interp(e[:, 1], lv, np.arange(len(lv)))).astype(int), 0, len(lv) - 1), np.clip(np.rint(np.interp(e[:, 0], lx, np.arange(len(lx)))).astype(int), 0, len(lx) - 1))
        pos = np.where(kind == 1, e[:, 0], e[:, 1]); P = np.where(kind == 1, 0, 0)
        i0 = np.empty(N, int); w = np.empty(N)
        for kd, Pk in ((1, PX), (0, PV)):
            m = kind == kd
            if m.any():
                q = np.clip(np.searchsorted(Pk, pos[m], side='right') - 1, 0, len(Pk) - 2); w[m] = np.clip((pos[m] - Pk[q]) / (Pk[q + 1] - Pk[q]), 0, 1); i0[m] = q
        base = np.where(kind == 1, li * len(PX), self.n1 + li * len(PV))
        return base + i0, base + i0 + 1, w

    def solve_bigfin(self, it=200000, tol=1e-9, B=BIG):
        """Неизвестное = большое КОНЕЧНОЕ B, обычная линейная интерполяция (без строгости fa&fb), Якоби сверху вниз до сходимости; в конце V>B/2 → inf.
        Строгая интерполяция стопорит фронт (hub-research-7: выход между конечной пробой и пробой фронта → inf навсегда)."""
        P = self.P; N = len(P); G = self.goal_mask(P[:, 0], P[:, 1]); V = np.full(N, B); V[G] = 0; E = []
        for k in (0, 1):
            t, e, kd, out = self.hit(P, k); i0, i1, w = self._interp(np.where(out[:, None], 0.0, e), kd); E.append((t, i0, i1, w, out | ~np.isfinite(t)))
        self.E = E; self.G = G
        for n in range(it):
            Vn = np.full(N, B)
            for t, i0, i1, w, out in E: Vn = np.minimum(Vn, np.where(out, B, t + (1 - w) * V[i0] + w * V[i1]))
            Vn[G] = 0; Vn = np.minimum(Vn, B); d = np.max(np.abs(Vn - V)); V = Vn
            if d < tol: break
        self.V = np.where(V > B / 2, np.inf, V); self.iters = n; return self.V

    def solve(self, it=400000, tol=1e-10, log=None):
        if self.bigfin: return self.solve_bigfin(it, tol)
        P = self.P; N = len(P); V = np.full(N, BIG); G = self.goal_mask(P[:, 0], P[:, 1]); V[G] = 0; E = []
        for k in (0, 1):
            t, e, kd, out = self.hit(P, k); i0, i1, w = self._interp(np.where(out[:, None], 0.0, e), kd); E.append((t, i0, i1, w, out))
        self.E = E; self.G = G; dead = E[0][4] & E[1][4] & ~G                  # проба на границе области с исходящим потоком обоих слоёв: V=BIG навсегда, соседа не отравляет
        ch = np.ones(N, bool); ch[:] = G                                      # изменившиеся на прошлом шаге (старт — цель)
        I0 = [e[1] for e in E]; I1 = [e[2] for e in E]
        for n in range(it):
            aff = np.zeros(N, bool)
            for k in range(2): aff |= ch[I0[k]] | ch[I1[k]]
            idx = np.nonzero(aff)[0]
            if not len(idx): break
            vn = V[idx].copy()
            for (t, i0, i1, w, out) in E:
                va, vb, ww = V[i0[idx]], V[i1[idx]], w[idx]; fa, fb = va < BIG / 2, vb < BIG / 2
                vi = np.where(fa & fb, (1 - ww) * va + ww * vb, np.where(fa & (dead[i1[idx]] | (ww < 0.5) & self.front), va, np.where(fb & (dead[i0[idx]] | (ww >= 0.5) & self.front), vb, BIG)))    # на фронте: ближайшая конечная проба
                val = t[idx] + vi; val = np.where(out[idx] | ~np.isfinite(t[idx]), BIG, val); vn = np.minimum(vn, val)
            vn = np.where(G[idx], 0.0, np.where(vn > BIG / 2, BIG, vn)); dch = vn < V[idx] - tol
            ch = np.zeros(N, bool); ch[idx[dch]] = True; V[idx] = vn
            if log and n % 2000 == 0: log(n, int(dch.sum()))
        self.V = np.where(V > BIG / 2, np.inf, V); self.iters = n; return self.V

    def Vat(self, e, kind):
        i0, i1, w = self._interp(e, kind); V = self.V; return np.where(w == 0, V[i0], np.where(w == 1, V[i1], (1 - w) * V[i0] + w * V[i1]))

    def rollout(self, Y0, maxstep=3000):
        """Агент из точек Y0 (N,2): решение только на гранях (первый шаг — до первой грани); T — реальное время; inf — не дошёл."""
        y = np.array(Y0, float); N = len(y); T = np.zeros(N); done = self.goal_mask(y[:, 0], y[:, 1]); sw = np.zeros(N); pk = -np.ones(N, int)
        for _ in range(maxstep):
            act = np.nonzero(~done)[0]
            if not len(act): break
            best = np.full(len(act), np.inf); bk = np.zeros(len(act), int); by = y[act].copy(); bt = np.zeros(len(act))
            for k in (0, 1):
                t, e, kd, out = self.hit(y[act], k); c = np.where(out, np.inf, t + self.Vat(np.where(out[:, None], 0.0, e), kd)); sel = c < best
                best = np.where(sel, c, best); bk = np.where(sel, k, bk); by[sel] = e[sel]; bt = np.where(sel, t, bt)
            dead = ~np.isfinite(best); T[act[dead]] = np.inf; done[act[dead]] = True
            ok = ~dead; ia = act[ok]; sw[ia] += (pk[ia] >= 0) & (bk[ok] != pk[ia]); pk[ia] = bk[ok]; y[ia] = by[ok]; T[ia] += bt[ok]; done[ia] |= self.goal_mask(y[ia, 0], y[ia, 1])
        T[~done] = np.inf; return T, sw
