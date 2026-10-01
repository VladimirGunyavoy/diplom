"""research hub-research-7: v7 п.3 без бокового перехода (слово пользователя 2026-10-01).
Сеть сечений = линии сетки x=x_i, v=v_j (DI, физ. единицы, [-L,L]^2, шаг h). Клетка = (грань, слой): поток слоя от грани до первой линии.
V живёт только на гранях (m проб на ребро, линейно вдоль ребра). Переключение слоя — только на гранях. Цель — граница квадрата сетки вокруг 0.
Rollout: агент из старта летит точно (аналитика DI), решение только при пересечении линии; T — реальное время пути."""
import numpy as np, sys, json, time

def tstar_pt(x0, v0, xg, vg):
    """Точное мин. время DI |u|<=1 из (x0,v0) в (xg,vg) (одно переключение)."""
    best = np.full(np.broadcast(x0, xg).shape, np.inf)
    a = (v0**2 + vg**2) / 2 + (xg - x0)                      # +- : vs = +sqrt(a), vs >= max(v0,vg)
    vs = np.sqrt(np.maximum(a, 0)); ok = (a >= 0) & (vs >= v0 - 1e-12) & (vs >= vg - 1e-12)
    best = np.where(ok, np.minimum(best, 2 * vs - v0 - vg), best)
    b = (v0**2 + vg**2) / 2 - (xg - x0)                      # -+ : vs = -sqrt(b), vs <= min(v0,vg)
    vs = -np.sqrt(np.maximum(b, 0)); ok = (b >= 0) & (vs <= v0 + 1e-12) & (vs <= vg + 1e-12)
    return np.where(ok, np.minimum(best, v0 + vg - 2 * vs), best)

def tstar_box(x0, v0, rho, n=2001):
    g = np.linspace(-rho, rho, n); B = np.concatenate([np.stack([g, np.full(n, -rho)], 1), np.stack([g, np.full(n, rho)], 1),
                                                       np.stack([np.full(n, -rho), g], 1), np.stack([np.full(n, rho), g], 1)])
    x0, v0 = np.atleast_1d(np.asarray(x0, float)), np.atleast_1d(np.asarray(v0, float))
    inside = (np.abs(x0) <= rho) & (np.abs(v0) <= rho); T = np.empty(len(x0))
    for a in range(0, len(x0), 500):                                   # чанками: целиком на h=.05 было 13 ГБ (OOM 2026-10-01)
        T[a:a + 500] = np.min(tstar_pt(x0[a:a + 500, None], v0[a:a + 500, None], B[:, 0], B[:, 1]), -1)
    return np.where(inside, 0.0, T)

class Net:
    def __init__(s, L=4.0, h=0.2, m=4, fam='xv'):
        s.L, s.h, s.m, s.fam = L, h, m, fam
        n = int(round(2 * L / h)); s.lines = -L + h * np.arange(n + 1)            # линии в обеих семьях; 0 — середина ячейки (n чётно → сдвиг)
        s.lines = s.lines + h / 2; s.lines = s.lines[s.lines < L]                  # сдвиг: 0 в центре квадрата цели [-h/2,h/2]^2
        s.rho = h / 2
        P, kind, line, pos = [], [], [], []                                        # пробы: на v-линиях (kind 1) по x; на x-линиях (kind 0) по v
        sub = np.concatenate([s.lines[:-1, None] + h * np.arange(m)[None, :] / m]).ravel(); sub = np.append(sub, s.lines[-1])
        s.sub = sub
        for j, vl in enumerate(s.lines):
            P += [(x, vl) for x in sub]; kind += [1] * len(sub)
        if 'x' in fam:
            for i, xl in enumerate(s.lines):
                P += [(xl, v) for v in sub]; kind += [0] * len(sub)
        s.P = np.array(P); s.kind = np.array(kind); s.ns = len(sub)

    def hit(s, x, v, u, first=False):
        """Первое пересечение линии сети из (x,v) при управлении u (векторно). Возвращает t, x1, v1, kind(1: v-линия, 0: x-линия)."""
        h, ln = s.h, s.lines
        # v-линии: следующая по направлению u строго впереди
        k = (v - ln[0]) / h; kn = np.floor(k + 1e-9) + 1 if u > 0 else np.ceil(k - 1e-9) - 1
        vt = ln[0] + kn * h; tv = np.abs(vt - v)
        t, kind = tv.copy(), np.ones_like(tv)
        if 'x' in s.fam:
            tx = np.full_like(x, np.inf)
            kx = (x - ln[0]) / h
            for dirn in (+1, -1):                                                  # ближайшая линия впереди по x в каждом направлении
                kn2 = np.floor(kx + 1e-9) + 1 if dirn > 0 else np.ceil(kx - 1e-9) - 1
                xt = ln[0] + kn2 * h; d = xt - x                                   # решить v t + u t^2/2 = d, наименьший t>0
                disc = v * v + 2 * u * d
                with np.errstate(invalid='ignore', divide='ignore'):
                    sq = np.sqrt(np.maximum(disc, 0))
                    r1 = (-v + sq) / u; r2 = (-v - sq) / u
                for r in (r1, r2):
                    ok = (disc >= 0) & (r > 1e-12)
                    tx = np.where(ok & (r < tx), r, tx)
            sel = tx < t; t = np.where(sel, tx, t); kind = np.where(sel, 0, kind)
        x1 = x + v * t + u * t * t / 2; v1 = v + u * t
        return t, x1, v1, kind

    def interp_idx(s, x1, v1, kind):
        """Индексы двух проб и вес для точки на линии (kind 1: v-линия, позиция по x; 0: x-линия, позиция по v)."""
        h, ln, ns = s.h, s.lines, s.ns
        a = np.where(kind == 1, v1, x1); b = np.where(kind == 1, x1, v1)
        li = np.rint((a - ln[0]) / h).astype(int)
        q = (b - s.sub[0]) / (h / s.m); qi = np.clip(np.floor(q).astype(int), 0, ns - 2); w = np.clip(q - qi, 0, 1)
        out = (li < 0) | (li >= len(ln)) | (b < s.sub[0] - 1e-9) | (b > s.sub[-1] + 1e-9)
        li = np.clip(li, 0, len(ln) - 1)
        base = np.where(kind == 1, li * ns, len(ln) * ns + li * ns)
        return base + qi, base + qi + 1, w, out

    def goal_mask(s, x, v): return (np.abs(x) <= s.rho + 1e-9) & (np.abs(v) <= s.rho + 1e-9)

    def solve(s, it=4000):
        P = s.P; N = len(P); BIG = 1e3; V = np.full(N, BIG); G = s.goal_mask(P[:, 0], P[:, 1]); V[G] = 0
        T, I0, I1, W, OUT = [], [], [], [], []
        for u in (1.0, -1.0):
            t, x1, v1, kd = s.hit(P[:, 0], P[:, 1], u); i0, i1, w, out = s.interp_idx(x1, v1, kd)
            T.append(t); I0.append(i0); I1.append(i1); W.append(w); OUT.append(out)
        for k in range(it):
            Vn = V.copy()
            for t, i0, i1, w, out in zip(T, I0, I1, W, OUT):
                val = t + (1 - w) * V[i0] + w * V[i1]; val[out] = BIG       # неизвестное = большое конечное (иначе строгая интерполяция гасит фронт)
                Vn = np.minimum(Vn, val)
            Vn[G] = 0
            Vn = np.minimum(Vn, BIG)
            if np.max(np.abs(Vn - V)) < 1e-12: V = Vn; break
            V = Vn
        V = np.where(V > BIG / 2, np.inf, V); s.V = V; s.iters = k; return V

    def Vat(s, x1, v1, kind):
        i0, i1, w, out = s.interp_idx(x1, v1, kind); V = s.V
        val = np.where(w == 0, V[i0], np.where(w == 1, V[i1], (1 - w) * V[i0] + w * V[i1])); return np.where(out, np.inf, val)

    def rollout(s, x0, v0, maxstep=4000):
        """Агент: решение только на линиях; первый шаг из произвольной точки — до первой линии. Возвращает реальное время до цели (inf — не дошёл)."""
        x, v = np.array(x0, float), np.array(v0, float); T = np.zeros_like(x); done = s.goal_mask(x, v); sw = np.zeros_like(x)
        prev_u = np.zeros_like(x)
        for _ in range(maxstep):
            if done.all(): break
            best = np.full_like(x, np.inf); bu = np.zeros_like(x); bx = x.copy(); bv = v.copy(); bt = np.zeros_like(x)
            for u in (1.0, -1.0):
                t, x1, v1, kd = s.hit(x, v, u); c = t + s.Vat(x1, v1, kd)
                # вход в квадрат цели по пути (грань цели — линия сети, так что достаточно проверки точки прихода)
                sel = c < best; best = np.where(sel, c, best); bu = np.where(sel, u, bu); bx = np.where(sel, x1, bx); bv = np.where(sel, v1, bv); bt = np.where(sel, t, bt)
            act = ~done & np.isfinite(best)
            sw += act & (prev_u != 0) & (bu != prev_u); prev_u = np.where(act, bu, prev_u)
            x = np.where(act, bx, x); v = np.where(act, bv, v); T = np.where(act, T + bt, T)
            done = done | (act & s.goal_mask(x, v)); dead = ~done & ~np.isfinite(best); T[dead] = np.inf; done = done | dead
        T[~done] = np.inf
        return T, sw

if __name__ == '__main__':
    rng = np.random.default_rng(0); Q = rng.uniform(-1.5, 1.5, (2000, 2)); Q2 = rng.uniform(-3, 3, (2000, 2)); res = []
    for fam in ('v', 'xv'):
        for h in (0.4, 0.2, 0.1, 0.05):
            for m in (1, 4):
                t0 = time.time(); S = Net(h=h, m=m, fam=fam); V = S.solve(); tsol = time.time() - t0
                fin = np.isfinite(V); ii = rng.choice(len(S.P), min(len(S.P), 5000), replace=False)   # эталон по подвыборке узлов
                Ts = tstar_box(S.P[ii, 0], S.P[ii, 1], S.rho); ok = fin[ii] & (Ts > .05); rv = V[ii][ok] / Ts[ok]
                out = dict(fam=fam, h=h, m=m, samples=len(S.P), finite=float(fin.mean()), iters=S.iters, sec=round(tsol, 2),
                           V_Tstar_nodes=[round(float(np.median(rv)), 4), round(float(rv.min()), 4), round(float(np.percentile(rv, 99)), 4)])
                for nm, QQ in (('roll15', Q), ('roll3', Q2)):
                    Tr, sw = S.rollout(QQ[:, 0], QQ[:, 1]); Tq = tstar_box(QQ[:, 0], QQ[:, 1], S.rho); ok = np.isfinite(Tr) & (Tq > 0.05)
                    r = Tr[ok] / Tq[ok]; d = Tr[ok] - Tq[ok]
                    if not ok.any(): out[nm] = dict(reach=0.0); continue
                    out[nm] = dict(reach=float(np.isfinite(Tr).mean()), mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), max=round(float(r.max()), 3),
                                       dT_mean=round(float(d.mean()), 4), dT_max=round(float(d.max()), 3), min_ratio=round(float(r.min()), 4), sw_med=float(np.median(sw[ok])))
                print(json.dumps(out), flush=True); res.append(out)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else 'v7_faces_di.json', 'w'), indent=1)
