"""Бабочки на дифдрайве в v7 (PLAN п.4, hub-worker-14): ядро — research-9 (`reports/research/butterfly_dd.py`: спора = точка × круг курсов, дуги формулой, окно RW),
здесь: (1) финиш min(TGT,TGTGT) при V ≤ VF; (2) ПРЕПЯТСТВИЯ-диски: споры внутри диска не сеем, дуга годна, только если её точки (NS штук) вне дисков+зазор.
Запуск из spores_2/v7: RW=.9 python3 src/cells7/butterfly_dd.py N [obst]. Эталон: dd_ref_60.npy (60 стартов rng(1) ±2, θ)."""
import sys, os, time, json
RES = os.path.abspath('../v5chain/reports/research'); sys.path.insert(0, RES); os.chdir(RES)
import numpy as np
import butterfly_dd as M
from butterfly_dd import wrap, BIG, move
from dd_rhombus_ref import tgt, tgtgt
OBS = [(1.0, -1.0, .5), (-1.0, 0.8, .5)]; MARGIN = .05; NS = 16


def arc_free(Y, PH, L, obs=OBS, margin=MARGIN, ns=NS):
    """(n,) bool: дуга из поз Y c (φ, L) не задевает диски (центр, r) с зазором; точки дуги — ns долей."""
    ok = np.ones(len(Y), bool)
    for f in np.linspace(0, 1, ns + 1)[1:]:
        ph = PH * f; Lf = L * f; th = Y[:, 2]; small = np.abs(ph) < 1e-9; R = np.where(small, 0., Lf / np.where(small, 1., ph))
        x = np.where(small, Y[:, 0] + Lf * np.cos(th), Y[:, 0] + R * (np.sin(th + ph) - np.sin(th)))
        y = np.where(small, Y[:, 1] + Lf * np.sin(th), Y[:, 1] - R * (np.cos(th + ph) - np.cos(th)))
        for cx, cy, r in obs: ok &= np.hypot(x - cx, y - cy) > r + margin
    return ok


class ButterflyDDObs(M.ButterflyDD):
    def __init__(s, N=1500, obs=None, **kw):
        s.obs = obs or []; super().__init__(N=N, **kw)
        if s.obs:                                                      # споры в дисках не нужны (кроме цели в (0,0), она свободна)
            free = np.ones(len(s.C), bool)
            for cx, cy, r in s.obs: free &= np.hypot(s.C[:, 0] - cx, s.C[:, 1] - cy) > r + MARGIN
            free[0] = True; s.C = s.C[free]; s.K = len(s.C); s.V = np.full((s.K, s.m), BIG); s.V[0] = np.abs(wrap(s.th)); s.tree = M.cKDTree(s.C)

    def cand(s, Y, own=None):
        iy, k, T, thl, PH, L = super().cand(Y, own)
        if s.obs and len(iy):
            ok = arc_free(Y[iy], PH, L, s.obs); iy, k, T, thl, PH, L = iy[ok], k[ok], T[ok], thl[ok], PH[ok], L[ok]
        return iy, k, T, thl, PH, L

    def solve(s, it=2000, tol=1e-7, rows=1500, pchunk=1500000):
        """Беллман с КОМПАКТНЫМИ парами (int32/float32/int8, hub-v5chain-worker-15): пары ~ N², в float64 N=2500 не лез в 1.5 ГБ. Кандидаты — кусками строк Y, итерации — кусками пар по границам узлов."""
        m = s.m; Y = np.c_[np.repeat(s.C, m, 0), np.tile(s.th, s.K)]; own = np.repeat(np.arange(s.K), m); I, Kk, Tt, Jj, Aa = [], [], [], [], []
        for a0 in range(0, len(Y), rows):
            iy, k, T, thl, _, _ = s.cand(Y[a0:a0 + rows], own[a0:a0 + rows]); f = (thl % (2 * np.pi)) / (2 * np.pi / m); j = np.floor(f).astype(int) % m; a = f - np.floor(f); o = np.argsort(iy, kind='stable')
            I.append((iy[o] + a0).astype(np.int32)); Kk.append(k[o].astype(np.int32)); Tt.append(T[o].astype(np.float32)); Jj.append(j[o].astype(np.int8)); Aa.append(a[o].astype(np.float32))
        iy, k, T, j, a = (np.concatenate(v) for v in (I, Kk, Tt, Jj, Aa)); del I, Kk, Tt, Jj, Aa; s.npairs = len(T)
        st = np.flatnonzero(np.r_[True, iy[1:] != iy[:-1]]); nodes = iy[st]; bnd = [0]
        while bnd[-1] < len(st): bnd.append(min(len(st), int(np.searchsorted(st, st[bnd[-1]] + pchunk, 'right'))) if bnd[-1] < len(st) - 1 else len(st))
        dj = np.abs(wrap(s.th[None, :] - s.th[:, None])); V = s.V.reshape(-1).copy()
        for n in range(it):
            new = V.copy()
            for sa, sb in zip(bnd[:-1], bnd[1:]):
                if sa >= sb: continue
                p0, p1 = st[sa], (st[sb] if sb < len(st) else len(T)); kk = k[p0:p1].astype(np.int64); jj = j[p0:p1].astype(np.int64); aa = a[p0:p1].astype(np.float64); j1 = (jj + 1) % m
                V0, V1 = V[kk * m + jj], V[kk * m + j1]; val = T[p0:p1] + (1 - aa) * V0 + aa * V1; val[((V0 >= BIG / 2) & (aa < 1 - 1e-9)) | ((V1 >= BIG / 2) & (aa > 1e-9))] = BIG
                nd = nodes[sa:sb]; new[nd] = np.minimum(V[nd], np.minimum.reduceat(val, st[sa:sb] - p0))
            W = new.reshape(s.K, m); W = np.minimum(W, (W[:, None, :] + dj[None]).min(2)); new = W.reshape(-1); new[0] = 0.
            d = np.max(np.abs(np.minimum(new, BIG) - np.minimum(V, BIG))); V = new
            if d < tol: break
        s.V = V.reshape(s.K, m); s.n_it = n; return s


def run(B, q, VF, smax=300):
    """агент + финиш min(TGT,TGTGT) при V ≤ VF (только без препятствий — пути финиша их не знают)."""
    y = np.array(q, float); t = 0.; at = -1; segs = 0
    for _ in range(smax):
        if np.hypot(y[0], y[1]) < 1e-9 and abs(wrap(y[2])) < 1e-9: return t, segs
        J, act = B.best(y, at)
        if J <= VF and not B.obs:
            a, b = tgt(*y), tgtgt(*y, starts=20)
            if min(a, b) <= J + 1e-6: return t + min(a, b), segs + (3 if a <= b else 5)
        if act is None or J >= BIG / 2: return np.inf, segs
        kind, phi, L, k = act
        if kind == 'turn': y = np.array([y[0], y[1], y[2] + phi]); t += abs(phi)
        else:
            y0 = y.copy(); y = move(y, phi, L); y[:2] = B.C[k]; t += abs(L) + abs(phi)
            if B.obs and not arc_free(y0[None], np.array([phi]), np.array([L]), B.obs, 0.)[0]: return np.inf, segs     # проверка столкновения (зазор 0)
        at = k; segs += 1
    return np.inf, segs


if __name__ == '__main__':
    N = int(sys.argv[1]); obst = len(sys.argv) > 2 and sys.argv[2] == 'obst'; ref = np.load('dd_ref_60.npy'); rng = np.random.default_rng(1)
    Q = np.c_[rng.uniform(-2, 2, (60, 2)), rng.uniform(-np.pi, np.pi, 60)]
    if obst: keep = np.array([all(np.hypot(q[0] - cx, q[1] - cy) > r + .1 for cx, cy, r in OBS) for q in Q]); Q, ref = Q[keep], ref[keep]
    t0 = time.time(); B = ButterflyDDObs(N=N, obs=OBS if obst else []).solve(); print(json.dumps(dict(N=N, RW=M.RW, obst=obst, спор=B.K, пар=B.npairs, итер=B.n_it, сек=round(time.time() - t0))), flush=True)
    for VF in ((0.,) if obst else (0., 3.)):
        R = [run(B, q, VF) for q in Q]; T = np.array([a for a, _ in R]); S = np.array([b for _, b in R]); f = np.isfinite(T); r = T[f] / ref[f]
        print(json.dumps(dict(VF=VF, n=len(Q), reach=round(float(f.mean()), 3), T_over_ref_free=dict(mean=round(float(r.mean()), 3), med=round(float(np.median(r)), 3), p90=round(float(np.percentile(r, 90)), 3), max=round(float(r.max()), 3)),
                              segs_med=float(np.median(S[f])), сек=round(time.time() - t0)), ensure_ascii=False), flush=True)
