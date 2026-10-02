"""research hub-v5chain-research-8: характеристики ПМП маятника с АДАПТИВНОЙ вставкой (к pend_characteristics.py: покрытие 77%, дыры от расхождения
у седла). Параметр θ ∈ [0, 8) — обход границы цели: углы (веер нормалей) и грани по очереди; p(θ) = ν·нормаль, ν из H = 0. Раунд: интегрировать
назад новые характеристики до S, записи каждые rec; если у соседей по θ (обе годны) в какой-то записи расстояние > dmax (норм. φ, ω) — вставить
середину. V_char(q) = min времени по записям в радиусе r. Проверка: покрытие, V_char/V_grid, V_char против реального J агента P2."""
import numpy as np, json, sys, time
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from pend_cost_grid import Grid, wrap, UM
R0 = .1
def F(z):
    x, w, px, pw = z; u = -UM * np.sign(pw)
    return np.stack([w, np.sin(x) + u, -pw * np.cos(x), -px])
CORNERS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
def terminal(th):
    """θ: целая часть 2k — угол k (веер от нормали предыдущей грани к следующей), 2k+1 — грань между углами k и k+1."""
    th = np.asarray(th, float) % 8; seg = np.floor(th).astype(int); f = th - seg; k = seg // 2
    cx = np.array([c[0] for c in CORNERS])[k] * R0; cw = np.array([c[1] for c in CORNERS])[k] * R0
    nx_ = np.array([c[0] for c in CORNERS])[(k + 1) % 4] * R0; nw_ = np.array([c[1] for c in CORNERS])[(k + 1) % 4] * R0
    face = seg % 2 == 1
    x = np.where(face, cx + f * (nx_ - cx), cx); w = np.where(face, cw + f * (nw_ - cw), cw)
    # нормаль грани k→k+1: перпендикуляр к ребру наружу
    ex, ew = nx_ - cx, nw_ - cw; fnx, fnw = ew, -ex; sgn = np.sign(fnx * cx + fnw * cw); fnx, fnw = fnx * sgn, fnw * sgn; nn = np.hypot(fnx, fnw); fnx, fnw = fnx / nn, fnw / nn
    # веер в углу k: от нормали грани (k−1→k) к нормали грани (k→k+1)
    px_, pw_ = np.array([c[0] for c in CORNERS])[(k - 1) % 4] * R0, np.array([c[1] for c in CORNERS])[(k - 1) % 4] * R0
    gx, gw = cx - px_, cw - pw_; pnx, pnw = gw, -gx; sg2 = np.sign(pnx * cx + pnw * cw); pnx, pnw = pnx * sg2, pnw * sg2; nn2 = np.hypot(pnx, pnw); pnx, pnw = pnx / nn2, pnw / nn2
    a0 = np.arctan2(pnw, pnx); a1 = np.arctan2(fnw, fnx); da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi; ang = a0 + f * da
    nx = np.where(face, fnx, np.cos(ang)); nw = np.where(face, fnw, np.sin(ang))
    su = np.where(np.abs(nw) > 1e-12, np.sign(nw), np.sign(nx)); u = -UM * su
    fn = nx * w + nw * (np.sin(x) + u); ok = fn < -1e-9; nu = np.where(ok, -1 / np.where(ok, fn, -1), np.nan)
    pw = nu * nw; pw = np.where(np.abs(pw) < 1e-15, su * 1e-12, pw)
    return np.stack([x, w, nu * nx, pw]), ok
def integrate(th, S=14., h=.004, rec=.02):
    z, ok = terminal(th); every = int(round(rec / h)); out = [np.stack([wrap(z[0]), z[1]], 1)]
    for i in range(int(S / h)):
        k1 = -F(z); k2 = -F(z + h / 2 * k1); k3 = -F(z + h / 2 * k2); k4 = -F(z + h * k3); z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % every == 0: out.append(np.stack([wrap(z[0]), z[1]], 1))
    return np.stack(out, 1).astype(np.float32), ok                                                      # (n, R, 2)
if __name__ == '__main__':
    dmax = float(sys.argv[1]) if len(sys.argv) > 1 else .05; cap = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    t0 = time.time(); th = np.linspace(0, 8, 2001)[:-1]; P, ok = integrate(th); TH, PP, OK = th, P, ok
    for rnd in range(20):
        o = np.argsort(TH); TH, PP, OK = TH[o], PP[o], OK[o]
        a, b = np.arange(len(TH)), (np.arange(len(TH)) + 1) % len(TH)
        dx = np.abs(wrap(PP[a, :, 0] - PP[b, :, 0])); dw = np.abs(PP[a, :, 1] - PP[b, :, 1]); inside = (np.abs(PP[a, :, 1]) < 4) & (np.abs(PP[b, :, 1]) < 4)
        gap = OK[a] & OK[b] & (np.where(inside, np.hypot(dx, dw), 0).max(1) > dmax)
        dth = (TH[b] - TH[a]) % 8; gap &= dth > 1e-7
        if not gap.any() or len(TH) >= cap: break
        new = (TH[a[gap]] + dth[gap] / 2) % 8
        if len(TH) + len(new) > cap: new = new[np.random.default_rng(rnd).choice(len(new), cap - len(TH), replace=False)]
        Pn, okn = integrate(new); TH, PP, OK = np.r_[TH, new], np.concatenate([PP, Pn]), np.r_[OK, okn]
        print('round', rnd, 'gaps', int(gap.sum()), 'total', len(TH), 'sec', round(time.time() - t0), flush=True)
    R = PP.shape[1]; Tm = np.broadcast_to(np.arange(R) * .02, PP.shape[:2]); Pf = PP[OK].reshape(-1, 2); Tf = Tm[OK].reshape(-1); keep = np.abs(Pf[:, 1]) <= 4
    Pf, Tf = Pf[keep], Tf[keep]; tree = cKDTree(Pf)
    def Vchar(Q, r=.03, k=64):
        d, i = tree.query(Q, k=k, distance_upper_bound=r); T = np.where(np.isfinite(d), Tf[np.minimum(i, len(Tf) - 1)], np.inf); return T.min(1)
    S = Grid(); S.solve((-1., 1.), 0.)
    rng = np.random.default_rng(5); Q = np.stack([rng.uniform(-np.pi, np.pi, 3000), rng.uniform(-2, 2, 3000)], 1)
    vc, vg = Vchar(Q), S.Vq(Q[:, 0], Q[:, 1]); okq = np.isfinite(vc) & (vg < 100) & (vg > .05); r = vc[okq] / vg[okq]
    out = dict(dmax=dmax, chars=int(OK.sum()), samples=len(Pf), covered=round(float(np.isfinite(vc).mean()), 3),
               Vchar_over_Vgrid=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4), p05=round(float(np.quantile(r, .05)), 3), p95=round(float(np.quantile(r, .95)), 3)))
    A = np.load('exact_switch_pend_P0_P2_P30.5.npy'); rq = np.random.default_rng(0); Q0 = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)[:40]
    v0 = Vchar(Q0); m = np.isfinite(v0); rr = v0[m] / A[1][m]
    out['Vchar_over_J_P2'] = dict(n=int(m.sum()), mean=round(float(rr.mean()), 4), min=round(float(rr.min()), 4), max=round(float(rr.max()), 4))
    out['Vgrid_over_J_P2'] = round(float((S.Vq(Q0[:, 0], Q0[:, 1]) / A[1]).mean()), 4); out['sec'] = round(time.time() - t0)
    print(json.dumps(out, ensure_ascii=False), flush=True)
