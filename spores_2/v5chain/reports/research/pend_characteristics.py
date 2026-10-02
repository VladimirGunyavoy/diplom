"""research hub-v5chain-research-8: споры-характеристики ПМП для маятника (к идее пользователя «сечение клетки = криволинейный базис»):
пары (x, p) интегрируются НАЗАД от цели, u = −UM·sign(p_ω), переключение — ноль p_ω (точно, на подшаге). Множество точек переключения =
кривые переключения (без V и без сетки); V вдоль характеристики = время до цели (точно, без диффузии). Проверка: V_char против сетки HJB
и против реально достигнутого времени агента P2 (exact_switch_pend.py). Цель |φ|,|ω| ≤ .1; p на границе = ν·нормаль (на углах — веер), ν из H = 0."""
import numpy as np, json, sys, time
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from pend_cost_grid import Grid, wrap, UM
R0 = .1
def F(z):
    x, w, px, pw = z; u = -UM * np.sign(pw)
    return np.stack([w, np.sin(x) + u, -pw * np.cos(x), -px])
def terminal(n_edge=1500, n_corner=150):
    Z = []
    g = np.linspace(-R0, R0, n_edge)
    for (nx, nw) in ((1, 0), (-1, 0), (0, 1), (0, -1)):                       # грани: x на грани, p ∥ нормали
        x = np.where(nx != 0, nx * R0, g); w = np.where(nw != 0, nw * R0, g); Z.append((x, w, np.full_like(g, nx, dtype=float), np.full_like(g, nw, dtype=float)))
    a = np.linspace(0, np.pi / 2, n_corner)
    for sx in (1, -1):
        for sw in (1, -1):                                                     # углы: веер нормалей
            Z.append((np.full_like(a, sx * R0), np.full_like(a, sw * R0), sx * np.cos(a), sw * np.sin(a)))
    x, w, nx, nw = [np.concatenate(c) for c in zip(*Z)]
    # u на конце: −UM·sign(p_ω); при p_ω = 0 (грань φ) — знак, который p_ω получит, идя назад: dp_ω/ds = +p_φ
    su = np.where(nw != 0, np.sign(nw), np.sign(nx)); u = -UM * su
    fn = nx * w + nw * (np.sin(x) + u); ok = fn < -1e-9                       # поток входит в цель
    nu = -1 / fn[ok]; pw = nu * nw[ok]; pw = np.where(pw == 0, su[ok] * 1e-12, pw)
    return np.stack([x[ok], w[ok], nu * nx[ok], pw])
def sweep(S=14., h=.004, rec=.02):
    z = terminal(); n = z.shape[1]; P, Tm, SWP = [], [], []
    every = int(round(rec / h)); s = 0.
    for i in range(int(S / h)):
        k1 = -F(z); k2 = -F(z + h / 2 * k1); k3 = -F(z + h / 2 * k2); k4 = -F(z + h * k3); zn = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        flip = np.sign(zn[3]) != np.sign(z[3])
        if flip.any(): SWP.append(np.stack([zn[0, flip], zn[1, flip], np.full(flip.sum(), s + h)], 1))
        z = zn; s += h
        if i % every == 0: P.append(np.stack([wrap(z[0]), z[1]], 1)); Tm.append(np.full(n, s))
    return np.concatenate(P), np.concatenate(Tm), np.concatenate(SWP), n
if __name__ == '__main__':
    t0 = time.time(); P, Tm, SWP, n = sweep(); keep = np.abs(P[:, 1]) <= 4
    P, Tm = P[keep], Tm[keep]; print('characteristics', n, 'samples', len(P), 'switch pts', len(SWP), 'sec', round(time.time() - t0), flush=True)
    tree = cKDTree(np.c_[P[:, 0], P[:, 1]])
    def Vchar(Q, r=.03, k=64):
        d, i = tree.query(Q, k=k, distance_upper_bound=r); T = np.where(np.isfinite(d), Tm[np.minimum(i, len(Tm) - 1)], np.inf); return T.min(1)
    S = Grid(); S.solve((-1., 1.), 0.)
    rng = np.random.default_rng(5); Q = np.stack([rng.uniform(-np.pi, np.pi, 3000), rng.uniform(-2, 2, 3000)], 1)
    vc, vg = Vchar(Q), S.Vq(Q[:, 0], Q[:, 1]); ok = np.isfinite(vc) & (vg < 100)
    r = vc[ok] / vg[ok]
    out = dict(covered=round(float(np.isfinite(vc).mean()), 3), Vchar_over_Vgrid=dict(mean=round(float(r.mean()), 4), med=round(float(np.median(r)), 4),
               p05=round(float(np.quantile(r, .05)), 3), p95=round(float(np.quantile(r, .95)), 3)))
    try:                                                                       # против реального времени агента P2 (40 случайных стартов)
        A = np.load('exact_switch_pend_P0_P2_P30.5.npy'); rq = np.random.default_rng(0); Q0 = np.stack([rq.uniform(-np.pi, np.pi, 100), rq.uniform(-2, 2, 100)], 1)[:40]
        v0 = Vchar(Q0); m = np.isfinite(v0); out['Vchar_over_J_P2'] = dict(n=int(m.sum()), mean=round(float((v0[m] / A[1][m]).mean()), 4), max=round(float((v0[m] / A[1][m]).max()), 4),
                                                                           min=round(float((v0[m] / A[1][m]).min()), 4))
        out['Vgrid_over_J_P2'] = round(float((S.Vq(Q0[:, 0], Q0[:, 1]) / A[1]).mean()), 4)
    except Exception as e: out['err'] = str(e)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    np.save('pend_char_switch.npy', SWP)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sw = SWP[SWP[:, 2] < 10]; ax[0].scatter(wrap(sw[:, 0]), sw[:, 1], s=.2, c=sw[:, 2], cmap='viridis'); ax[0].set_title('точки переключения характеристик (цвет — время до цели)')
    gx = np.linspace(-np.pi, np.pi, 361); gw = np.linspace(-2.5, 2.5, 201); GX, GW = np.meshgrid(gx, gw)
    c = [.03 + S.Vq(*__import__('pend_cost_grid').step(GX.ravel(), GW.ravel(), u * UM, .03)) for u in (1., -1.)]
    ax[1].imshow((c[0] <= c[1]).reshape(GX.shape), origin='lower', extent=[-np.pi, np.pi, -2.5, 2.5], aspect='auto', cmap='coolwarm', alpha=.6)
    ax[1].scatter(wrap(sw[:, 0]), sw[:, 1], s=.2, c='k'); ax[1].set_title('слой argmin по сетке V (+UM красный) и кривые характеристик')
    for a in ax: a.set_xlabel('φ'); a.set_ylabel('ω'); a.set_ylim(-2.5, 2.5)
    plt.tight_layout(); plt.savefig('figs/pend_char_switch.png', dpi=110)
