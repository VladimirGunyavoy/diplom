"""research-22, эксп. 16: картинка di4 для сводки всех систем — траектории агента на двух фазовых плоскостях + T/эталон по стартам. L1600, SBLAY 1, финиш shoot_pol VF 1."""
import sys, os, pickle, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, os.path.expanduser('~/spore_v5/r22/13_di4_fast_finish'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import growN as G, fin
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from scipy.optimize import minimize
import finish_gen as FGm
def shoot_pol(y, f, US, RHOV, wrapy, tmax, NA=3, inits=2):
    T0, tp = fin.shoot_grid(y, f, US, RHOV, wrapy, tmax, NA)
    if tp is None: return T0, tp
    R = .9 * np.asarray(RHOV); best = (T0, tp); n = len(tp)
    def end(d):
        z = np.array(y, float)
        for k, dt in zip(tp, d): z = FGm.flow(z, US[k], max(dt, 0.), f)
        return wrapy(z)
    cons = [{"type": "ineq", "fun": lambda d: R - np.abs(end(d))}]
    for d0 in (np.full(n, T0 / n), np.r_[T0 * .6, np.full(n - 1, T0 * .4 / max(n - 1, 1))][:n]):
        r = minimize(lambda d: d.sum(), d0, method="SLSQP", bounds=[(0, tmax)] * n, constraints=cons, options=dict(maxiter=60, ftol=1e-7))
        if r.success and np.all(np.abs(end(r.x)) <= np.asarray(RHOV)) and r.x.sum() < best[0]: best = (float(r.x.sum()), tp)
    return best
d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d_['layers']; A.finish(); r = np.load('di4_ref_60_rho35.npy'); Q = r[:, :4]; TR = r[:, 6]; A.V = np.load(os.environ['VFILE']); G.FG.shoot = shoot_pol
T, sw, path = A.rollout(Q); ok = np.isfinite(T); stop = np.abs(np.c_[Q[:, 0] + Q[:, 2] * abs(Q[:, 2]) / 2, Q[:, 1] + Q[:, 3] * abs(Q[:, 3]) / 2]).max(1); inf_ = stop <= 2.5; q = T / TR
np.savez('data/di4_L1600_agent.npz', Q=Q, T=T, TR=TR, sw=sw, path=path, infield=inf_)
B, O, INK, MUT, GRID, BG = '#2a78d6', '#eb6834', '#1a1a19', '#6b6a63', '#e4e3dc', '#fcfcfb'
fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG, gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for k, (ix, iv) in enumerate(((0, 2), (1, 3))):
    a = ax[k]; a.set_facecolor(BG)
    for i in range(60):
        c = B if ok[i] else O; P = path[:, i]; a.plot(P[:, ix], P[:, iv], color=c, lw=.9 if ok[i] else 1.4, alpha=.55 if ok[i] else .95, zorder=2 if ok[i] else 3); a.plot(Q[i, ix], Q[i, iv], 'o', color=c, ms=4.5, mec=BG, mew=.8, zorder=4)
    a.add_patch(plt.Rectangle((-.35, -.35), .7, .7, fill=False, ec=INK, lw=1.4, zorder=5)); a.annotate('цель ±0.35', (.35, .35), xytext=(8, 8), textcoords='offset points', color=INK, fontsize=9)
    a.axvline(2.5, color=MUT, lw=.8, ls=(0, (4, 3))); a.axvline(-2.5, color=MUT, lw=.8, ls=(0, (4, 3))); a.set_xlim(-3.3, 3.3); a.set_ylim(-2.4, 2.4); a.grid(color=GRID, lw=.6); a.set_axisbelow(True)
    a.set_xlabel('x%d' % (k + 1), color=INK); a.set_ylabel('v%d' % (k + 1), color=INK); a.set_title('ось %d: фазовая плоскость (x%d, v%d)' % (k + 1, k + 1, k + 1), color=INK, fontsize=11, loc='left')
    for s_ in a.spines.values(): s_.set_color(GRID)
    a.tick_params(colors=MUT)
ax[0].plot([], [], color=B, lw=1.4, label='дошёл (%d)' % ok.sum()); ax[0].plot([], [], color=O, lw=1.4, label='не дошёл (%d): тормозной путь за полем ±2.5' % (~ok).sum()); ax[0].legend(loc='lower left', fontsize=8.5, frameon=False, labelcolor=INK)
a = ax[2]; a.set_facecolor(BG); o = np.argsort(TR); xs = np.arange(60); m = ok[o]
a.vlines(xs[m], 1., q[o][m], color=B, lw=1.6, zorder=2); a.plot(xs[m], q[o][m], 'o', color=B, ms=5, mec=BG, mew=.8, zorder=3); a.plot(xs[~m], np.full((~m).sum(), 1.0), 'x', color=O, ms=7, mew=1.8, zorder=3)
a.axhline(1., color=INK, lw=1); a.set_ylim(.97, max(1.3, np.nanmax(q[ok]) + .03)); a.grid(axis='y', color=GRID, lw=.6); a.set_axisbelow(True); a.set_xlabel('старты по возрастанию эталонного времени (%.1f … %.1f с)' % (TR.min(), TR.max()), color=INK); a.set_ylabel('T агента / эталон', color=INK)
mm = ok & inf_; a.set_title('T/эталон: медиана %.3f, среднее %.3f, максимум %.2f' % (np.median(q[mm]), q[mm].mean(), q[mm].max()), color=INK, fontsize=11, loc='left')
for s_ in a.spines.values(): s_.set_color(GRID)
a.tick_params(colors=MUT)
fig.suptitle('Двойной интегратор 4D: 6400 клеток (1600 на слой), 2.81 млн узлов; в поле дошли %d из %d; запрос ~90 мс' % (mm.sum(), inf_.sum()), color=INK, fontsize=13, x=.01, ha='left'); fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig('pics/di4_L1600_agent.png', dpi=130, facecolor=BG)
print('готово: дошли %d/60, в поле %d/%d, T/эт мед. %.3f mean %.3f max %.2f' % (ok.sum(), mm.sum(), inf_.sum(), np.median(q[mm]), q[mm].mean(), q[mm].max()))
