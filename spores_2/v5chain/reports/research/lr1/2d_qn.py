"""Квазинормальные фронты клетки, 2D (lr1, worker-2). Запуск: SYS=di|pend python 2d_qn.py  → 2d_<sys>_*.png, 2d_<sys>.json.
Варианты V0 плоский / V1 позвоночник / V2 МНК-подгонка времени / V3 нормальные кривые; динамика — growN (v8, не правится)."""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', 'v8')))
os.environ.setdefault('TQDM_MI', '1000')
from src.algo import growN as g
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tqdm import tqdm

NF = [0]; _f = g.f
def fc(y, u): NF[0] += int(np.size(y) // g.N); return _f(y, u)
g.f = fc
f = g.f; H = g.DTN; K = 11; NS = 15; N = g.N
SUB = 20                                  # подшагов на h в V1
def unit(v): return v / np.linalg.norm(v, axis=-1, keepdims=True)

def flow(y, u, T, hmax=H / 2):
    n = max(1, int(np.ceil(abs(T) / hmax - 1e-9))); return g.rk4(y, u, T / n, n)

def trajs(X, u, T):                       # траектории узлов на [0,T], подшаг H/SUB → (SUB*T/H+1, K, 2)
    n = int(round(SUB * T / H)); out = [X]
    for _ in range(n): out.append(g.rk4(out[-1], u, H / SUB, 1))
    return np.array(out)

def V0(p, u, r):
    e = g.basis(p, u)[0]; s = np.linspace(-r, r, K)[:, None]; X = p + s * e; F = [X]
    for _ in range(NS): X = g.rk4(X, u, H, 1); F.append(X)
    return np.array(F), None

def V1(p, u, r):
    e = g.basis(p, u)[0]; s = np.linspace(-r, r, K)[:, None]; X = p + s * e; F = [X]; c = p.copy(); taus = []
    for k in range(1, NS + 1):
        c = g.rk4(c, u, H, 1); fc_ = f(c, u); Y = X.copy(); tau = np.zeros(K)
        for i in range(K):
            x = X[i]; gx = (x - c) @ fc_; sg = -np.sign(gx) if gx != 0 else 1.; t = 0.
            for _ in range(6 * SUB):
                xn = g.rk4(x, u, sg * H / SUB, 1); gn = (xn - c) @ fc_
                if gn * gx <= 0: w = gx / (gx - gn); x = x + w * (xn - x); t += w * H / SUB; break
                x, gx, t = xn, gn, t + H / SUB
            Y[i] = x; tau[i] = sg * t
        X = Y; F.append(X); taus.append(tau)
    return np.array(F), np.array(taus)

def V2(p, u, r):
    e = g.basis(p, u)[0]; s = np.linspace(-r, r, K)[:, None]; X = p + s * e; F = [X]; taus = []; res = []; ic = K // 2
    for k in range(1, NS + 1):
        X = g.rk4(X, u, H, 1); fx = f(X, u); nh = unit(fx[1:] + fx[:-1]); b = np.einsum('ij,ij->i', fx, np.vstack([nh[:1], (nh[1:] + nh[:-1]) / 2, nh[-1:]]))   # временно; ниже строим систему точно
        A = np.zeros((K - 1, K)); a = np.zeros(K - 1)
        for i in range(K - 1):
            A[i, i + 1] = fx[i + 1] @ nh[i]; A[i, i] = -fx[i] @ nh[i]; a[i] = (X[i + 1] - X[i]) @ nh[i]
        keep = [j for j in range(K) if j != ic]; d = np.zeros(K)
        sol = np.linalg.lstsq(A[:, keep], -a, rcond=None)[0]; d[keep] = sol; res.append(float(np.abs(A @ d + a).max()))
        Y = X.copy()
        for i in range(K): Y[i] = flow(X[i], u, d[i]) if d[i] != 0 else X[i]
        X = Y; F.append(X); taus.append(d)
    return np.array(F), np.array(taus), res

def V3(p, u, r, ds_div=50):
    """нормальные кривые из оси c_k; K=11 узлов: j = -5..5, шаг по длине дуги r/5, ds = r/ds_div, на каждой стороне 10 подшагов → (K-1)/2 узлов на сторону."""
    e = g.basis(p, u)[0]; c = p.copy(); F = []; X = c + np.linspace(-r, r, K)[:, None] * e; F.append(X); per = (K - 1) // 2; ds = r / ds_div; sub = ds_div // per
    for k in range(1, NS + 1):
        c = g.rk4(c, u, H, 1); fcv = f(c, u); fh = fcv / np.linalg.norm(fcv); e = e - (e @ fh) * fh; e = unit(e)
        if e @ F[-1][-1] - e @ F[-1][0] < 0: e = -e    # непрерывность знака (в 2D — единственная ось)
        X = np.zeros((K, 2)); X[per] = c
        for sg in (+1, -1):
            x = c.copy(); d = sg * e
            for j in range(1, per + 1):
                for _ in range(sub):
                    fx = f(x, u); fh_ = fx / np.linalg.norm(fx); d = d - (d @ fh_) * fh_; d = d / np.linalg.norm(d); x = x + ds * d
                X[per + sg * j] = x
        F.append(X)
    return np.array(F), None

def ortho(F, u):                           # cos(касательная фронта, f) по узлам, по фронтам 1..NS
    cs = []
    for X in F[1:]:
        T = np.gradient(X, axis=0); T = unit(T); fx = unit(f(X, u)); cs.append(np.abs(np.einsum('ij,ij->i', T, fx)))
    cs = np.array(cs); return float(cs.mean()), float(cs.max()), float(cs[-1].mean()), float(cs[-1].max())

def consist(Fold, Fnew, u, r):             # расстояние узлов нового фронта до траекторий узлов старого на [0, 2H] (в долях r) и время прихода
    Tr = trajs(Fold, u, 2 * H); dt = H / SUB; P = Tr.reshape(len(Tr), -1, 2); dist = []; tarr = []
    for y in Fnew:
        best = (1e9, 0.)
        for i in range(P.shape[1]):
            A, B = P[:-1, i], P[1:, i]; AB = B - A; w = np.clip(np.einsum('ij,ij->i', y - A, AB) / np.einsum('ij,ij->i', AB, AB), 0, 1); Q = A + w[:, None] * AB; dd = np.linalg.norm(Q - y, axis=1); j = dd.argmin()
            if dd[j] < best[0]: best = (dd[j], (j + w[j]) * dt)
        dist.append(best[0] / r); tarr.append(best[1])
    return np.array(dist), np.array(tarr)

def width(F, u):
    def w(X): fc_ = unit(f(X[K // 2], u)); ch = X[-1] - X[0]; return float(np.linalg.norm(ch - (ch @ fc_) * fc_))
    return w(F[-1]) / w(F[0])

def run(p, u, r, name):
    out = {}; D = {}
    for v, fn in (('V0', V0), ('V1', V1), ('V2', V2), ('V3', V3)):
        NF[0] = 0; t0 = time.perf_counter(); R = fn(p, u, r); ms = 1e3 * (time.perf_counter() - t0); nf = NF[0]
        F = R[0]; o = ortho(F, u); m = dict(cos_mean_all=o[0], cos_max_all=o[1], cos_mean_last=o[2], cos_max_last=o[3], width_ratio=width(F, u), f_calls_per_front=nf / NS, ms_per_front=ms / NS)
        if v in ('V1', 'V2'):
            tau = R[1]; acc = np.cumsum(tau, axis=0) if v == 'V2' else np.cumsum(tau, axis=0)
            m['tau_spread_last_step'] = float(tau[-1].max() - tau[-1].min()); m['tau_spread_accum'] = float(acc[-1].max() - acc[-1].min())
        if v == 'V2': m['lsq_resid_max'] = float(max(R[2]))
        dist, tarr = consist(F[-2], F[-1], u, r); m['clone_dist_max_r'] = float(dist.max()); m['clone_dist_mean_r'] = float(dist.mean())
        if v == 'V3': m['tau_spread_last_step'] = float(tarr.max() - tarr.min())
        m['arrival_t_mean'] = float(tarr.mean()); out[v] = m; D[v] = F
    return out, D

def plot(p, u, D, r, ax):
    cols = dict(V0='tab:gray', V1='tab:blue', V2='tab:green', V3='tab:red'); lw = dict(V0=1.4, V1=1.0, V2=1.0, V3=1.0)
    F0 = D['V0']
    for i in range(0, K, 2):
        tr = trajs(F0[0], u, NS * H)[:, i]; ax.plot(*tr.T, color='k', lw=.4, alpha=.35)
    for v in ('V0', 'V1', 'V2', 'V3'):
        for k, X in enumerate(D[v]):
            ax.plot(*X.T, color=cols[v], lw=lw[v], alpha=.9, label=v if k == 1 else None, ls='-' if v != 'V3' else '--')
    ax.plot(*p, '*k', ms=9); ax.set_aspect('equal'); ax.set_title('r=%.1f' % r); ax.legend(fontsize=7)

if __name__ == '__main__':
    cfg = {'di': [(1., [(0., 0.), (.5, .8)])], 'pend': [(.3, [(.5, 0.), (2.5, .3)])]}[g.SYS]; allr = {}
    for u, seeds in cfg:
        for si, p in enumerate(seeds):
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            for ai, r in enumerate((.2, .5)):
                key = '%s_u%s_seed%d_(%s,%s)_r%s' % (g.SYS, u, si, p[0], p[1], r); res, D = run(np.array(p, float), u, r, key); allr[key] = res; plot(np.array(p), u, D, r, axs[ai])
            fig.suptitle('%s u=%s seed=%s: fronts V0..V3, %d steps h=%.1f' % (g.SYS, u, p, NS, H)); fig.tight_layout(); fig.savefig(os.path.join(HERE, '2d_%s_seed%d.png' % (g.SYS, si)), dpi=110); plt.close(fig)
    json.dump(allr, open(os.path.join(HERE, '2d_%s.json' % g.SYS), 'w'), indent=1)
    for k, v in allr.items():
        print(k)
        for vv, m in v.items(): print('  ', vv, {a: round(b, 4) for a, b in m.items()})
