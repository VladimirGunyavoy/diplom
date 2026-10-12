"""lr4: застревание агента на v≈0 в слое u=0 (наблюдение пользователя 2026-10-12: «у нуля что-то странное с переключением»).
Гипотеза research-3: (v≈0, u=0) — равновесие слоя 0, Q_0 ≈ V*, агент выбирает слой 0 и стоит до tmax.
Проверка: слои из LAYERS (нужен 0), расчёты Q lin (qsolve) и «вдоль потока» K=3 (qsolve4); агенты look и commit (свои копии с записью конца и пути);
лечения — запрет ВЫБОРА слоя агентом: none / no0 (слой 0 запрещён всегда) / fthr (слой 0 запрещён при |f| = |v| < FTHR).
Запуск из папки v8: python3 stuck0.py; env M, LAYERS (1,-1,0), SEED (0), N (1000), FTHR (.05), OUT."""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, qsolve3 as Q3, qsolve4 as Q4
from src.algo import ref_di_disk as R
JS = np.array([0, 1, 2, 5, 10, 20, 40]); FTHR = float(os.environ.get('FTHR', '.05')); OUT = os.environ.get('OUT', 'stuck0')


def make_ban(kind, i0, L):
    def ban(X):
        b = np.zeros((len(X), L), bool)
        if i0 is None or kind == 'none': return b
        b[:, i0] = True if kind == 'no0' else np.abs(X[:, 1]) < FTHR
        return b
    return ban


def run_look(G, X0, tmax, ban, dt=.01, eps=1e-3, path=None, rec=5):
    """копия solve8.run_agent(look=True) + запрет выбора слоя + конец траектории"""
    n = len(X0); X = X0.copy(); t = np.zeros(n); alive = np.ones(n, bool); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(G.us)
    Vl = G.Vlayers(X); Vl[ban(X)] = np.inf; cur = Vl.argmin(1); alive &= np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= G.rho; T[inn] = 0; alive &= ~inn; step = 0
    while alive.any():
        a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
        hit = np.linalg.norm(X[a], axis=1) <= G.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
        if path is not None and step % rec == 0: path.append((a, X[a].copy(), cur[a].copy()))
        a = np.flatnonzero(alive)
        if not len(a): break
        sc_ = np.empty((len(a), G.L))
        for li, u_ in enumerate(uvec): xn = np.stack([X[a, 0] + X[a, 1] * dt + u_ * dt * dt / 2, X[a, 1] + u_ * dt], 1); sc_[:, li] = G.Vlayers(xn).min(1)
        sc_[ban(X[a])] = np.inf
        best = sc_.argmin(1); vc = sc_[np.arange(len(a)), cur[a]]; vb = sc_[np.arange(len(a)), best]; chg = (vb < vc - eps * dt) | (~np.isfinite(vc) & np.isfinite(vb)); sw[a[chg]] += 1; cur[a[chg]] = best[chg]; alive[a[~np.isfinite(vb)]] = False
    return T, sw, X, cur


def run_commit(G, X0, tmax, ban, dt=.01, eps=1e-3, path=None, rec=5):
    """копия lr3/qagent.run_commit + запрет выбора слоя + конец траектории"""
    n = len(X0); X = X0.copy(); t = np.zeros(n); T = np.full(n, np.nan); sw = np.zeros(n, int); uvec = np.array(G.us); Vl = G.Vlayers(X); Vl[ban(X)] = np.inf; cur = Vl.argmin(1); alive = np.isfinite(Vl.min(1)); inn = np.linalg.norm(X, axis=1) <= G.rho; T[inn] = 0; alive &= ~inn; step = 0
    while alive.any():
        a = np.flatnonzero(alive); u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; ar = np.arange(len(a))
        best = np.full(len(a), np.inf); bj = np.zeros(len(a), int); bl = cur[a].copy(); qcur = None
        for j in JS:
            tj = j * dt; Pj = np.stack([x + v * tj + u * tj * tj / 2, v + u * tj], 1); Q = G.Vlayers(Pj); Q[ban(Pj)] = np.inf
            if j == 0: qcur = Q[ar, cur[a]].copy()
            Q[ar, cur[a]] = np.inf; g = tj + Q.min(1); better = g < best - 1e-12; best[better] = g[better]; bj[better] = j; bl[better] = Q.argmin(1)[better]
        chg = (bj == 0) & np.isfinite(best) & ((best < qcur - eps) | ~np.isfinite(qcur)); lost = ~np.isfinite(best) & ~np.isfinite(qcur)
        cur[a[chg]] = bl[chg]; sw[a[chg]] += 1; alive[a[lost]] = False; a = np.flatnonzero(alive)
        if not len(a): break
        u = uvec[cur[a]]; x, v = X[a, 0], X[a, 1]; X[a, 0] = x + v * dt + u * dt * dt / 2; X[a, 1] = v + u * dt; t[a] += dt; step += 1
        hit = np.linalg.norm(X[a], axis=1) <= G.rho; T[a[hit]] = t[a[hit]]; alive[a[hit]] = False; alive[a[t[a] > tmax[a]]] = False
        if path is not None and step % rec == 0: path.append((a, X[a].copy(), cur[a].copy()))
    return T, sw, X, cur


def lines(path, ids):
    d = {int(i): ([], []) for i in ids}
    for a, Xr, cr in path:
        m = np.isin(a, ids)
        for i, xr, c in zip(a[m], Xr[m], cr[m]): d[int(i)][0].append(xr); d[int(i)][1].append(int(c))
    return {i: (np.array(p), np.array(c)) for i, (p, c) in d.items() if len(p)}


if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, int(os.environ.get('SEED', '0')), par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho
    uvec = np.array(G.us); i0 = int(np.flatnonzero(uvec == 0)[0]) if (uvec == 0).any() else None
    X0 = S.starts(g, int(os.environ.get('N', '1000')), 1); Tr = R.T_ref(X0, rho); mx = S.path_extent(X0, rho); ins = mx <= float(g.XLV[0]) + g.GM; inw = mx <= float(g.XLV[0]); tmax = 3 * Tr + 2
    print('RES слои %s (us %s, слой 0 = %s) M=%d узлов %d стартов %d, в росте %d, в поле %d, FTHR %g' % (layers, G.us, i0, G.M, G.n, len(X0), ins.sum(), inw.sum(), FTHR), flush=True)
    res = {}; pics = {}

    def rep(name):
        t1 = time.perf_counter(); Tt, sw = G.run_agent(X0, tmax, look=True); ok = np.isfinite(Tt); print('RES %-9s проверка: штатный look дошли %.3f (в поле %.3f) %.0f с' % (name, ok.mean(), ok[inw].mean(), time.perf_counter() - t1), flush=True)
        for kind in ('none', 'no0', 'fthr'):
            if i0 is None and kind != 'none': continue
            ban = make_ban(kind, i0, G.L)
            for pn, fn in (('look', run_look), ('commit', run_commit)):
                path = []; t1 = time.perf_counter(); Tt, sw, Xe, ce = fn(G, X0, tmax, ban, path=path); dtm = time.perf_counter() - t1; ok = np.isfinite(Tt); r = Tt[ok & ins] / Tr[ok & ins]; rw = Tt[ok & inw] / Tr[ok & inw]
                fail = ~ok; vl = G.Vlayers(X0); has = np.isfinite(vl.min(1)); slow = fail & (np.abs(Xe[:, 1]) < FTHR) & (np.linalg.norm(Xe, axis=1) > rho); st0 = slow & (ce == i0) if i0 is not None else slow & False
                row = dict(reached=float(ok.mean()), reached_grow=float(ok[ins].mean()), reached_field=float(ok[inw].mean()), T_med=float(np.median(r)), T_p90=float(np.percentile(r, 90)), T_max=float(r.max()), Tw_med=float(np.median(rw)), Tw_p90=float(np.percentile(rw, 90)), Tw_max=float(rw.max()),
                           sw_med=float(np.median(sw[ins])), sw_mean=float(sw[ins].mean()), fail=int(fail.sum()), fail_field=int((fail & inw).sum()), fail_hasV=int((fail & has).sum()), stuck_v0=int(slow.sum()), stuck_v0_u0=int(st0.sum()), stuck_v0_u0_field=int((st0 & inw).sum()), sec=dtm)
                res['%s|%s|%s' % (name, kind, pn)] = row
                print('RES %-9s %-4s %-6s дошли %.3f (в росте %.3f, в поле %.3f) | провалов %d (в поле %d, с V* на старте %d), из них стоят на |v|<%g вне цели %d, в слое 0: %d (в поле %d) | T/T_ref мед %.3f p90 %.3f max %.2f | в поле мед %.3f p90 %.3f max %.2f | перекл мед %.0f ср %.1f | %.0f с' % (
                    name, kind, pn, row['reached'], row['reached_grow'], row['reached_field'], row['fail'], row['fail_field'], row['fail_hasV'], FTHR, row['stuck_v0'], row['stuck_v0_u0'], row['stuck_v0_u0_field'], row['T_med'], row['T_p90'], row['T_max'], row['Tw_med'], row['Tw_p90'], row['Tw_max'], row['sw_med'], row['sw_mean'], dtm), flush=True)
                if kind == 'none' and st0.any():
                    ids = np.flatnonzero(st0); Vq = G.Vlayers(Xe[ids]); oth = Vq.copy(); oth[:, i0] = np.inf
                    print('RES %-9s %-4s %-6s застрявшие: концы x %s | Q_0 − min(Q других) в конце: %s' % (name, kind, pn, np.round(Xe[ids[:12], 0], 2).tolist(), np.round((Vq[:, i0] - oth.min(1))[:12], 4).tolist()), flush=True)
                    pics[(name, pn)] = ids
                if (name, pn) in pics: pics[(name, pn, kind)] = lines(path, pics[(name, pn)])

    V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; rep('Q lin')
    Q4.patch_flow(G); V, Q, it, ts = Q3.solve_q3(G, Q3.prep_sub(G, 3)); G.V = Q; rep('flow K=3')
    json.dump(res, open(OUT + '.json', 'w'), ensure_ascii=False, indent=1)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        keys = [k for k in pics if len(k) == 2]
        if keys:
            name, pn = keys[0]; kinds = [k for k in ('none', 'no0', 'fthr') if (name, pn, k) in pics]; fig, ax = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5.5), squeeze=False); col = {1.: 'tab:red', -1.: 'tab:purple', 0.: 'tab:green'}
            for axi, kind in zip(ax[0], kinds):
                for i, (p, c) in pics[(name, pn, kind)].items():
                    cut = np.flatnonzero(np.r_[True, c[1:] != c[:-1], True])
                    for a, b in zip(cut[:-1], cut[1:]): axi.plot(p[a:b + 1, 0], p[a:b + 1, 1], color=col.get(float(uvec[c[a]]), 'k'), lw=1.2, ls='--' if uvec[c[a]] == 0 else '-')
                    axi.plot(*X0[i], 'k.', ms=4); axi.plot(*p[-1], 'kx', ms=6)
                th = np.linspace(0, 2 * np.pi, 100); axi.plot(rho * np.cos(th), rho * np.sin(th), 'g-'); axi.axhline(0, color='k', lw=.4); axi.set_title('%s, агент %s, лечение: %s (%d стартов, × — конец)' % (name, pn, kind, len(pics[(name, pn)]))); axi.set_xlabel('x'); axi.set_ylabel('v'); axi.grid(alpha=.3)
            fig.suptitle('Старты, застрявшие в слое u=0 на v≈0 (красный u=+1, фиолетовый u=−1, зелёный штрих u=0), M=%d' % G.M); fig.tight_layout(); fig.savefig(OUT + '.png', dpi=110); print('RES картинка', OUT + '.png', flush=True)
    except Exception as e:
        print('RES картинки нет:', repr(e), flush=True)
