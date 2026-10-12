"""lr4: «горб» V* перед кривой переключения (наблюдение пользователя 2026-10-12 ~16:45: «за мгновение до прихода к кривой переключения есть горб — потенциальный барьер»).
Профиль вдоль ОПТИМАЛЬНОЙ траектории (первое управление по T_ref, держим до оптимального момента переключения t_sw и дальше оптимально):
e(t) = V*(x(t)) − T_ref(x(t)) и Q_u − T_ref по слоям, t − t_sw ∈ [−0.8, +0.3]. Горб = max e на [−0.5, 0] − e(−0.8).
Варианты: Q lin (переключение только в узлах), Q lin + K подточек (qsolve3), «вдоль потока» + K подточек (qsolve4). env M, LAYERS, SEED, N, OUT. Запуск из v8."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
import solve8 as S, qsolve as QS, qsolve3 as Q3, qsolve4 as Q4
from src.algo import ref_di_disk as R
OUT = os.environ.get('OUT', 'hump'); TG = np.round(np.arange(-.8, .3001, .02), 2)


def hit_time(x, v, u, rho, S):
    """первое время входа в круг под постоянным u из (x, v) (векторы) по сетке S с линейным уточнением; inf — не вошёл"""
    xs = x[:, None] + v[:, None] * S + u * S * S / 2; vs = v[:, None] + u * S; r = np.sqrt(xs * xs + vs * vs) - rho; inside = r <= 0; k = inside.argmax(1); ok = inside.any(1); k1 = np.maximum(k, 1); ar = np.arange(len(x))
    r0 = r[ar, k1 - 1]; r1 = r[ar, k1]; t = S[k1 - 1] + (S[k1] - S[k1 - 1]) * r0 / np.maximum(r0 - r1, 1e-12); t = np.where(k == 0, 0., t); return np.where(ok, t, np.inf)


def opt_switch(X0, rho, h=.004, tmax=7.):
    """оптимум ДИ → круг в классе «u0 до t1, затем −u0» (перебор t1 по сетке h + время входа): u0, t1, T (t1 = inf — без переключения)"""
    S = np.arange(0, tmax, h); n = len(X0); U0 = np.zeros(n); T1 = np.full(n, np.inf); TT = np.full(n, np.inf)
    for i, (x0, v0) in enumerate(X0):
        for u in (1., -1.):
            t0 = hit_time(np.array([x0]), np.array([v0]), u, rho, S)[0]
            if t0 < TT[i]: TT[i], T1[i], U0[i] = t0, np.inf, u
            x1 = x0 + v0 * S + u * S * S / 2; v1 = v0 + u * S; tot = S + hit_time(x1, v1, -u, rho, S); tot[S >= t0] = np.inf; k = tot.argmin()
            if tot[k] < TT[i] - 1e-9: TT[i], T1[i], U0[i] = tot[k], S[k], u
    return U0, T1, TT


if __name__ == '__main__':
    g = S.load_growN(); layers = [float(x) for x in os.environ.get('LAYERS', '1,-1,0').split(',')]; atlas, _ = S.build_atlas(layers, int(os.environ.get('SEED', '0')), par=False); G = S.make_graph(g, atlas, 'N'); rho = G.rho; uvec = np.array(G.us)
    X0 = S.starts(g, int(os.environ.get('N', '600')), 1); mx = S.path_extent(X0, rho); X0 = X0[mx <= float(g.XLV[0])]; U0, T1, TT = opt_switch(X0, rho)
    chk = R.T_ref(X0[:60], rho); print('RES проверка оптимума перебором против T_ref (60 стартов): max |T − T_ref| = %.4f' % np.abs(TT[:60] - chk).max(), flush=True)
    ids = np.flatnonzero(np.isfinite(T1) & (T1 >= .8) & (TT - T1 >= .3)); PTS = np.zeros((len(ids), len(TG), 2)); TRF = np.zeros((len(ids), len(TG)))
    for r, i in enumerate(ids):
        t = T1[i] + TG; a = np.minimum(t, T1[i]); b = np.maximum(t - T1[i], 0.); u = U0[i]; x1 = X0[i, 0] + X0[i, 1] * a + u * a * a / 2; v1 = X0[i, 1] + u * a; PTS[r, :, 0] = x1 + v1 * b - u * b * b / 2; PTS[r, :, 1] = v1 - u * b; TRF[r] = TT[i] - t
    print('RES слои %s M=%d узлов %d: стартов в поле %d, с переключением (≥.8 с до него, ≥.3 с после) %d' % (layers, G.M, G.n, len(X0), len(ids)), flush=True)
    res = {}; prof = {}

    def rep(name):
        E = np.full((len(ids), len(TG)), np.nan); EL = np.full((len(ids), len(TG), G.L), np.nan)
        for r in range(len(ids)):
            Vl = G.Vlayers(PTS[r]); EL[r] = Vl - TRF[r][:, None]; E[r] = Vl.min(1) - TRF[r]
        E[~np.isfinite(E)] = np.nan; i8 = 0; win = (TG >= -.5) & (TG <= 0); full = np.isfinite(E[:, i8]) & np.isfinite(E[:, win]).all(1); Ef = E[full]; hump = Ef[:, win].max(1) - Ef[:, i8]; tpk = TG[win][Ef[:, win].argmax(1)]
        dV = np.diff(Ef + (-TG)[None, :], axis=1); up = (dV[:, (TG[1:] >= -.5) & (TG[1:] <= 0)] > 0).any(1)                 # V* растёт вдоль оптимальной траектории (должна падать со скоростью 1) — барьер для агента
        own = np.array([np.flatnonzero(uvec == U0[i])[0] for i in ids])[full]; Eown = EL[full][np.arange(full.sum()), :, own]; Eown[~np.isfinite(Eown)] = np.nan
        row = dict(n=int(full.sum()), hump_med=float(np.median(hump)), hump_p90=float(np.percentile(hump, 90)), hump_max=float(hump.max()), t_peak_med=float(np.median(tpk)), frac_hump_gt02=float((hump > .02).mean()), frac_V_rises=float(up.mean()),
                   e_far=float(np.median(Ef[:, i8])), e_at_sw=float(np.median(Ef[:, TG == 0])), e_after=float(np.median(Ef[:, TG == .2][np.isfinite(Ef[:, TG == .2])])), prof_med=np.round(np.nanmedian(Ef, 0), 4).tolist(), prof_p90=np.round(np.nanpercentile(Ef, 90, 0), 4).tolist(), own_med=np.round(np.nanmedian(Eown, 0), 4).tolist())
        res[name] = row; prof[name] = row
        print('RES %-14s трасс %d | горб (max e на −.5..0 минус e(−.8)): мед %.3f p90 %.3f max %.3f, пик при t−t_sw мед %+.2f | доля с горбом >.02: %.2f, доля где V* РАСТЁТ вдоль траектории: %.2f | e=V*−T_ref мед: далеко до %.3f, в t_sw %.3f, +.2 после %.3f' % (
            name, row['n'], row['hump_med'], row['hump_p90'], row['hump_max'], row['t_peak_med'], row['frac_hump_gt02'], row['frac_V_rises'], row['e_far'], row['e_at_sw'], row['e_after']), flush=True)
        print('RES %-14s профиль мед e(t): %s' % (name, dict(zip(TG[::5].tolist(), np.round(np.nanmedian(Ef, 0), 3)[::5].tolist()))), flush=True)

    V, Q, it, ts = QS.solve_q(G, 0.); G.V = Q; rep('Q lin K=0')
    for k in (1, 3, 7):
        V, Q, it, ts = Q3.solve_q3(G, Q3.prep_sub(G, k)); G.V = Q; rep('Q lin K=%d' % k)
    Q4.patch_flow(G)
    for k in (3, 7, 15):
        V, Q, it, ts = Q3.solve_q3(G, Q3.prep_sub(G, k)); G.V = Q; rep('flow K=%d' % k)
    json.dump(dict(t=TG.tolist(), res=res), open(OUT + '.json', 'w'), ensure_ascii=False, indent=1)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for name, row in prof.items():
            ax[0].plot(TG, row['prof_med'], label=name, lw=2 if name in ('Q lin K=0', 'flow K=3') else 1); ax[1].plot(TG, row['prof_p90'], label=name, lw=2 if name in ('Q lin K=0', 'flow K=3') else 1)
        for a, ttl in zip(ax, ('медиана', '90-й процентиль')):
            a.axvline(0, color='k', lw=.6); a.axhline(0, color='k', lw=.6); a.set_xlabel('t − t_sw, с (0 — оптимальный момент переключения)'); a.set_ylabel('V* − T_ref'); a.set_title('Ошибка V* вдоль оптимальной траектории, %s (M=%d, %d трасс)' % (ttl, G.M, prof['Q lin K=0']['n'])); a.grid(alpha=.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT + '.png', dpi=110); print('RES картинка', OUT + '.png', flush=True)
    except Exception as e:
        print('RES картинки нет:', repr(e), flush=True)
