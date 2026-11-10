"""ENV1 (b14, 2026-11-11): статик кар = среда journal-2026 ТОЧНО. Проверяльщик на ИХ шаге (SafeCarGoalEnv.step: Эйлер dt .08, θ←θ+ω dt, v←clip(.95 v+a dt, −1, 2), p←p+v dt(cosθ, sinθ)),
рядом с нашим непрерывным (common.verify, RK4 h .01). План споры = (u_k, dt_k) (открытый контур из атласа growCar + финиш-стрельба ≤ 3 дуг), у них — кусочно-постоянное u на шаге .08 (по середине шага).
Метрики: успех (центр в круге цели в отсчёте шага) ∧ без нарушения (их cost: центр в диске / вне ±5 в отсчёте); отдельно «непрерывное» касание — 20 точек на шаг (линейно между отсчётами).
Запуск (aida 2, из bench/): LOAD=~/spore_v5/car/c1600.pkl SYS=car python envs2026.py  → data/env_car2026.json. Отклонение: финиш — сетка ≤ 3 дуг H .1 БЕЗ доводки SLSQP (shoot_pol её имеет, но не отдаёт длительности)."""
import os, sys, json, time
os.environ.setdefault('SYS', 'car'); os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'v7', 'src', 'cells7'))
import numpy as np
HZ = np.array([[0., .2], [1.8, 1.5], [-1.6, 2.3]]); HR = .70; GOAL = np.array([4., 3.6]); GR = .45; DT = .08
UA, UW = 1.5, 1.8

def their_step(y, u):
    """один шаг SafeCarGoalEnv: y=(x,y,θ,v), u=(a,ω) (клип)"""
    a = np.clip(u[0], -UA, UA); w = np.clip(u[1], -UW, UW); th = y[2] + w * DT; v = np.clip(.95 * y[3] + a * DT, -1., 2.)
    return np.array([y[0] + v * DT * np.cos(th), y[1] + v * DT * np.sin(th), th, v])

def dclear(p): return float((np.linalg.norm(HZ - p, axis=1) - HR).min())

def their_replay(x0, us, dts, tmax=40.):
    """проигрывает план на их шаге. Возврат dict: ok (успех в отсчёте), T, viol_step (число отсчётов с cost), viol_cont (макс. проникновение между отсчётами), steps"""
    t_end = float(np.sum(dts)); cum = np.cumsum(dts); y = np.array(x0, float); n = 0; vs = 0; pen = 0.; ok = False
    if np.linalg.norm(y[:2] - GOAL) <= GR: return dict(ok=True, T=0., viol_step=0, viol_cont=0., steps=0)
    nmax = int(np.ceil(max(t_end, 1e-9) / DT)) + 1
    for j in range(nmax):
        tm = (j + .5) * DT; k = min(int(np.searchsorted(cum, tm)), len(us) - 1); yn = their_step(y, us[k]); n += 1
        for s in np.linspace(0, 1, 21)[1:]:
            p = y[:2] * (1 - s) + yn[:2] * s; pen = max(pen, -dclear(p), np.abs(p).max() - 5.)
        if dclear(yn[:2]) < 0 or np.abs(yn[:2]).max() > 5.: vs += 1
        y = yn
        if np.linalg.norm(y[:2] - GOAL) <= GR: ok = True; break
    return dict(ok=ok, T=n * DT if ok else None, viol_step=vs, viol_cont=float(max(pen, 0.)), steps=n)

def finish_plan(G, y, tmax=3., H=.1, NA=3):
    """сетка ≤ NA дуг шагом H (как finish_gen.shoot_grid), но с длительностями. Цель — вписанный квадрат круга (θ, v любые, как _wcar). Возврат (T, [(u, dt)]) или (inf, None)"""
    K = int(round(tmax / H)); R = .9 * GR / np.sqrt(2); nu = len(G.US); tj = np.arange(K + 1) * H; best = (np.inf, None)
    def traj(Y, u):
        u = np.asarray(u, float); out = [Y]
        for _ in range(K):
            k1 = G.f(Y, u); k2 = G.f(Y + H / 2 * k1, u); k3 = G.f(Y + H / 2 * k2, u); k4 = G.f(Y + H * k3, u); Y = Y + H / 6 * (k1 + 2 * k2 + 2 * k3 + k4); out.append(Y)
        return np.stack(out, 1)
    S = np.asarray(y, float)[None]; T = np.zeros(1); TOP = np.zeros((1, 0), int); DUR = np.zeros((1, 0)); last = np.full(1, -1)
    for lev in range(NA):
        nS, nT, nTOP, nDUR, nL = [], [], [], [], []
        for a in range(nu):
            m = (last != a) & (T < min(best[0], tmax))
            if not m.any(): continue
            tr = traj(S[m], G.US[a]); Tt = T[m][:, None] + tj[None]
            ok = (np.abs(tr[..., :2] - G.GC) <= R).all(-1)
            if ok.any():
                tt = np.where(ok, Tt, np.inf); i = np.unravel_index(tt.argmin(), tt.shape)
                if tt[i] < best[0]:
                    top = list(TOP[m][i[0]]) + [a]; dur = list(DUR[m][i[0]]) + [float(tj[i[1]])]
                    best = (float(tt[i]), [(G.US[p], d) for p, d in zip(top, dur) if d > 1e-12])
            if lev < NA - 1:
                n0 = int(m.sum()); nS.append(tr[:, 1:].reshape(-1, tr.shape[-1])); nT.append(Tt[:, 1:].reshape(-1))
                nTOP.append(np.concatenate([np.repeat(TOP[m], K, 0), np.full((n0 * K, 1), a)], 1)); nDUR.append(np.concatenate([np.repeat(DUR[m], K, 0), np.tile(tj[1:], n0)[:, None]], 1)); nL.append(np.full(n0 * K, a))
        if lev == NA - 1 or not nS: break
        S, T, TOP, DUR, last = map(np.concatenate, (nS, nT, nTOP, nDUR, nL)); k = T < min(best[0], tmax); S, T, TOP, DUR, last = S[k], T[k], TOP[k], DUR[k], last[k]
        if not len(S): break
    return best

def plan(G, A, x, tmax=40., VF=1.5):
    """открытый контур агента: жадный по J = min(tgoal, DTN + V*(step)) как Atlas.rollout; у цели (J ≤ VF) — финиш-стрельба. Возврат (us, dts, T_agent) или None"""
    y = G.wrapy(np.asarray(x, float)[None]); us, dts, t = [], [], 0.
    for _ in range(int(tmax / G.DTN)):
        if G.ingoal(y)[0]: return us, dts, t
        tgs = np.array([A.tgoal(y, u)[0] for u in G.US]); vs = np.array([A.vstar(G.step(y, u))[0] for u in G.US]); J = np.minimum(tgs, G.DTN + vs); k = int(J.argmin())
        if J[k] >= G.BIG / 2: return None
        if J[k] <= VF:
            Tf, pl = finish_plan(G, y[0])
            if pl is not None: return us + [list(u) for u, d in pl], dts + [d for u, d in pl], t + Tf
        h = float(tgs[k]) if np.isfinite(tgs[k]) else G.DTN
        us.append(list(G.US[k])); dts.append(h); y = G.step(y, G.US[k]) if not np.isfinite(tgs[k]) else G.rk4(y, G.US[k], h / 8, 8); y = G.wrapy(y)
        y[:, 3] = np.clip(y[:, 3], G.VLO, G.VHI); t += h
    return None

_A = None; _G = None
def _one(args):
    i, x = args; t0 = time.perf_counter(); r = plan(_G, _A, x); dq = time.perf_counter() - t0
    if r is None: return dict(i=i, why='no_plan', t_query=dq)
    us, dts, Ta = r; th = their_replay(x, us, dts)
    import common; e = common.make('car'); Tv, why = common.verify(e, x, us, dts)
    return dict(i=i, T_agent=Ta, T_cont=Tv, why_cont=why, their=th, t_query=dq, nsteps=len(us))

if __name__ == '__main__':
    import pickle, multiprocessing as mp, growCar as G, common
    import __main__, inspect
    for nm, o in vars(G).items():
        if inspect.isclass(o): setattr(__main__, nm, o)   # дамп сделан growCar как __main__
    _G = G; d = pickle.load(open(os.path.expanduser(os.environ['LOAD']), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; _A = A
    e = common.make('car'); X0 = e.X0; Tref = e.Tref
    t0 = time.perf_counter()
    with mp.get_context('fork').Pool(int(os.environ.get('NP', 20))) as p: res = p.map(_one, list(enumerate(X0)))
    wall = time.perf_counter() - t0; res.sort(key=lambda r: r['i'])
    n = len(res); Tc = np.array([np.nan if r.get('T_cont') is None else r['T_cont'] for r in res]); Tt = np.array([np.nan if 'their' not in r or r['their']['T'] is None else r['their']['T'] for r in res])
    okc = np.isfinite(Tc); okt = np.isfinite(Tt); cleant = np.array([('their' in r) and r['their']['ok'] and r['their']['viol_step'] == 0 for r in res]); cleanc = np.array([('their' in r) and r['their']['ok'] and r['their']['viol_step'] == 0 and r['their']['viol_cont'] <= 1e-3 for r in res])
    S = dict(env='car2026', n=n, cells=[len(l) for l in A.layers], nodes=int(A.N), cont_reach=float(okc.mean()), cont_ratio_med=float(np.nanmedian(Tc / Tref)) if okc.any() else None,
             their_reach=float(okt.mean()), their_clean_step=float(cleant.mean()), their_clean_cont=float(cleanc.mean()), their_ratio_med=float(np.nanmedian(Tt / Tref)) if okt.any() else None,
             their_T_med_steps=float(np.nanmedian(Tt) / DT) if okt.any() else None, t_query_med=float(np.median([r['t_query'] for r in res])), wall_parallel=wall, per_start=res, Tref=Tref.tolist())
    json.dump(S, open(os.path.join(common.DATA, 'env_car2026.json'), 'w'), indent=1, default=float); print({k: v for k, v in S.items() if k not in ('per_start', 'Tref')})
