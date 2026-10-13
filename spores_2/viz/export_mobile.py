"""viz (линия B): выгрузка траекторий дифдрайва/машины в viz/data/<система>.json (формат — viz/README.md).
Запуск (хаб, CPU; ~2 мин на систему):  python viz/export_mobile.py dd|ddobst [--starts 4 --png]
Методы: «споры» — реплика rollout() с записью (u, dt)-сегментов (bench/spores_dd.py), траектория пересчитана единой динамикой bench/common.py: Env.step (RK4, шаг ≤ .005)
и снята на сетку dt = 0.02 с. «эталон» — пока из файла (viz/data/ref_<система>.npz, если есть): OCP CasADi (bench/ocp_casadi.py), см. export_ref ниже."""
import os, sys, json, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(HERE, '..', 'bench'); sys.path.insert(0, BENCH)
DT = .02
SRC = dict(dd='dd_spores.json', ddobst='ddobst_spores.json', car='car_spores.json', carfree='')        # главная строка бенча (table_dd.md / table_ddobst7.md / table_car.md): правило research-27

def replay(env, y0, segs, dt=DT, h=.005):
    """(u_k, d_k) → сетка состояний с шагом dt: X (K+1, n) с непрерывным θ (развёрнутым), U (K, m); обрезка по попаданию в цель. Возврат X, U, T (время попадания или None)."""
    y = np.asarray(y0, float)[None].copy(); t = 0.; ts = [0.]; ys = [y[0].copy()]; us_ = []; T = None
    if env.in_goal(y)[0]: return np.array(ys), np.zeros((0, env.m)), 0.
    for u, d in segs:
        u = env.clip_u(np.asarray(u, float).reshape(1, -1)); k = max(1, int(np.ceil(d / h - 1e-9))); hh = d / k
        for _ in range(k):
            y = env.step(y, u, hh); t += hh; ts.append(t); ys.append(y[0].copy()); us_.append((t, u[0].copy()))
            if env.in_goal(y)[0]: T = t; break
            if env.out_of_box(y)[0]: break                                           # провал (выход из коробки / касание диска) — показываем путь до него, T = None
        if T is not None or env.out_of_box(y)[0]: break
    ys = np.array(ys); th = env.period > 0
    for j in np.flatnonzero(th): ys[:, j] = np.unwrap(ys[:, j])
    tg = np.arange(0., t + 1e-9, dt); X = np.stack([np.interp(tg, ts, ys[:, j]) for j in range(ys.shape[1])], 1)
    tu = np.array([a for a, _ in us_]); Uu = np.array([b for _, b in us_]); U = np.stack([Uu[np.minimum(np.searchsorted(tu, tg[:-1] + 1e-9), len(tu) - 1), j] for j in range(Uu.shape[1])], 1)
    return X, U, (T if T is not None else None)

def lst(a, nd=4): return np.round(np.asarray(a, float), nd).tolist()

def episode(env, i, method, segs, tref, src=''):
    X, U, T = replay(env, env.X0[i], segs)
    if X is None or len(X) < 2: return dict(method=method, start=int(i), dt=DT, X=[], U=[], T=None, T_ref=float(tref), ok=False, src=src)
    return dict(method=method, start=int(i), dt=DT, X=lst(X), U=lst(U, 3), T=None if T is None else round(float(T), 3), T_ref=round(float(tref), 3), ok=bool(T is not None), src=src)

def pick(env, n):
    """показательные старты: дифдрайв — большое рассогласование курса с направлением на цель и далеко (манёвр с разворотом) + самый длинный; прочее — по T_ref"""
    X = env.X0
    if X.shape[1] == 3 and len(X) > n:
        mis = np.abs((np.arctan2(-X[:, 1], -X[:, 0]) - X[:, 2] + np.pi) % (2 * np.pi) - np.pi); sc = mis * np.hypot(X[:, 0], X[:, 1]) ** .5 + .15 * env.Tref
        o = list(np.argsort(-sc)); out = [int(o[0]), int(np.argmax(env.Tref))] + [int(k) for k in o[1:]]
    else: out = [int(k) for k in np.argsort(-env.Tref)]
    r = []
    for k in out:
        if k not in r: r.append(k)
    return r[:n]

def export_dd(name, nstarts, ids=None):
    os.environ['BENCH_ENV'] = {'dd': 'dd', 'ddobst': 'ddobst'}[name]
    import spores_dd as S, common
    e = common.make(os.environ['BENCH_ENV']); A = S.G.Atlas(); A.solve(); eps = []
    for i in (ids or pick(e, nstarts)):
        T, seg = S.rollout_rec(A, e.X0[i:i + 1]); eps.append(episode(e, i, 'споры', [(u, d) for u, d in seg[0]], e.Tref[i], SRC[name])); print(name, i, eps[-1]['T'], eps[-1]['T_ref'], len(eps[-1]['X']), flush=True)
    d = dict(sys=name, title={'dd': 'Дифдрайв: |v|+|ω| ≤ 1', 'ddobst': 'Дифдрайв с двумя препятствиями'}[name],
             params=dict(body_r=.2, wheel_base=.34, fov=[-3, 3, -3, 3], view=[-3, -3, 3, 3], umax=[1, 1]), goal=dict(type='box', c=[0, 0], half=[.05, .05], theta=0, half_theta=.05),
             obstacles=[dict(type='disc', c=[float(c[0]), float(c[1])], r=float(c[2])) for c in getattr(e, 'DISKS', [])], episodes=eps)
    d['params']['obstacles'] = d['obstacles']; d['params']['goal'] = d['goal']                # draw(…, params, …) видит и препятствия, и цель
    return d

def finalize(d):
    """камера по данным: ограничивающий прямоугольник всех траекторий (все методы) + цель + диски, поле 12 %; размер тела — доля от размаха (читаемо при любом масштабе)"""
    P = [np.array(e['X'])[:, :2] for e in d['episodes'] if len(e['X'])]; g = d['goal']; r = g.get('r', .15) if g['type'] == 'disc' else .15
    P.append(np.array([[g['c'][0] - r, g['c'][1] - r], [g['c'][0] + r, g['c'][1] + r]]))
    for o in d.get('obstacles', []): P.append(np.array([[o['c'][0] - o['r'], o['c'][1] - o['r']], [o['c'][0] + o['r'], o['c'][1] + o['r']]]))
    A = np.concatenate(P); lo, hi = A.min(0), A.max(0); span = (hi - lo).max(); m = .12 * span; lo, hi = lo - m, hi + m; p = d['params']; p['view'] = [round(float(lo[0]), 2), round(float(lo[1]), 2), round(float(hi[0]), 2), round(float(hi[1]), 2)]
    sp = float((hi - lo).max())
    if d['sys'].startswith('dd'): p['body_r'] = round(.045 * sp, 3); p['wheel_base'] = round(.075 * sp, 3); p['umax'] = [1, 1]
    else: p['car_len'] = round(.075 * sp, 3); p['car_wid'] = round(.042 * sp, 3); p['umax'] = [1.5, 1.8]
    p['obstacles'] = d['obstacles']; p['goal'] = d['goal']
    return d

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('system'); ap.add_argument('--starts', type=int, default=4); ap.add_argument('--ids', default=''); a = ap.parse_args()
    d = finalize(export_dd(a.system, a.starts, [int(k) for k in a.ids.split(',')] if a.ids else None)); p = os.path.join(HERE, 'data', a.system + '.json'); json.dump(d, open(p, 'w'), ensure_ascii=False, separators=(',', ':')); print(p, os.path.getsize(p) // 1024, 'КБ')
