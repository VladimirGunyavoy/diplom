"""линия E: экспорт данных визуализации для dp1 (двойной маятник), manip (2 зв. 4D), manipf (+ стол и самостолкновение). ТРЕБУЕТ код спор (aida, не ноут).
python export_arm2.py spores SYS [старты через запятую]  — споры: атлас из PKL, запись (u, dt) как rollout_rec в bench/spores_manip.py; venv-sb3
python export_arm2.py ocp    SYS [старты]               — эталон: dp1 — OCP r23 (ref.solve_one, 20/20); manip/manipf — CasADi+IPOPT (bench/ocp_casadi.build, свободный T)
Оба этапа дописывают эпизоды своего метода в viz/data/SYS.json (остальные не трогают). Траектории пересчитываются единым проверяльщиком (common.Env.step) на сетке DT=.02.
Главные строки бенча (compare.md/table_*.md): dp1 — dp1_spores_load.json (7/20, атлас w25 dp1_V2000.pkl), manip — manip_spores_load.json (A_g300so.pkl), manipf — manipf_spores_load_f2b.json (A_f2b.pkl, FORB=1)."""
import os, sys, json, time, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__)); BENCH = os.path.join(HERE, '..', 'bench'); sys.path.insert(0, BENCH); sys.path.insert(0, HERE)
import numpy as np
PI = np.pi; DT = .02; OUT = os.path.join(HERE, 'data')
W25 = os.path.expanduser('~/spore_v5/w25'); B11 = os.path.expanduser('~/spore_v5/b11')
CFG = dict(   # env-переменные до импорта spores_manip, атлас, файл главной строки бенча
    dp1=dict(env=dict(SYS='dp1', ENVN='dp1', SRC_DIR=W25 + '/v7/src/cells7', PKL=W25 + '/dp1_V2000.pkl', WARM='1'), src='dp1_spores_load.json'),
    manip=dict(env=dict(ENVN='manip', WARM='1'), src='manip_spores_load.json'),
    manipf=dict(env=dict(ENVN='manipf', FORB='1', WARM='1', SRC_DIR='~/spore_v5/c1/maze_src', PKL=B11 + '/A_f2b.pkl'), src='manipf_spores_load_f2b.json'))
TITLES = dict(dp1='Двойной маятник, g = 1, |τ| ≤ 1: в стойку', manip='Манипулятор 2 звена, без g, |τ| ≤ 1', manipf='Манипулятор: стол и самостолкновение')
VIEW = dict(dp1=[-2.9, -2.4, 2.9, 2.4], manip=[-2.7, -2.3, 2.7, 2.3], manipf=[-2.7, -2.3, 2.7, 2.3])

def load(sysn):
    fn = os.path.join(OUT, sysn + '.json')
    if os.path.exists(fn): return json.load(open(fn))
    p = dict(links=[1., 1.], umax=[1., 1.], view=VIEW[sysn], unames=['плечо', 'локоть'], q_off=[PI / 2 if sysn == 'dp1' else 0., 0.])
    obst = []
    if sysn == 'manipf': p.update(table_y=-.6, fold=.5); obst = [dict(type='halfplane', n=[0., -1.], d=.6)]          # n·p ≥ d ⇒ y ≤ −.6 запрещено
    if sysn == 'dp1': p['gravity'] = True
    return dict(sys=sysn, title=TITLES[sysn], params=p, goal=dict(q=[PI / 2 if sysn == 'dp1' else 0., 0.]), obstacles=obst, episodes=[])
def save(sysn, data):
    os.makedirs(OUT, exist_ok=True); fn = os.path.join(OUT, sysn + '.json'); json.dump(data, open(fn, 'w'), ensure_ascii=False, separators=(',', ':')); print('записано', fn, os.path.getsize(fn), 'байт', flush=True)
def put(sysn, method, eps):
    d = load(sysn); d['episodes'] = [e for e in d['episodes'] if e['method'] != method] + eps; d['episodes'].sort(key=lambda e: (e['start'], e['method'] != 'споры')); save(sysn, d)
def ep(method, i, X, U, Tv, Tref, src): return dict(method=method, start=int(i), dt=DT, X=np.round(X, 4).tolist(), U=np.round(U, 3).tolist(), T=Tv, T_ref=float(Tref), ok=Tv is not None, src=src)

def pick(sysn, nshow=5):
    """показательные старты из главной строки: успешные с большим T_ref (длинные раскачки) + один честный провал спор, если есть"""
    d = json.load(open(os.path.join(HERE, '..', '..', 'reports', 'bench', 'data', CFG[sysn]['src']))); ps = d['per_start']; Tr = np.array(d['Tref']); ok = [p['i'] for p in ps if p['T_verify'] is not None]
    keep = ok if sysn != 'manipf' else [p['i'] for p in ps if p['T_verify'] is not None]
    good = sorted(keep, key=lambda i: -Tr[i])[:nshow]; bad = [p['i'] for p in ps if p['T_verify'] is None and (sysn != 'manipf' or p.get('why') != 'start_forbidden')]
    return sorted(good + (bad[:1] if sysn == 'dp1' else []))

def run_spores(sysn, starts):
    for k, v in CFG[sysn]['env'].items(): os.environ.setdefault(k, os.path.expanduser(v))
    import common, spores_manip as SP, growN as G, pickle
    e = common.make(sysn); d = pickle.load(open(os.path.expanduser(os.environ.get('PKL', '~/spore_v5/wb5/m45/A_g300so.pkl')), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish(); A.V = d['V']; A.n_it = 0; A.edges = 0
    SP.rollout_rec(A, e.X0[:1]); eps = []                                                    # прогрев (не в замере)
    for i in starts:
        T, seg, nodur = SP.rollout_rec(A, e.X0[i:i + 1]); sg = seg[0]
        X, U, Tv = resample(e, e.X0[i], [list(u) for u, dd in sg], [dd for u, dd in sg])
        if len(U) == 0: U = np.zeros((1, e.m))
        print('споры', sysn, i, 'T', Tv, 'T_ref', round(float(e.Tref[i]), 3), 'без длительностей финиша' if nodur[0] else '', flush=True)
        eps.append(ep('споры', i, X, U, Tv, e.Tref[i], CFG[sysn]['src']))
    put(sysn, 'споры', eps)

def run_ocp(sysn, starts):
    import common; e = common.make(sysn); eps = []
    for i in starts:
        if sysn == 'dp1': r = ocp_dp1(e, i)
        else: r = ocp_arm(e, i)
        if r is None: print('эталон', sysn, i, 'не нашёлся', flush=True); continue
        T, Ur = r; X, U, Tv = resample(e, e.X0[i], list(Ur), [T / len(Ur)] * len(Ur)); print('эталон', sysn, i, 'T', Tv, 'T_ref', round(float(e.Tref[i]), 3), flush=True)
        eps.append(ep('эталон (OCP)', i, X, U, Tv, e.Tref[i], 'r23 ref.solve_one (CasADi/IPOPT N=160)' if sysn == 'dp1' else 'bench/ocp_casadi (CasADi+IPOPT N=60, свободный T)'))
    put(sysn, 'эталон (OCP)', eps)

def ocp_dp1(e, q):
    R23 = os.path.expanduser('~/spore_v5/r23/06_dp1_ref'); sys.path.insert(0, R23); sys.argv = ['ref.py', '160', '4']; import ref
    Q = ref.starts(); dd = json.loads(open('%s/out_%d.jsonl' % (R23, q)).readline()); k, g = dd['k'], dd['g']; rng = np.random.default_rng(1000 * q + 10 * g + 3 * k[0] + k[1] + 4); x0 = Q[q]
    T0 = 2.5 + .7 * np.abs(np.r_[np.asarray(ref.C3) + 2 * np.pi * np.asarray(k) - x0[:2]]).max(); st, T, U = ref.solve_one(x0, k, rng, T0); return T, np.asarray(U)

def ocp_arm(e, i, N=60):
    """как ocp_casadi.solve, но возвращает (T, U) лучшего прошедшего проверку решения"""
    import ocp_casadi as OC, common
    y0 = np.asarray(e.X0[i], float); rng = np.random.default_rng(100 + i); opti, T, X, U, px0, ptg = OC.build(e, N); per = [j for j in range(e.n) if e.period[j] > 0]; ks = []; g0 = np.zeros(e.n)
    for j in per: P = e.period[j]; ks.append([g0[j] + np.floor((y0[j] - g0[j]) / P) * P, g0[j] + np.ceil((y0[j] - g0[j]) / P) * P])
    best = None
    for comb in itertools.product(*ks):
        tg = g0.copy()
        for j, v in zip(per, comb): tg[j] = v
        for T0 in ((1.5, 2.5, 4., 6., 10.) if e.name == 'manipf' else (3., 6., 10.)):
            opti.set_value(px0, y0); opti.set_value(ptg, tg); a = np.linspace(0, 1, N + 1); opti.set_initial(X, y0[:, None] + (tg - y0)[:, None] * a[None]); opti.set_initial(U, rng.uniform(-.1, .1, (e.m, N)) * e.umax[:, None]); opti.set_initial(T, T0)
            try: opti.solve()
            except RuntimeError: pass
            Tv = float(opti.debug.value(T)); Uv = np.asarray(opti.debug.value(U)).reshape(e.m, N)
            if not (np.isfinite(Tv) and Tv >= .05 - 1e-6 and np.isfinite(Uv).all()): continue
            Tc, why = common.verify(e, y0, Uv.T, np.full(N, Tv / N))
            if Tc is not None and (best is None or Tc < best[0]): best = (Tc, Tv, Uv.T.copy())
    return None if best is None else (best[1], best[2])

if __name__ == '__main__':
    from export_links import resample
    stage, sysn = sys.argv[1], sys.argv[2]; st = [int(v) for v in sys.argv[3].split(',')] if len(sys.argv) > 3 else pick(sysn)
    print('старты', st, flush=True)
    {'spores': run_spores, 'ocp': run_ocp}[stage](sysn, st)
