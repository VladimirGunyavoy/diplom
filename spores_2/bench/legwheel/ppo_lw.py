"""линия D: PPO пользователя (чекпоинт 0073, SB3) на стартах бенча и на всей сетке eval. Проверка постановки: те же старты, шаг, исход, что metrics.json/episodes.csv.
python ppo_lw.py check   → вся сетка 41×41, сверка с episodes.csv;  python ppo_lw.py bench [N] → N=50 стартов бенча (T, успех, мс/запрос)"""
import os, sys, time, json, csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import env_lw as W, lw_dyn as L, common
from lw_dyn import C
from stable_baselines3 import PPO
RUN = os.environ.get('LWRUN') or os.path.abspath(os.path.join(HERE, '..', '..', '..', 'colab', 'runs', '003_165713'))
model = PPO.load(os.path.join(RUN, 'weights', '0073', 'model.zip'), device='cpu')
def obs_of(y): return np.array([y[0], y[1], y[2] - C.ALPHA_STAR, y[3], y[4], y[5]], np.float32)
def run_ppo(y0, tmax=C.EPISODE_TIMEOUT):
    """эпизод политики ровно как env робота: 1 действие на шаг .002, deterministic; → (T, причина, шагов, список действий)"""
    from legwheel_robot import env as E
    y = list(y0); acts = []; nmax = int(round(tmax / C.PHYSICS_DT))
    for k in range(nmax):
        a, _ = model.predict(obs_of(y), deterministic=True); tw, th = float(a[0]), float(a[1]); acts.append((tw, th))
        tw = max(-C.MAX_TORQUE_WHEEL, min(C.MAX_TORQUE_WHEEL, tw)); th = max(-C.MAX_TORQUE_HIP, min(C.MAX_TORQUE_HIP, th))
        try: y = E.rk4_step(y, tw, th, C.PHYSICS_DT)
        except ZeroDivisionError: return None, 'diverged', k + 1, acts
        if W.oob_phys(y): return None, 'oob', k + 1, acts
        if W.at_goal_phys(y): return (k + 1) * C.PHYSICS_DT, 'ok', k + 1, acts
    return None, 'timeout', nmax, acts
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'check':
        rows = list(csv.DictReader(open(os.path.join(RUN, 'evals', 'data', '0073', 'episodes.csv')))); Ts = []; dif = 0; t0 = time.perf_counter(); nok = 0
        for r in rows:
            T, why, n, _ = run_ppo(W.phys_start(float(r['psi0']), float(r['psi_dot0']))); ref = float(r['duration_s'])
            nok += (why == 'ok' and r['outcome'] == 'success'); Ts.append(T if T is not None else np.nan)
            if why != 'ok' or abs(T - ref) > 1e-9: dif += 1
        Ts = np.array(Ts); print('эпизодов', len(rows), 'успех', nok, 'расхождений с episodes.csv', dif, 'T ср %.4f (metrics 0.44555)' % np.nanmean(Ts), 'всего %.1f с' % (time.perf_counter() - t0), flush=True)
    if mode == 'diff':
        rows = list(csv.DictReader(open(os.path.join(RUN, 'evals', 'data', '0073', 'episodes.csv')))); d = []
        for r in rows:
            T, why, n, _ = run_ppo(W.phys_start(float(r['psi0']), float(r['psi_dot0']))); d.append(round((T - float(r['duration_s'])) / C.PHYSICS_DT))
        d = np.array(d); print('разница в шагах: значения', dict(zip(*np.unique(d, return_counts=True))))
    if mode == 'bench':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50; e = W.LW(n); R = []
        for i in range(n):
            y0 = L.to_phys(e.X0[i]); t0 = time.perf_counter(); T, why, k, acts = run_ppo(y0); dt = time.perf_counter() - t0
            R.append(dict(i=i, T=T, why=why, steps=k, t=dt, t_act_ms=1e3 * dt / max(k, 1)))
        # время ОДНОГО решения (инференс сети) отдельно
        o = obs_of(L.to_phys(e.X0[0])); t0 = time.perf_counter()
        for _ in range(2000): model.predict(o, deterministic=True)
        t_inf = (time.perf_counter() - t0) / 2000
        s = common.summarize(e, [r['T'] for r in R]); s.update(env='lw', method='PPO пользователя (0073, 9.57M шагов), deterministic, 1 действие/.002 с', t_infer_ms=1e3 * t_inf, per_start=R, hw='CPU 1 процесс',
                  n_params=int(sum(p.numel() for p in model.policy.parameters())))
        json.dump(s, open(os.path.join(common.DATA, 'lw_ppo.json'), 'w'), default=float); print({k: v for k, v in s.items() if k != 'per_start'})
