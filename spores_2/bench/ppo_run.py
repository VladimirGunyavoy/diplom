"""C4: PPO (SB3) на Env bench. python ppo_run.py ENV STEPS SEED  (aida: ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python, CPU)
Награда: −dt за шаг + потенциал (Ng–Harada–Russell, Φ=−‖embed(y)/масштаб‖) + бонус цели; конец — цель / выход из коробки / 3·T_max. Оценка: детерминированный rollout с тех же стартов → (u_k,dt_k) → verify(). Каждые EVAL шагов — успех/T/T_ref (t_train до 95%)."""
import os, sys, time, json, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import common
DT = float(os.environ.get('PPO_DT', .05)); TMAXEP = float(os.environ.get('PPO_TEP', 20.)); GAMMA = .99

class BenchGym(gym.Env):
    def __init__(s, env, starts=None):
        s.e = env; s.sc = (env.hi - env.lo) / 2; d = s.e.embed(np.zeros(env.n)).shape[0]
        s.observation_space = gym.spaces.Box(-10, 10, (d,), np.float32); s.action_space = gym.spaces.Box(-1, 1, (env.m,), np.float32); s.rng = np.random.default_rng()
        s.span = 0.5 * (env.umax - env.umin); s.mid = 0.5 * (env.umax + env.umin)
    def phi(s, y): return -np.linalg.norm(s.e.embed(y) if s.e.period.max() > 0 else y / s.sc)
    def reset(s, seed=None, options=None):
        if seed is not None: s.rng = np.random.default_rng(seed)
        while True:
            y = s.e.sample_field(s.rng)
            if not s.e.in_goal(y): break
        s.y = y; s.t = 0.; return s.e.embed(y).astype(np.float32), {}
    def step(s, a):
        u = s.e.clip_u(s.mid + s.span * np.clip(a, -1, 1))[None]; y2 = s.e.step(s.y[None], u, DT)[0]; r = -DT + GAMMA * s.phi(y2) - s.phi(s.y); s.y = y2; s.t += DT
        goal = bool(s.e.in_goal(y2)); box = bool(s.e.out_of_box(y2)); 
        if goal: r += 5.
        if box: r -= 5.
        return s.e.embed(y2).astype(np.float32), float(r), goal or box, s.t >= TMAXEP, {}

def rollout(model, vn, e, y0, Tref):
    """детерминированная политика, шаг DT → (us, dts); верх по времени 3·T_ref. Время — только прокатка (без verify)"""
    y = np.array(y0, float); us = []; env = BenchGym(e); n = int(np.ceil(3 * Tref / DT))
    for _ in range(n):
        o = e.embed(y).astype(np.float32); o = vn.normalize_obs(o[None]) if vn is not None else o[None]; a, _ = model.predict(o, deterministic=True); u = e.clip_u(env.mid + env.span * np.clip(a[0], -1, 1)); us.append(u); y = e.step(y[None], u[None], DT)[0]
        if e.in_goal(y[None])[0] or e.out_of_box(y[None])[0]: break
    return np.array(us), np.full(len(us), DT)

def rollout_batch(model, vn, e, Y0, Tref):
    """все старты разом (пакет): predict на (K, obs)"""
    env = BenchGym(e); Y = np.array(Y0, float); K = len(Y); alive = np.ones(K, bool)
    for _ in range(int(np.ceil(3 * Tref.max() / DT))):
        if not alive.any(): break
        o = e.embed(Y).astype(np.float32); o = vn.normalize_obs(o) if vn is not None else o; a, _ = model.predict(o, deterministic=True); u = e.clip_u(env.mid + env.span * np.clip(a, -1, 1)); Y2 = e.step(Y, u, DT)
        Y = np.where(alive[:, None], Y2, Y); alive &= ~(e.in_goal(Y) | e.out_of_box(Y))

def evaluate(model, vn, e, timing=False):
    """→ Ts (проверяльщик вне таймера); при timing — ещё t_roll (с на старт, по одному), t_batch (с, пакет всех стартов)"""
    R = []; tr = []
    for y0, Tr in zip(e.X0, e.Tref):
        t1 = time.perf_counter(); us, dts = rollout(model, vn, e, y0, Tr); tr.append(time.perf_counter() - t1); R.append((y0, us, dts))
    Ts = [common.verify(e, y0, us, dts)[0] for y0, us, dts in R]
    if not timing: return Ts
    t1 = time.perf_counter(); rollout_batch(model, vn, e, e.X0, e.Tref); tb = time.perf_counter() - t1
    return Ts, float(np.mean(tr)), float(tb)

class EvalCb(BaseCallback):
    def __init__(s, e, vn_ref, every, log): super().__init__(); s.e = e; s.vn = vn_ref; s.every = every; s.log = log; s.t0 = time.time(); s.next = every; s.hit95 = None
    def _on_step(s):
        if s.num_timesteps >= s.next:
            s.next += s.every; Ts = evaluate(s.model, s.vn, s.e); r = common.summarize(s.e, Ts); r.update(steps=s.num_timesteps, t_train=time.time() - s.t0); s.log.append(r)
            print('EVAL', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if s.hit95 is None and r['reached'] >= .95: s.hit95 = r['t_train']
        return True

if __name__ == '__main__':
    name, steps, seed = sys.argv[1], int(float(sys.argv[2])), int(sys.argv[3]); e = common.make(name); NE = 16
    venv = VecNormalize(DummyVecEnv([lambda: BenchGym(common.make(name)) for _ in range(NE)]), norm_obs=True, norm_reward=True, gamma=GAMMA)
    model = PPO('MlpPolicy', venv, n_steps=256, batch_size=1024, n_epochs=10, gamma=GAMMA, gae_lambda=.95, learning_rate=3e-4, ent_coef=0., clip_range=.2, seed=seed, device='cpu', verbose=0, policy_kwargs=dict(net_arch=[64, 64]))
    log = []; cb = EvalCb(e, venv, max(steps // 20, 5000), log); t0 = time.time(); model.learn(steps, callback=cb); t_train = time.time() - t0
    Ts, t_inf, t_batch = evaluate(model, venv, e, timing=True); s = common.summarize(e, Ts); e.nf = 0; e.step(e.X0[:1], np.zeros((1, e.m)), DT); nfs = e.nf
    s.update(env=name, method='PPO SB3', steps=steps, seed=seed, t_train=t_train, t_train_95=cb.hit95, t_infer_per_start=t_inf, t_infer_batch20=t_batch, nf_train=steps * nfs, nf_per_step=nfs, hw='CPU 1 proc, 16 envs Dummy', curve=log, T=[None if t is None else float(t) for t in Ts], Tref=e.Tref.tolist())
    os.makedirs(common.DATA, exist_ok=True); json.dump(s, open(os.path.join(common.DATA, f'{name}_ppo_{steps}_s{seed}.json'), 'w'), indent=1, default=float); os.makedirs(os.path.expanduser('~/spore_v5/c1/models'), exist_ok=True); model.save(os.path.expanduser(f'~/spore_v5/c1/models/{name}_{steps}_s{seed}')); venv.save(os.path.expanduser(f'~/spore_v5/c1/models/{name}_{steps}_s{seed}_vn.pkl')); print('DONE', {k: v for k, v in s.items() if k not in ('curve', 'T', 'Tref')})
