"""CA4: PPO с целью на входе (одна тренировка на все цели: цели и старты случайные), оценка на переходах A (40) и B (20), как у спор/SST.
Вход: embed(y) + (cos, sin) углов цели (динамика манипулятора зависит от абсолютного q2); потенциал Φ = −‖embed(wrap(y − goal))‖; награда/штрафы как ppo_run (γ, BOXPEN=25 из PPO_*).
python mg_ppo.py STEPS SEED (aida, venv-sb3, CPU) → data/mg_ppo_STEPS_sSEED.json (t_train, на переход T_verify, t_infer)."""
import os, sys, time, json, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import common, ppo_run as P, manip_goals as MGm
PI = np.pi; DT = P.DT; GAMMA = P.GAMMA; BOXPEN = P.BOXPEN

class Prog(BaseCallback):
    def __init__(s, total): super().__init__(); s.tot = total; s.t0 = time.time(); s.last = 0
    def _on_step(s):
        if time.time() - s.last > 30: s.last = time.time(); print('PPO %d/%d шагов, %.0f с' % (s.num_timesteps, s.tot, time.time() - s.t0), flush=True)
        return True

def obs_of(e, y, g): return np.r_[e.embed(y), np.cos(g[:2]), np.sin(g[:2])].astype(np.float32)
class GoalGym(gym.Env):
    def __init__(s):
        s.e = MGm.MG([0., 0.], [1., 1.]); s.rng = np.random.default_rng(); s.sc = (s.e.hi - s.e.lo) / 2; d = len(obs_of(s.e, np.zeros(4), np.zeros(4)))
        s.observation_space = gym.spaces.Box(-10, 10, (d,), np.float32); s.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    def phi(s, y): return -np.linalg.norm(s.e.embed(common.wrap(y - s.e.goal, s.e.period)))
    def reset(s, seed=None, options=None):
        if seed is not None: s.rng = np.random.default_rng(seed)
        s.e.goal = np.r_[s.rng.uniform(-PI, PI, 2), 0., 0.]
        while True:
            y = s.e.sample_field(s.rng)
            if not s.e.in_goal(y): break
        s.y = y; s.t = 0.; return obs_of(s.e, y, s.e.goal), {}
    def step(s, a):
        u = np.clip(a, -1, 1)[None]; y2 = s.e.step(s.y[None], u, DT)[0]; r = -DT + GAMMA * s.phi(y2) - s.phi(s.y); s.y = y2; s.t += DT
        goal = bool(s.e.in_goal(y2)); box = bool(s.e.out_of_box(y2))
        if goal: r += 5.
        if box: r -= BOXPEN
        return obs_of(s.e, y2, s.e.goal), float(r), goal or box, s.t >= P.TMAXEP, {}

def run_transitions(model, vn, kind):
    tr, _ = MGm.transitions(kind); ref = {r['k']: r['Tref'] for r in json.load(open(os.path.join(common.DATA, 'mg_%s_ref.json' % kind)))}; out = []
    for k, (st, g) in enumerate(tr):
        e = MGm.MG(g, st); y = e.X0[0].copy(); us = []; t0 = time.perf_counter()
        for _ in range(int(np.ceil(3 * ref[k] / DT))):
            o = obs_of(e, y, e.goal)[None]; o = vn.normalize_obs(o) if vn is not None else o; a, _ = model.predict(o, deterministic=True); u = np.clip(a[0], -1, 1); us.append(u); y = e.step(y[None], u[None], DT)[0]
            if e.in_goal(y[None])[0] or e.out_of_box(y[None])[0]: break
        ti = time.perf_counter() - t0; Tv, why = common.verify(e, e.X0[0], np.array(us), np.full(len(us), DT)); out.append(dict(k=k, Tref=ref[k], T_verify=Tv, why=why, t_query=ti))
    return out

if __name__ == '__main__':
    steps, seed = int(float(sys.argv[1])), int(sys.argv[2]); NE = 16
    venv = VecNormalize(DummyVecEnv([GoalGym for _ in range(NE)]), norm_obs=True, norm_reward=True, gamma=GAMMA)
    model = PPO('MlpPolicy', venv, n_steps=256, batch_size=1024, n_epochs=10, gamma=GAMMA, gae_lambda=.95, learning_rate=3e-4, ent_coef=0., clip_range=.2, seed=seed, device='cpu', verbose=0, policy_kwargs=dict(net_arch=[64, 64]))
    t0 = time.time(); model.learn(total_timesteps=steps, callback=Prog(steps)); tt = time.time() - t0; venv.training = False; res = {}
    for kind in 'AB':
        R = run_transitions(model, venv, kind); ok = [r for r in R if r['T_verify'] is not None and r['T_verify'] <= 3 * r['Tref']]
        res[kind] = dict(per=R, reached=len(ok) / len(R), ratio_med=float(np.median([r['T_verify'] / r['Tref'] for r in ok])) if ok else None)
        print(kind, 'дошли %.2f' % res[kind]['reached'], 'T/Tref мед', res[kind]['ratio_med'], flush=True)
    json.dump(dict(steps=steps, seed=seed, t_train=tt, gamma=GAMMA, boxpen=BOXPEN, res=res), open(os.path.join(common.DATA, 'mg_ppo_%d_s%d.json' % (steps, seed)), 'w'), indent=1, default=float); print('MGPPODONE')
