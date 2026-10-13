"""CA4: SAC + HER (SB3 HerReplayBuffer, стратегия future, 4 цели) — стандарт RL для «цель на входе» (совет research-25, Andrychowicz 2017). Одна тренировка на все цели:
случайные цели и старты (с w из поля), оценка на переходах A (40) и B (20) тем же проверяльщиком. Разрежённая награда 0 в цели / −1 за шаг (DT=.05), выход из коробки (|ω|>3) — конец с штрафом −(шагов эпизода + 100) (= −(T_ep+5 с)).
Наблюдение: embed(y) (cos/sin q, w/3); achieved = embed(y); desired = embed цели (cos/sin q, 0, 0). python mg_sac.py STEPS SEED (aida, venv-sb3, CPU) → data/mg_sac_STEPS_sSEED.json"""
import os, sys, time, json, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
import common, manip_goals as MGm
PI = np.pi; DT = .05; TEP = 20.; PEN = float(os.environ.get('SAC_PEN', 0.)); GAM = float(os.environ.get('SAC_GAMMA', .99))
def emb(e, y): return e.embed(y).astype(np.float32)          # (cos q1, sin q1, cos q2, sin q2, w1/3, w2/3)
def goal_vec(g): return np.r_[np.cos(g[0]), np.sin(g[0]), np.cos(g[1]), np.sin(g[1]), 0., 0.].astype(np.float32)
def success(a, d):
    a = np.atleast_2d(a); d = np.atleast_2d(d); qa = np.stack([np.arctan2(a[:, 1], a[:, 0]), np.arctan2(a[:, 3], a[:, 2])], 1); qd = np.stack([np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[:, 3], d[:, 2])], 1)
    dq = np.abs((qa - qd + PI) % (2 * PI) - PI); return (dq <= .3).all(1) & (np.abs(a[:, 4:] * 3) <= .5).all(1)

class Prog(BaseCallback):
    def __init__(s, total): super().__init__(); s.tot = total; s.t0 = time.time(); s.last = 0; s.eps = []
    def _on_step(s):
        for d, i in zip(s.locals['dones'], s.locals['infos']):
            if d: s.eps.append(bool(i.get('is_success', False)))
        if time.time() - s.last > 30: s.last = time.time(); print('SAC %d/%d шагов, %.0f с, is_success (последние 100 эп.) %.2f, эп. %d' % (s.num_timesteps, s.tot, time.time() - s.t0, np.mean(s.eps[-100:]) if s.eps else 0., len(s.eps)), flush=True)
        return True

class GoalEnv(gym.Env):
    def __init__(s):
        s.e = MGm.MG([0., 0.], [1., 1.]); s.rng = np.random.default_rng(); b = gym.spaces.Box(-10, 10, (6,), np.float32)
        s.observation_space = gym.spaces.Dict(dict(observation=b, achieved_goal=b, desired_goal=b)); s.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    def obs(s): o = emb(s.e, s.y); return dict(observation=o, achieved_goal=o.copy(), desired_goal=goal_vec(s.e.goal))
    def reset(s, seed=None, options=None):
        if seed is not None: s.rng = np.random.default_rng(seed)
        s.e.goal = np.r_[s.rng.uniform(-PI, PI, 2), 0., 0.]
        while True:
            y = s.e.sample_field(s.rng)
            if not s.e.in_goal(y): break
        s.y = y; s.n = 0; return s.obs(), {}
    def step(s, a):
        s.y = s.e.step(s.y[None], np.clip(a, -1, 1)[None], DT)[0]; s.n += 1; goal = bool(s.e.in_goal(s.y[None])[0]); box = bool(s.e.out_of_box(s.y[None])[0])
        return s.obs(), 0. if goal else -1., goal or box, s.n * DT >= TEP and not goal, dict(is_success=goal, pen=((s.n + 100.) if box else 0.) * PEN)
    def compute_reward(s, achieved_goal, desired_goal, info):
        r = np.where(success(achieved_goal, desired_goal), 0., -1.)
        if info is not None:
            try: r = r - np.array([i.get('pen', 0.) if isinstance(i, dict) else 0. for i in info]) * (r < 0)
            except TypeError: pass
        return r.astype(np.float32)

def run_transitions(model, kind):
    tr, _ = MGm.transitions(kind); ref = {r['k']: r['Tref'] for r in json.load(open(os.path.join(common.DATA, 'mg_%s_ref.json' % kind)))}; out = []
    for k, (st, g) in enumerate(tr):
        e = MGm.MG(g, st); y = e.X0[0].copy(); us = []; t0 = time.perf_counter(); dg = goal_vec(g)
        for _ in range(int(np.ceil(3 * ref[k] / DT))):
            o = emb(e, y); a, _ = model.predict(dict(observation=o[None], achieved_goal=o[None], desired_goal=dg[None]), deterministic=True); u = np.clip(a[0], -1, 1); us.append(u); y = e.step(y[None], u[None], DT)[0]
            if e.in_goal(y[None])[0] or e.out_of_box(y[None])[0]: break
        ti = time.perf_counter() - t0; Tv, why = common.verify(e, e.X0[0], np.array(us), np.full(len(us), DT)); out.append(dict(k=k, Tref=ref[k], T_verify=Tv, why=why, t_query=ti))
    return out

if __name__ == '__main__':
    steps, seed = int(float(sys.argv[1])), int(sys.argv[2]); env = GoalEnv()
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'), learning_starts=2000, batch_size=256, gamma=GAM, train_freq=(4, 'step'), gradient_steps=2,
                policy_kwargs=dict(net_arch=[256, 256]), seed=seed, device='cpu', verbose=0)
    t0 = time.time(); model.learn(total_timesteps=steps, callback=Prog(steps)); tt = time.time() - t0; res = {}
    for kind in 'AB':
        R = run_transitions(model, kind); ok = [r for r in R if r['T_verify'] is not None and r['T_verify'] <= 3 * r['Tref']]
        res[kind] = dict(per=R, reached=len(ok) / len(R), ratio_med=float(np.median([r['T_verify'] / r['Tref'] for r in ok])) if ok else None); print(kind, 'дошли %.2f' % res[kind]['reached'], 'T/Tref мед', res[kind]['ratio_med'], flush=True)
    json.dump(dict(steps=steps, seed=seed, t_train=tt, gamma=GAM, res=res), open(os.path.join(common.DATA, 'mg_sac_%d_s%d.json' % (steps, seed)), 'w'), indent=1, default=float); print('MGSACDONE')
