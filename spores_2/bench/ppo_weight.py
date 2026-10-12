"""вес PPO: число параметров, байты state_dict (float32) и VecNormalize (pkl) по моделям 5M сид 0 → data/ppo_weight.json. venv-sb3."""
import os, json, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
import common
M = os.path.expanduser('~/spore_v5/c1/models'); out = {}
for env in ('di', 'pend', 'dd', 'manip', 'ddobst', 'car'):
    p = f'{M}/{env}_5000000_s0'
    if not os.path.exists(p + '.zip'): continue
    m = PPO.load(p, device='cpu'); n = sum(x.numel() for x in m.policy.parameters()); b = sum(x.numel() * x.element_size() for x in m.policy.state_dict().values())
    out[env] = dict(params=n, bytes=b, vecnorm_bytes=os.path.getsize(p + '_vn.pkl'), zip_bytes=os.path.getsize(p + '.zip'))
json.dump(out, open(os.path.join(common.DATA, 'ppo_weight.json'), 'w'), indent=1); print(out)
