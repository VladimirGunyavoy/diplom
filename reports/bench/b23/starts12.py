"""b23: стартовые состояния сидов 200000–99 (позиция, курс) + сравнение дерево 12 против остальных по сиду"""
import sys, json, glob, numpy as np
sys.path.insert(0, '.'); import canal as C
env = C.mkenv(); t = env.unwrapped.task; St = []
for s in range(200000, 200100):
    env.reset(seed=s); p = t.agent.pos; R_ = t.agent.mat; th = np.arctan2(R_[1, 0], R_[0, 0]); St.append((np.hypot(p[0], p[1]), np.arctan2(p[1], p[0]), (th - np.arctan2(p[1], p[0])) % (2 * np.pi)))
St = np.array(St); np.save('starts.npy', St); print('ok')
