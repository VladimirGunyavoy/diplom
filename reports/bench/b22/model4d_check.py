"""b22 канава, условие (1): ошибка 4D (x,y,θ,v; скорость вдоль курса) и 5D (x,y,θ,vx,vy) моделей к MuJoCo SafetyPointCircle2 за 2 с, управления {-1,0,1}², hold .4 с."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, safety_gymnasium
M_, A_, CD, FM, WS = .005188790204786391, .0192723151357623, .01, .015, 3.0
def f5(p, u, T, h=.01):
    x, y, th, vx, vy = p; w = WS * u[1]; F = FM * u[0]
    for _ in range(int(round(T / h))):
        c, s = np.cos(th), np.sin(th); ax = (F*c - CD*vx)/M_ + A_*w*w*c; ay = (F*s - CD*vy)/M_ + A_*w*w*s
        vxm, vym, thm = vx + .5*h*ax, vy + .5*h*ay, th + .5*h*w; cm, sm = np.cos(thm), np.sin(thm)
        axm = (F*cm - CD*vxm)/M_ + A_*w*w*cm; aym = (F*sm - CD*vym)/M_ + A_*w*w*sm
        x += h*vxm; y += h*vym; th += h*w; vx += h*axm; vy += h*aym
    return np.array([x, y, th, vx, vy])
def f4(p, u, T, h=.01):
    x, y, th, v = p; w = WS * u[1]; F = FM * u[0]
    for _ in range(int(round(T / h))):
        v += h * (F - CD * v) / M_; x += h * v * np.cos(th + .5*h*w); y += h * v * np.sin(th + .5*h*w); th += h * w
    return np.array([x, y, th, v])
e = safety_gymnasium.make('SafetyPointCircle2-v0'); e.reset(seed=1); t = e.unwrapped.task; d = t.data
rng = np.random.default_rng(7); E4, E5, P = [], [], []
for i in range(60):
    e.reset(seed=100 + i); d = t.data; R = t.agent.mat; th0 = np.arctan2(R[1, 0], R[0, 0]); p5 = np.array([d.qpos[0], d.qpos[1], d.qpos[2], d.qvel[0], d.qvel[1]]); p4 = np.array([p5[0], p5[1], p5[2], 0.])
    U = rng.integers(-1, 2, (5, 2)).astype(float)
    if i % 4 == 0: U[:, 0] = 1.
    p0 = p5[:2].copy(); L = 0.
    for u in U:
        for _ in range(20): e.step(u)
        p5 = f5(p5, u, .4); p4 = f4(p4, u, .4)
        # (MuJoCo-состояние для 5D сбрасывается в конце 2 с; ошибка — за ВСЕ 2 с без пере-привязки)
    d = t.data; q = np.array([d.qpos[0], d.qpos[1]]); E5.append(np.linalg.norm(p5[:2]-q)); E4.append(np.linalg.norm(p4[:2]-q)); P.append(np.linalg.norm(q-p0))
for n, E in (('5D', E5), ('4D', E4)): E = np.array(E); print('%s: ошибка положения за 2 с мед %.3f p90 %.3f max %.3f м (путь мед %.2f м)' % (n, np.median(E), np.percentile(E, 90), E.max(), np.median(P)))
