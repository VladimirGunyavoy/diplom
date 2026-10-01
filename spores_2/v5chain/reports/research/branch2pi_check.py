"""research hub-research-6: гипотеза «ветвь 2π» (PLAN 0е(1)): сравнить ветвь (число оборотов k_i каждого сустава) у пути коридора и лучшего OCP, 3 зв.+g .3 вверх.
Запуск из v6: python3 ../v5chain/reports/research/branch2pi_check.py <папка с ref6d_*.npy и stats_3up.json>"""
import sys, os, json, glob, numpy as np
sys.path.insert(0, '.')
from src.atlas6.manip3dyn import flow, f
D = sys.argv[1]; GG = 0.3; C3 = np.array([np.pi/2, 0, 0]); Rq, Rw = 0.3, 0.6
wr = lambda a: (a + np.pi) % (2*np.pi) - np.pi
rng = np.random.default_rng(0)
for _ in range(32): rng.uniform(-1, 1, 6)                      # seeds (тот же поток rng, что в stats_manip6.py)
Q = [np.array([*rng.uniform(-np.pi, np.pi, 3), *rng.uniform(-1, 1, 3)]) for _ in range(32)]   # без OBST clear0 > .05 всегда
R = json.load(open(os.path.join(D, 'stats_3up.json')))['res']
def path_corr(x0, seq, dts):
    X = [x0]; x = x0.copy()
    for s, t in zip(seq, dts):
        if t > 1e-9:
            for _ in range(max(1, int(np.ceil(t / .01)))): x = flow(x, s, t / max(1, int(np.ceil(t / .01))), dt_max=.005, g=GG); X.append(x.copy())
    return np.array(X)
def path_ocp(x0, w):
    T, U = w[0], w[1:].reshape(-1, 3); x = x0.copy(); X = [x0]; h = T / len(U) / 10
    for u in U:
        for _ in range(10):
            k1 = f(x, u, GG); k2 = f(x + h/2*k1, u, GG); k3 = f(x + h/2*k2, u, GG); k4 = f(x + h*k3, u, GG); x = x + h/6*(k1+2*k2+2*k3+k4); X.append(x.copy())
    return np.array(X), T
def branch(X, x0):
    dq = np.sum(wr(np.diff(X[:, :3], axis=0)), 0)                # развёрнутый поворот
    return np.round((dq - wr(C3 - x0[:3])) / (2*np.pi)).astype(int), dq
for qi in [int(a) for a in (sys.argv[2] if len(sys.argv) > 2 else '14,9,32,30,28').split(',')]:
    x0 = Q[qi-1]; r = R[qi-1]; Xc = path_corr(x0, r['seq'], r['dts']); kc, dqc = branch(Xc, x0)
    endc = np.max(np.abs(wr(Xc[-1, :3] - C3))), np.max(np.abs(Xc[-1, 3:]))
    print(f'q{qi}: коридор T {r["T"]:.3f} ветвь {kc} Δq {np.round(dqc, 2)} конец |Δq|∞ {endc[0]:.2f} |w|∞ {endc[1]:.2f}  max|w| {np.abs(Xc[:, 3:]).max():.1f}')
    for fn in sorted(glob.glob(os.path.join(D, f'ref6d_s32up*_q{qi}_G0.3_N40_W*.npy'))):
        Xo, T = path_ocp(x0, np.load(fn)); ko, dqo = branch(Xo, x0); e = np.max(np.abs(wr(Xo[-1, :3] - C3)))
        print(f'   {os.path.basename(fn)[6:-4]:28} T {T:.3f} ветвь {ko} Δq {np.round(dqo, 2)} конец {e:.2f} max|w| {np.abs(Xo[:, 3:]).max():.1f}')
