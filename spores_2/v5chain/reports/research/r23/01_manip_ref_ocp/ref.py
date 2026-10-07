"""research-23: эталон T для манипулятора 2 зв. 4D (growN SYS=manip, исправленная динамика b4) — прямая оптимизация CasADi/IPOPT,
multiple shooting N интервалов RK4, свободное T, |τᵢ| ≤ 1 (коробка = выпуклая оболочка 4 слоёв-углов, динамика аффинна по τ ⇒ inf по слоям = T* коробки),
|wᵢ| ≤ WMP вдоль пути (WM поля growN = 3; WMP=50 ≈ без предела — сверка с перебором manip_bruteforce.py), цель — коробка |qᵢ − 2πkᵢ| ≤ .3, |wᵢ| ≤ .5.
Мультистарт: ветви k ∈ {−1,0,1}² × NG затравок (прямая линия + шум). Допустимость проверяется мелким RK4 numpy с кусочно-постоянным τ.
Запуск: PYTHONPATH=<casadi> WMP=3 python3 ref.py [N NG] [индексы стартов] > out.jsonl"""
import os, sys, json, time, itertools
import numpy as np
import casadi as ca
from tqdm import tqdm
AA, BB, DD = 2.5, .5, 1.
Rq, Rw = .3, .5; WMP = float(os.environ.get('WMP', 3.)); NQ = int(os.environ.get('NQ', 8))
N, NG = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (60, 3)
ONLY = [int(a) for a in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def acc(x, t, m=np):
    c, s = m.cos(x[1]), m.sin(x[1]); w1, w2 = x[2], x[3]; h = -BB * s
    r1 = t[0] - h * (2 * w1 * w2 + w2 ** 2); r2 = t[1] + h * w1 ** 2
    m11, m12, m22 = AA + 2 * BB * c, DD + BB * c, DD; det = m11 * m22 - m12 ** 2
    return (m22 * r1 - m12 * r2) / det, (-m12 * r1 + m11 * r2) / det


def f_np(x, u): a1, a2 = acc(x, u); return np.array([x[2], x[3], a1, a2])
def f_ca(x, u): a1, a2 = acc(x, u, ca); return ca.vertcat(x[2], x[3], a1, a2)


def starts():
    """Те же старты, что growN.starts_ref (SYS=manip) и manip_bruteforce.py; NQ > 8 — продолжение того же потока rng."""
    rng = np.random.default_rng(0)
    for _ in range(32): rng.uniform(-1, 1, 4)
    return [np.array([*rng.uniform(-np.pi, np.pi, 2), *rng.uniform(-1, 1, 2)]) for _ in range(NQ)]


def check(x0, U, T, k, sub=20):
    """Мелкий RK4 numpy: конец в коробке цели (ветвь k) и max|w| вдоль пути."""
    x = np.array(x0, float); h = T / N / sub; W = 0.
    for u in U:
        for _ in range(sub):
            k1 = f_np(x, u); k2 = f_np(x + h / 2 * k1, u); k3 = f_np(x + h / 2 * k2, u); k4 = f_np(x + h * k3, u); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            W = max(W, abs(x[2]), abs(x[3]))
    e = np.r_[np.abs(x[:2] - 2 * np.pi * np.asarray(k)) - Rq, np.abs(x[2:]) - Rw]
    return float(e.max()), W


def solve_one(x0, k, guess_rng, T0):
    opti = ca.Opti(); X = opti.variable(4, N + 1); U = opti.variable(2, N); T = opti.variable(); h = T / N
    opti.minimize(T); opti.subject_to(X[:, 0] == x0); opti.subject_to(opti.bounded(.05, T, 15.))
    for i in range(N):
        xi, ui = X[:, i], U[:, i]
        k1 = f_ca(xi, ui); k2 = f_ca(xi + h / 2 * k1, ui); k3 = f_ca(xi + h / 2 * k2, ui); k4 = f_ca(xi + h * k3, ui)
        opti.subject_to(X[:, i + 1] == xi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    opti.subject_to(opti.bounded(-1, ca.vec(U), 1)); opti.subject_to(opti.bounded(-WMP, ca.vec(X[2:, :]), WMP))
    qf = 2 * np.pi * np.asarray(k, float)
    for j in range(2): opti.subject_to(opti.bounded(qf[j] - Rq, X[j, N], qf[j] + Rq)); opti.subject_to(opti.bounded(-Rw, X[2 + j, N], Rw))
    s = np.linspace(0, 1, N + 1); xg = np.zeros((4, N + 1))
    for j in range(2): xg[j] = x0[j] + (qf[j] - x0[j]) * s; xg[2 + j] = np.clip((qf[j] - x0[j]) / T0 + guess_rng.normal(0, .3), -WMP, WMP)
    opti.set_initial(X, xg); opti.set_initial(U, guess_rng.uniform(-1, 1, (2, N))); opti.set_initial(T, T0)
    opti.solver('ipopt', dict(print_time=False), dict(print_level=0, max_iter=1500, tol=1e-8, acceptable_tol=1e-6, sb='yes'))
    try: sol = opti.solve(); st = 'ok'
    except RuntimeError: sol = opti.debug; st = 'fail'
    return st, float(sol.value(T)), np.array(sol.value(U)).T


if __name__ == '__main__':
    Q = starts(); idx = ONLY if ONLY is not None else range(len(Q))
    jobs = [(q, k, g) for q in idx for k in itertools.product((-1, 0, 1), repeat=2) for g in range(NG)]
    best = {q: (np.inf, None) for q in idx}; t0 = time.time()
    for q, k, g in tqdm(jobs, desc='OCP manip WMP %g' % WMP, mininterval=10, file=sys.stderr):
        x0 = Q[q]; rng = np.random.default_rng(1000 * q + 10 * g + 3 * k[0] + k[1] + 4)
        T0 = 1.5 + .7 * np.abs(np.r_[2 * np.pi * np.asarray(k) - x0[:2]]).max()
        st, T, U = solve_one(x0, k, rng, T0)
        if st != 'ok': continue
        err, W = check(x0, U, T, k)
        if err <= 2e-3 and W <= WMP + 1e-2 and T < best[q][0]: best[q] = (T, dict(k=list(k), g=g, err=err, maxw=round(W, 3)))
    for q in idx:
        T, info = best[q]; print(json.dumps(dict(q=q, x=np.round(Q[q], 4).tolist(), WMP=WMP, N=N, T=None if not np.isfinite(T) else round(T, 4), **(info or {}))), flush=True)
    print('всего %.0f с' % (time.time() - t0), file=sys.stderr)
