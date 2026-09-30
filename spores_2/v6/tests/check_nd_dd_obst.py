"""dd 3D с дисковым препятствием (1.0, 0.3, r=.4): replay_value; V должно быть ≥ V без препятствия, путь не касается диска. Запуск из v6."""
import sys, time; sys.path.insert(0, '.')
exec(open('tests/check_nd_dd.py').read().split("def eref_pt")[0].split("E = [wref")[0])
OB = [(0.9, 0.9, 0.5), (-0.9, -0.5, 0.4)]; blk = lambda P: np.any([np.hypot(P[..., 0] - o[0], P[..., 1] - o[1]) < o[2] for o in OB], axis=0)
So = SysN(fl, 4, S.scale, ing, [(0.0, 0.0, 0.0)], per=S.per, ok=S.ok, blocked=blk)
rng = np.random.default_rng(3); Q = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(10)]
rho, NB, NF = 0.08, 600, 600; b0 = build_back(S, tau, NB, rho); b1 = build_back(So, tau, NB, rho)
for x in Q:
    if blk(x): print('старт в препятствии'); continue
    V0 = replay_value(S, tau, x, NB, NF, rho, rho, back=b0)[0]; V1, _, _, path = replay_value(So, tau, x, NB, NF, rho, rho, back=b1, )[:4]
    # проверка: проигрыш пути по dt=0.02 не входит в диск
    X = So.wrap(x); ok = True
    for s, d in (path or []):
        for _ in range(int(round(d / 0.02))):
            X = So.wrap(flow(X, LAYERS[s], 0.02)); ok &= not blk(X)
    print('V без преп. %.2f с преп. %.2f, путь свободен: %s' % (V0, V1, ok), flush=True)
