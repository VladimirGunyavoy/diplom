"""Живое проигрывание траекторий (клавиши P / O в окне): пути, посчитанные агентом (vsurf.agent_paths), раскрываются по времени — 1 записанная точка за DTPT с (rec·dt = .03 с) × SPEED; каждый кадр дорисовываются ТОЛЬКО новые сегменты (без перестройки всего):
хвостовой слот пула слоёв (~FPS_SLOT кадров) пересобирается из накопленных новых сегментов, заполненный слот замораживается, следом — следующий. Чистый numpy: геометрию в панду кладёт growview.live_update. Новый агент (O) подключается в любой момент (свой k0)."""
import numpy as np
from . import vsurf as VS

DTPT = .03                 # с модельного времени на одну записанную точку (rec=3, dt=.01)
SPEED = float(__import__('os').environ.get('V8_LIVE_SPEED', 2.))   # во сколько раз быстрее реального времени идёт проигрывание
SLOT_FRAMES = 20           # кадров на слот пула
NSLOT = 80                 # слотов (≈1600 кадров; последний слот растёт дальше)


class LivePlay:
    def __init__(s, ax=(0, 1)):
        s.ax = ax; s.reset()

    def reset(s):
        s.t = 0.; s.A = []; s.slot = 0; s.fr = 0; s.tail = dict(floor=[], fcol=[], surf=[], scol=[]); s.done_slots = 0

    def add(s, res):
        """подключить агентов (результат agent_paths): каждый стартует с текущего момента; res['paths'|'V'|'U']"""
        k0 = s.t / DTPT
        for P, V, U in zip(res['paths'], res['V'], res['U']):
            P = np.asarray(P, float); U = np.asarray(U, float).copy()
            if len(P) >= 2: U[0] = U[1]
            s.A.append(dict(P=P, V=np.asarray(V, float), U=U, k0=k0, r=0))

    @property
    def active(s): return any(a['r'] < len(a['P']) for a in s.A)

    def step(s, dt):
        """→ dict(new=bool, slot=int, tail={floor,fcol,surf,scol: (n,·)}, frozen=bool, heads=(floor (m,3), surf (m,3), col (m,4)))"""
        s.t += dt * SPEED; kk = s.t / DTPT; paths, Vs, Us = [], [], []; heads = []
        for a in s.A:
            n = len(a['P']); r = int(min(n, max(kk - a['k0'], 0.) + 1)) if n else 0
            if r > a['r']:
                lo = max(a['r'] - 1, 0); paths.append(a['P'][lo:r]); Vs.append(a['V'][lo:r]); Us.append(a['U'][lo:r])
                if a['r'] == 0 and len(Us[-1]) >= 2: pass
                a['r'] = r
        if paths:
            g = VS.traj_geom(dict(paths=paths, V=Vs, U=Us), s.ax)
            for k in ('floor', 'fcol', 'surf', 'scol'): s.tail[k].append(g[k])
        out = dict(new=bool(paths), slot=min(s.slot, NSLOT - 1))
        out['tail'] = {k: (np.concatenate(v) if v else None) for k, v in s.tail.items()}
        s.fr += 1; out['frozen'] = False
        if s.fr >= SLOT_FRAMES and s.slot < NSLOT - 1: s.fr = 0; s.slot += 1; s.tail = dict(floor=[], fcol=[], surf=[], scol=[]); out['frozen'] = True      # слот заполнен: заморожен, следующий — с нуля
        H = []; HC = []
        for a in s.A:
            if a['r'] > 0:
                i = a['r'] - 1; f = np.zeros(3); f[0] = a['P'][i, s.ax[0]]; f[2] = a['P'][i, s.ax[1]]; H.append(f); HC.append(VS.ctrl_color(a['U'][i:i + 1])[0])
        out['heads'] = (np.array(H) if H else np.zeros((0, 3)), np.array(HC) if HC else np.zeros((0, 4)))
        return out
