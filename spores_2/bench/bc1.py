"""BC1 (b14, 2026-11-11): граничная клетка v = v_max. 4D атлас car (x, y, θ, v; VGM: жёсткая граница v ≤ 2) + 3D атлас ПОВЕРХНОСТИ v = 2 (growCarS, SYS=carsurf: ẋ = 2cosθ, ẏ = 2sinθ, θ̇ = ω ∈ {0, ±1.8}, a = KV·2 = 1.25 — v̇ = 0).
Общий solve (Якоби на GPU, один граф): узлы 4D [0, N4), узлы 3D [N4, N4 + N3).
  E0  4D → 4D: как в growCar.solve (стенсилы шага DTN по всем 6 слоям), цена DTN
  E1  4D → 3D: слои a > 0, шаг пересекает v = 2 ⇒ точка пересечения (подшаги rk4 ×8) → V3 билинейно, цена = время до пересечения
  E2  3D → 3D: шаг по своему ω (поверхность), цена DTN
  E3  3D → 4D: выход слоем a = −1.5 (ω ∈ {±1.8, 0}) с v = 2, цена DTN
Агент (BCAtlas.vstar/tgoal): V*(y) = min(V4(y), V3(x, y, θ) если v ≥ 2 − 1e-9); состояние на их шаге клипуется v ∈ [−1, 2].
Запуск: python bc1.py L4.pkl L3.pkl out.pkl  (решает и сохраняет V) ; потом envs2026-агент с этим атласом — bench/envs2026.py BC=out.pkl.
"""
import os, sys, time, pickle, importlib.util, inspect
HERE = os.path.dirname(os.path.abspath(__file__)); CELLS = os.path.join(HERE, '..', 'v7', 'src', 'cells7'); sys.path.insert(0, CELLS)
import numpy as np

def load_mod(name, sysname):
    """загрузить growCar.py / growCarS.py как отдельный модуль с SYS = sysname (глобалы модуля читаются при импорте)"""
    old = os.environ.get('SYS'); os.environ['SYS'] = sysname
    spec = importlib.util.spec_from_file_location(name, os.path.join(CELLS, 'growCar.py' if sysname == 'car' else 'growCarS.py')); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    if old is None: os.environ.pop('SYS', None)
    else: os.environ['SYS'] = old
    return m

def load_atlas(G, path):
    import __main__
    for nm, o in vars(G).items():
        if inspect.isclass(o): setattr(__main__, nm, o)
    d = pickle.load(open(os.path.expanduser(path), 'rb')); A = G.Atlas.__new__(G.Atlas); A.layers = d['layers']; A.finish()
    if 'V' in d: A.V = d['V']
    return A

def sat_rk4(G4, y, u, h, n):
    """rk4 car с насыщением v ∈ [VLO, VHI] после каждого подшага (как их клип)"""
    for _ in range(n):
        k1 = G4.f(y, u); k2 = G4.f(y + h / 2 * k1, u); k3 = G4.f(y + h / 2 * k2, u); k4 = G4.f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y[..., 3] = np.clip(y[..., 3], G4.VLO, G4.VHI)
    return y

def build_edges(G4, G3, A4, A3):
    """списки групп рёбер (I, IDX, W, cost): индексы в объединённом пространстве. Возврат (groups, Vg (N4+N3))"""
    N4, N3 = A4.N, A3.N; DTN = G4.DTN; groups = []; Vg = np.full(N4 + N3, np.inf)
    # E0
    for u in G4.US:
        Vg[:N4] = np.minimum(Vg[:N4], A4.tgoal(A4.P, u)); a, b, c = A4.stencils(G4.step(A4.P, u)); groups.append((a, b.astype(np.int64), c, DTN, 'E0')); print('E0', u, len(a), flush=True)
    # E1: слои a > 0: пересечение v = 2
    for u in G4.US:
        if u[0] <= 0: continue
        y = A4.P.copy(); tc = np.full(len(y), np.inf); pc = np.zeros((len(y), 3)); n8 = 8
        for i in range(1, n8 + 1):
            y = G4.rk4(y, u, DTN / n8, 1); hit = np.isinf(tc) & (y[:, 3] >= G4.VHI - 1e-12); tc = np.where(hit, DTN * i / n8, tc); pc[hit] = y[hit][:, :3]
        m = np.flatnonzero(np.isfinite(tc))
        if not len(m): continue
        P3 = G3.wrapy(pc[m]); a, b, c = A3.stencils(P3); groups.append((m[a], b.astype(np.int64) + N4, c, tc[m][a], 'E1')); print('E1', u, len(m), len(a), flush=True)
    # E2
    for u in G3.US:
        Vg[N4:] = np.minimum(Vg[N4:], A3.tgoal(A3.P, u)); a, b, c = A3.stencils(G3.step(A3.P, u)); groups.append((a + N4, b.astype(np.int64) + N4, c, DTN, 'E2')); print('E2', u, len(a), flush=True)
    # E3: выход a = −1.5
    lift = np.c_[A3.P, np.full(N3, G4.VHI)]
    for u in G4.US:
        if u[0] >= 0: continue
        a, b, c = A4.stencils(G4.step(lift, u)); groups.append((a + N4, b.astype(np.int64), c, DTN, 'E3')); print('E3', u, len(a), flush=True)
    return groups, Vg

def solve(groups, Vg, A4, A3, G4, it=int(os.environ.get('BCIT', 1500)), PESS=1., WTHR=None):
    import torch
    dev = 'cuda'; N = A4.N + A3.N; BIG = G4.BIG; WTHR = WTHR if WTHR is not None else float(os.environ.get('WTHR', .5))
    goal = np.r_[A4.goal, A3.goal]; V0 = np.full(N, BIG); V0[goal] = 0.; V = np.minimum(V0, Vg)
    Vt = torch.as_tensor(V, device=dev).double(); Vgt = torch.as_tensor(Vg, device=dev).double(); gl = torch.as_tensor(goal, device=dev)
    hasE = torch.zeros(N, dtype=torch.bool, device=dev); G = []
    for I, IDX, W, cost, tag in groups:
        gI = torch.as_tensor(np.asarray(I, np.int64), device=dev); hasE[gI] = True
        G.append((gI, torch.as_tensor(IDX.astype(np.int32 if N < 2 ** 31 else np.int64), device=dev), torch.as_tensor(W, device=dev), torch.as_tensor(np.broadcast_to(cost, len(I)).astype(np.float64), device=dev)))
    ch = 1 << 22; n = 0
    from tqdm import tqdm
    bar = tqdm(desc='BC1 Якоби', mininterval=10, leave=False)
    while n < it:
        new = Vt.clone(); new[hasE] = Vgt[hasE]                                                   # NOLATCH, как в итоговом c1600
        for gI, gX, gW, gC in G:
            for a in range(0, len(gI), ch):
                vi = Vt[gX[a:a + ch].long()]; w_ = gW[a:a + ch].double(); ok = vi < BIG / 2; w = w_ * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
                v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + PESS), nan=0., posinf=0., neginf=0.); v[sm < WTHR] = BIG
                new.scatter_reduce_(0, gI[a:a + ch], gC[a:a + ch] + v, 'amin')
        new[gl] = 0.; d = float((new - Vt).abs().max()); Vt = new; n += 1; bar.update(1)
        if n % 50 == 0: print('BC1 it', n, 'd', d, 'V4 big', float((new[:A4.N] >= BIG / 2).float().mean()), 'V3 big', float((new[A4.N:] >= BIG / 2).float().mean()), flush=True)
        if d < float(os.environ.get('BCTOL', 1e-7)): break
    print('BC1 solve: проходов', n, flush=True); return Vt.cpu().numpy(), n

class Shim:
    """модуль growCar с потоком, насыщающим v (как у них): для envs2026.plan/finish_plan"""
    def __init__(s, G4): s._g = G4
    def __getattr__(s, n): return getattr(s._g, n)
    def f(s, y, u):
        dv = u[0] - s._g.KV * y[..., 3]; dv = np.where(((y[..., 3] >= s._g.VHI) & (dv > 0)) | ((y[..., 3] <= s._g.VLO) & (dv < 0)), 0., dv)
        return np.stack([y[..., 3] * np.cos(y[..., 2]), y[..., 3] * np.sin(y[..., 2]), u[1] + 0 * y[..., 2], dv], -1)
    def rk4(s, y, u, h, n):
        for _ in range(n):
            k1 = s.f(y, u); k2 = s.f(y + h / 2 * k1, u); k3 = s.f(y + h / 2 * k2, u); k4 = s.f(y + h * k3, u); y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); y[..., 3] = np.clip(y[..., 3], s._g.VLO, s._g.VHI)
        return y
    def step(s, y, u, sg=1.): return s.rk4(y, u, sg * s._g.DTN / 2, 2)

class BCAtlas:
    """интерфейс для envs2026.plan: tgoal(Y, u), vstar(Y) на объединённой V; Y клипуется по v"""
    def __init__(s, G4, G3, A4, A3, V):
        s.G4, s.G3, s.A4, s.A3 = G4, G3, A4, A3; s.V4 = V[:A4.N]; s.V3 = V[A4.N:]; s.N = A4.N + A3.N
    def tgoal(s, Y, u): return s.A4.tgoal(Y, u)
    def vstar(s, Y):
        G4 = s.G4; Y = np.array(Y, float); Y[..., 3] = np.clip(Y[..., 3], G4.VLO, G4.VHI)
        A4 = s.A4; saveV = A4.V; A4.V = s.V4; out = A4.vstar(Y); A4.V = saveV
        top = Y[..., 3] >= G4.VHI - 1e-9
        if top.any():
            A3 = s.A3; P3 = s.G3.wrapy(Y[top][:, :3]); saveV = A3.V; A3.V = s.V3; v3 = A3.vstar(P3); A3.V = saveV; o = out[top]; out[top] = np.minimum(o, v3)
        return out

if __name__ == '__main__':
    t0 = time.time(); G4 = load_mod('growCar', 'car'); G3 = load_mod('growCarS', 'carsurf'); A4 = load_atlas(G4, sys.argv[1]); A3 = load_atlas(G3, sys.argv[2]); print('N4', A4.N, 'N3', A3.N, flush=True)
    groups, Vg = build_edges(G4, G3, A4, A3); t_edges = time.time() - t0; t1 = time.time(); V, n = solve(groups, Vg, A4, A3, G4); t_solve = time.time() - t1
    pickle.dump(dict(V=V, N4=A4.N, N3=A3.N, t_edges=t_edges, t_solve=t_solve, iters=n), open(sys.argv[3], 'wb')); print('t_edges %.0f t_solve %.0f' % (t_edges, t_solve), flush=True)
