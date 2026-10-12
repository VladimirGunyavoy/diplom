"""SPD1 (b9): V на новую цель без пересборки слоёв и стенсилов (growN, manip/любая N-мерная система).
Рёбра атласа (I, IDX, W: стенсилы SONE/SBCAUS) от цели НЕ зависят (профиль 2026-10-12: solve g300so 361 с = стенсилы ~270 + Якоби ~75 + tgoal 13) —
строятся один раз (growN.Atlas.solve с SAVEE=префикс, либо edges_from_atlas), дальше смена цели = tgoal у границы цели + GPU-Якоби.
    S = GoalSolver(A, prefix)             # рёбра на GPU
    S.retarget(gc, tol=1e-9, lcut=None)   # gc — центр цели (N чисел, как growN.GC); lcut — остановка, когда V<lcut устоялась
Результат: A.V, G.GC = gc (rollout/ingoal работают с новой целью). Якоби — тот же оператор, что в growN.solve (SOLVEGPU, PESS, NOLATCH)."""
import time, numpy as np, torch
import __main__ as _m
import growN as G


class GoalSolver:
    def __init__(s, A, prefix, dev='cuda'):
        s.A, s.dev = A, dev
        I = np.load(prefix + '_I.npy'); IDX = np.load(prefix + '_IDX.npy'); W = np.load(prefix + '_W.npy')
        s.gI = torch.as_tensor(I.astype(np.int64), device=dev); s.gX = torch.as_tensor(IDX, device=dev); s.gW = torch.as_tensor(W, device=dev)
        s.hasE = torch.zeros(A.N, dtype=torch.bool, device=dev); s.hasE[s.gI] = True; s.ne = len(I)
        s.P = A.P; s.fmax = None

    def tgoal_min(s, gc):
        """Vg = min по u времени входа в цель за шаг DTN (growN.Atlas.tgoal), считается только у границы цели."""
        A = s.A; G.GC = np.asarray(gc, float); d = np.abs(G.wrapg(A.P)) - G.RHOV; dist = (np.maximum(d, 0.) ** 2).sum(-1) ** .5
        Vg = np.full(A.N, np.inf); nn = np.zeros(A.N, bool)
        for u in G.US:                                                                                          # узел может войти в цель за DTN, только если до коробки ближе 3·DTN·|f| (локальная скорость)
            fu = np.linalg.norm(G.f(A.P, u), axis=-1); idx = np.flatnonzero(dist <= 3. * G.DTN * fu); nn[idx] = True
            if len(idx): Vg[idx] = np.minimum(Vg[idx], A.tgoal(A.P[idx], u))
        return Vg, int(nn.sum())

    def retarget(s, gc, tol=1e-9, lcut=None, maxit=20000, log=print, ch=1 << 22, fp32=False):
        A, dev = s.A, s.dev; t0 = time.time(); Vg, nn = s.tgoal_min(gc); A.goal = G.ingoal(A.P); t1 = time.time()
        V = np.full(A.N, G.BIG); V[A.goal] = 0.; V = np.minimum(V, Vg)
        dt = torch.float32 if fp32 else torch.float64; Vt = torch.as_tensor(V, device=dev).to(dt); Vgt = torch.as_tensor(Vg, device=dev).to(dt); goal = torch.as_tensor(A.goal, device=dev); pe = G.PESS if G.PESS >= 0 else 0.
        n = 0; d = 1.
        while n < maxit:
            new = Vt.clone()
            if G.NOLATCH: new[s.hasE] = Vgt[s.hasE]
            for a in range(0, s.ne, ch):
                vi = Vt[s.gX[a:a + ch].long()]; w_ = s.gW[a:a + ch].to(dt)
                ok = vi < G.BIG / 2; w = w_ * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
                v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + pe), nan=0., posinf=0., neginf=0.); v[sm < G.WTHR] = G.BIG
                new.scatter_reduce_(0, s.gI[a:a + ch], G.DTN + v, 'amin')
            new[goal] = 0.; df = (new - Vt).abs()
            if lcut is not None: df = torch.where(new < lcut, df, torch.zeros_like(df))
            d = float(df.max()); Vt = new; n += 1
            if d < tol: break
        A.V = Vt.double().cpu().numpy(); A.n_it = n; torch.cuda.synchronize(); t2 = time.time()
        log('retarget: gc %s | у цели %d узлов | tgoal %.1f с | Якоби %d проходов %.1f с | всего %.1f с' % (np.round(gc, 3).tolist(), nn, t1 - t0, n, t2 - t1, t2 - t0)); return A
