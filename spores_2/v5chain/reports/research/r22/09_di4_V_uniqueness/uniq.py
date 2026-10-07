"""research-22, эксп. 09: единственна ли V при PESS-интерполяции и защёлке V ← min(V, TV)? (эксп. 04 и b5 на manip: «V не до бита».)
GPU, рёбра эксп. 03 (di4 L400). Варианты порядка обновлений + невязка Беллмана TV − V на сошедшемся V (TV > V ⇒ узел «защёлкнут» ниже своего значения)."""
import os, time, pickle, numpy as np, torch
E = os.environ['SAVEE']; dev = 'cuda'; I = np.load(E + '_I.npy'); IDX = np.load(E + '_IDX.npy'); W = np.load(E + '_W.npy'); Vg = np.load(E + '_Vg.npy'); Vref = pickle.load(open(os.environ['VREFP'], 'rb')); LB = np.load(os.environ['LBP'])
BIG = 1e3; DTN = .1; WTHR = .5; N = len(Vref); DT = torch.float64; gI = torch.as_tensor(I.astype(np.int64), device=dev); gX = torch.as_tensor(IDX.astype(np.int64), device=dev); gW = torch.as_tensor(W, device=dev).to(DT); goal = torch.as_tensor(Vref == 0., device=dev); gVg = torch.as_tensor(Vg, device=dev).to(DT)
def TV(V, a, b, PESS):
    vi = V[gX[a:b]]; ok = vi < BIG / 2; w = gW[a:b] * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
    if PESS == 'renorm': v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) / sm.clamp_min(1e-12)
    else: v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + PESS), nan=0., posinf=0., neginf=0.)
    v[sm < WTHR] = BIG; return DTN + v
def bell(V, PESS, ch=1 << 22):
    t = torch.full((N,), BIG, dtype=DT, device=dev)
    for a in range(0, len(gI), ch): t.scatter_reduce_(0, gI[a:a + ch], TV(V, a, a + ch, PESS), 'amin')
    t[goal] = 0.; return torch.minimum(t, gVg)
def solve(mode, PESS=1., latch=True, ch=1 << 18, seed=0):
    V = torch.full((N,), BIG, dtype=DT, device=dev); V[goal] = 0.; V = torch.minimum(V, gVg); n = 0; g = torch.Generator().manual_seed(seed); t0 = time.time()
    while n < 3000:
        old = V.clone()
        if mode == 'jacobi': t = bell(V, PESS); V = torch.minimum(V, t) if latch else t
        else:
            st = list(range(0, len(gI), ch)); st = st[::-1] if mode == 'gs_rev' else [st[i] for i in torch.randperm(len(st), generator=g)] if mode == 'gs_rand' else st
            for a in st:
                t = torch.full((N,), BIG, dtype=DT, device=dev); t.scatter_reduce_(0, gI[a:a + ch], TV(V, a, a + ch, PESS), 'amin'); V = torch.minimum(V, t) if latch else torch.where(t < BIG, torch.minimum(t, gVg), V); V[goal] = 0.
        n += 1; d = float((V - old).abs().max())
        if d < 1e-9: break
    torch.cuda.synchronize(); return V, n, time.time() - t0
far = torch.as_tensor(LB > .5, device=dev); gLB = torch.as_tensor(LB, device=dev); base = None
for name, kw in (('Якоби PESS 1 (как growN)', dict(mode='jacobi')), ('GS чанками вперёд', dict(mode='gs')), ('GS чанками назад', dict(mode='gs_rev')), ('GS случайный порядок', dict(mode='gs_rand')), ('Якоби без защёлки', dict(mode='jacobi', latch=False)),
                 ('Якоби PESS 0', dict(mode='jacobi', PESS=0.)), ('Якоби перенормировка весов', dict(mode='jacobi', PESS='renorm')), ('GS назад, перенормировка', dict(mode='gs_rev', PESS='renorm'))):
    V, n, dt = solve(**kw); P = kw.get('PESS', 1.); fin = V < BIG / 2; r = bell(V, P) - V; f = fin & far; q = (V[f] / gLB[f])
    if base is None: base = V
    fb = fin & (base < BIG / 2); dv = (V - base)[fb]
    print('%-28s проходов %4d, %.1f с | конечных %d | V/LB мед. %.4f, V < .98 LB: %.4f | невязка TV − V: max %.3g, узлов с TV > V + 1e-6: %.4f, с TV < V − 1e-6: %.4f | к Якоби: max|ΔV| %.3g mean %.3g' % (name, n, dt, int(fin.sum()), float(q.median()), float((q < .98).double().mean()), float(r[fin].abs().max()), float((r[fin] > 1e-6).double().mean()), float((r[fin] < -1e-6).double().mean()), float(dv.abs().max()), float(dv.abs().mean())), flush=True)
