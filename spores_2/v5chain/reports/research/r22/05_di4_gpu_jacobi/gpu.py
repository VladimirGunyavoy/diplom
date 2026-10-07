"""research-22, эксп. 05: тот же Якоби solve (все рёбра, PESS-интерполяция) на GPU aida (torch, float64). Рёбра и V_ref — эксп. 03 (di4 L400, SBCAUS 1 SBLAY 0).
REP — во сколько раз размножить задачу (оценка для полных слоёв п.38: память и время)."""
import os, time, pickle, numpy as np, torch
from tqdm import tqdm
E = os.environ['SAVEE']; dev = 'cuda'; t00 = time.time(); I = np.load(E + '_I.npy'); IDX = np.load(E + '_IDX.npy'); W = np.load(E + '_W.npy'); Vg = np.load(E + '_Vg.npy'); Vref = pickle.load(open(os.environ['VREFP'], 'rb'))
BIG = 1e3; DTN = .1; PESS = 1.; WTHR = .5; N = len(Vref); REP = int(os.environ.get('REP', 1)); DT = torch.float64 if int(os.environ.get('F64', 1)) else torch.float32
if REP > 1: I = np.concatenate([I + k * N for k in range(REP)]); IDX = np.concatenate([IDX + k * N for k in range(REP)]); W = np.tile(W, (REP, 1)); Vg = np.tile(Vg, REP); Vref = np.tile(Vref, REP); N *= REP
t0 = time.time(); gI = torch.as_tensor(I.astype(np.int64), device=dev); gX = torch.as_tensor(IDX.astype(np.int64), device=dev); gW = torch.as_tensor(W, device=dev).to(DT); goal = torch.as_tensor(Vref == 0., device=dev)
V = torch.full((N,), BIG, dtype=DT, device=dev); V[goal] = 0.; V = torch.minimum(V, torch.as_tensor(Vg, device=dev).to(DT)); torch.cuda.synchronize(); tl = time.time() - t0; ch = 1 << 22; n = 0
bar = tqdm(desc='GPU Якоби REP %d' % REP, mininterval=10); t0 = time.time()
while True:
    new = V.clone()
    for a in range(0, len(gI), ch):
        vi = V[gX[a:a + ch]]; ok = vi < BIG / 2; w = gW[a:a + ch] * ok; sm = w.sum(1); mx = torch.where(ok, vi, torch.full_like(vi, -float('inf'))).max(1).values
        v = (w * torch.where(ok, vi, torch.zeros_like(vi))).sum(1) + torch.nan_to_num((1 - sm) * (mx + PESS), nan=0., posinf=0., neginf=0.); v[sm < WTHR] = BIG
        new.scatter_reduce_(0, gI[a:a + ch], DTN + v, 'amin')
    new[goal] = 0.; d = float((new - V).abs().max()); V = new; n += 1; bar.update(1)
    if d < 1e-9 or n > 2000: break
torch.cuda.synchronize(); dt = time.time() - t0; Vc = V.cpu().numpy(); f = (Vref < BIG / 2) & (Vc < BIG / 2)
print('RESULT GPU REP %d %s | узлов %d рёбер %d | проходов %d | solve %.1f с (%.1f мс/проход) | перенос на GPU %.1f с, чтение с диска %.1f с | память GPU пик %.1f ГБ | max|V − Vref| %.3g mean %.3g | конечных %d / %d' % (REP, str(DT)[6:], N, len(I), n, dt, 1e3 * dt / n, tl, t0 - t00 - tl, torch.cuda.max_memory_allocated() / 2 ** 30, np.abs(Vc[f] - Vref[f]).max(), np.abs(Vc[f] - Vref[f]).mean(), (Vc < BIG / 2).sum(), (Vref < BIG / 2).sum()), flush=True)
