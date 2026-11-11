# research-32: этап 2 — пересчёт V_Б на оболочке Якоби на GPU (torch), сравнение с LS (этап 1, gj1_r32.py). Запуск: python gj2_r32.py [f32|f16w]
import sys, time, numpy as np, torch
from tqdm import tqdm
dev = 'cuda'; z = np.load(sys.argv[1] if len(sys.argv) > 1 else '/home/random/spore_v5/r32/gj_shell.npz')
BIG, DTN, WTHR, pe = float(z['BIG']), float(z['DTN']), float(z['WTHR']), .3
t0 = time.time()
T = lambda a, dt=None: torch.as_tensor(np.ascontiguousarray(a)).to(dev) if dt is None else torch.as_tensor(np.ascontiguousarray(a)).to(dev, dt)
Xl = T(z['Xl']).long(); W = T(z['W']); own = T(z['own']).long(); s_out, sm_out, mx_out = T(z['s_out']), T(z['sm_out']), T(z['mx_out'])
nreg = len(z['V0r']); E = Xl.shape[0]
Vgn = torch.full((nreg,), BIG, device=dev); Vgn[T(z['nd']).long()] = T(z['Vg_n']); goal = T(z['goal_l'])
valid_x = Xl >= 0; Xc = Xl.clamp(min=0); del Xl
torch.cuda.synchronize(); print('на GPU: узлов %d, рёбер %d, перенос %.2f с, память %.2f ГБ' % (nreg, E, time.time() - t0, torch.cuda.memory_allocated() / 2**30), flush=True)
ia = int(z['ia_l'])
def jac(V, forb, tol=1e-4, maxit=400, log=None):
    """Якоби: V_i = min(Vg_i, min_e DTN + s_e + (1-sm_e)(mx_e+pe)) при sm_e >= WTHR; запрет -> BIG, цель -> 0."""
    V = V.clone(); V[forb] = BIG; V[goal] = 0.; hist = []
    for it in range(maxit):
        G = V[Xc]; ok = valid_x & (G < BIG / 2)
        Wk = torch.where(ok, W, 0.)
        s = s_out + (Wk * G).sum(1); sm = sm_out + Wk.sum(1)
        mx = torch.maximum(mx_out, torch.where(ok, G, -1e30).amax(1))
        val = torch.where(sm >= WTHR, DTN + s + (1 - sm) * (mx + pe), BIG)
        Vn = Vgn.clone().scatter_reduce_(0, own, val, 'amin')
        Vn[forb] = BIG; Vn[goal] = 0.
        fin = (Vn < BIG / 2) | (V < BIG / 2)
        ch = (Vn - V).abs()[fin].max().item() if (it % 2 == 1 or it < 3) else 1.
        V = Vn
        if log is not None and it % 2 == 1: hist.append((it + 1, ch))
        if ch < tol: return V, it + 1, hist
    return V, maxit, hist
def cmp(V, Vls, name, it, t):
    V = V.cpu().numpy(); a, b = V < BIG / 2, Vls < BIG / 2; both = a & b; d = V[both] - Vls[both]
    print('%-34s проходов %3d  %.3f с (%.1f мс/проход) | кон. GPU %d LS %d (только GPU %d, только LS %d) | GPU−LS мед %+.4f p10 %+.4f p90 %+.4f min %+.3f max %+.3f | V(аг) GPU %.3f LS %.3f' % (
        name, it, t, 1e3 * t / it, a.sum(), b.sum(), (a & ~b).sum(), (~a & b).sum(), np.median(d), np.percentile(d, 10), np.percentile(d, 90), d.min(), d.max(), V[ia], Vls[ia]), flush=True)
    return V
V0r = T(z['V0r']); forbs = {k: T(z['forb%d' % k]) for k in (0, 6, 12, 24)}
jac(torch.full((nreg,), BIG, device=dev), forbs[0], maxit=3)   # прогрев
Vb = {}
for init in tqdm(['BIG', 'V0'], desc='k0 холодный', mininterval=10):
    torch.cuda.synchronize(); t0 = time.time()
    V, it, hist = jac(torch.full((nreg,), BIG, device=dev) if init == 'BIG' else V0r, forbs[0], log=1); torch.cuda.synchronize(); t = time.time() - t0
    Vb[init] = cmp(V, z['Vls0'], 'k0 старт ' + init, it, t); print('   сходимость (проход, max|ΔV|):', ' '.join('%d:%.3g' % h for h in hist[::max(1, len(hist) // 12)]))
    if init == 'BIG': Vprev = V
# качество от числа проходов (обрыв раньше сходимости) — от BIG
for n in (10, 20, 30, 40, 60):
    torch.cuda.synchronize(); t0 = time.time(); V, it, _ = jac(torch.full((nreg,), BIG, device=dev), forbs[0], tol=0, maxit=n); torch.cuda.synchronize()
    cmp(V, z['Vls0'], 'k0 BIG, обрыв на %d' % n, it, time.time() - t0)
# инкрементально: старт с V_Б(t=0) для новых запретов
for k in (6, 12, 24):
    for init in ('prevV', 'BIG'):
        torch.cuda.synchronize(); t0 = time.time()
        V, it, _ = jac(Vprev if init == 'prevV' else torch.full((nreg,), BIG, device=dev), forbs[k]); torch.cuda.synchronize()
        cmp(V, z['Vls%d' % k], 'k%d старт %s' % (k, init), it, time.time() - t0)
print('пик памяти GPU %.2f ГБ' % (torch.cuda.max_memory_allocated() / 2**30))
