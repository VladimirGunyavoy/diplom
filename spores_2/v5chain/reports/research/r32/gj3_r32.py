# research-32: этап 3 — удешевить проход GPU-Якоби (этап 2: 25 мс/проход): индексы int32, W f16, torch.compile слияние. 20 проходов от BIG (достаточно, этап 2).
import sys, time, numpy as np, torch
from tqdm import tqdm
dev = 'cuda'; z = np.load('/home/random/spore_v5/r32/gj_shell.npz')
BIG, DTN, WTHR, pe = float(z['BIG']), float(z['DTN']), float(z['WTHR']), .3
T = lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(dev)
Xl = T(z['Xl']); W32 = T(z['W']); own = T(z['own']).long(); s_out, sm_out, mx_out = T(z['s_out']), T(z['sm_out']), T(z['mx_out'])
nreg = len(z['V0r']); E = Xl.shape[0]; ia = int(z['ia_l'])
Vgn = torch.full((nreg,), BIG, device=dev); Vgn[T(z['nd']).long()] = T(z['Vg_n']); goal = T(z['goal_l']); forb = T(z['forb0']); Vls = z['Vls0']
# ячейка-«пустышка» nreg: V = BIG — вместо маски valid (Xl<0 -> nreg)
Xi32 = torch.where(Xl >= 0, Xl, nreg).int(); Xi64 = Xi32.long(); del Xl
def step(V, Xi, W):
    G = V[Xi]; ok = G < BIG / 2
    Wk = torch.where(ok, W, 0).float()
    s = s_out + (Wk * G).sum(1); sm = sm_out + Wk.sum(1)
    mx = torch.maximum(mx_out, torch.where(ok, G, -1e30).amax(1))
    val = torch.where(sm >= WTHR, DTN + s + (1 - sm) * (mx + pe), BIG)
    Vn = Vgn.clone().scatter_reduce_(0, own, val, 'amin'); Vn = torch.where(forb, BIG, Vn); Vn = torch.where(goal, 0., Vn)
    return torch.cat([Vn, Vn.new_full((1,), BIG)])
def run(fn, Xi, W, n=20):
    V = torch.full((nreg + 1,), BIG, device=dev); V[:nreg][goal] = 0
    for _ in range(n): V = fn(V, Xi, W)
    return V
def bench(name, fn, Xi, W):
    run(fn, Xi, W, 3); torch.cuda.synchronize(); ts = []
    for _ in range(3):
        t0 = time.time(); V = run(fn, Xi, W); torch.cuda.synchronize(); ts.append(time.time() - t0)
    V = V[:nreg].cpu().numpy(); b = (V < BIG / 2) & (Vls < BIG / 2); d = V[b] - Vls[b]
    print('%-28s 20 проходов %.3f с (%.1f мс/проход) | GPU−LS мед %+.4f p90 %+.4f | V(аг) %.3f (LS %.3f)' % (name, min(ts), 50 * min(ts), np.median(d), np.percentile(d, 90), V[ia], Vls[ia]), flush=True)
W16 = W32.half()
cases = [('int64, W f32', step, Xi64, W32), ('int32, W f32', step, Xi32, W32), ('int32, W f16', step, Xi32, W16)]
try:
    cstep = torch.compile(step, dynamic=False)
    cases += [('compile int32 W f32', cstep, Xi32, W32), ('compile int32 W f16', cstep, Xi32, W16)]
except Exception as ex: print('compile недоступен:', ex)
for c in tqdm(cases, desc='варианты', mininterval=10):
    try: bench(*c)
    except Exception as ex: print(c[0], 'ОШИБКА', repr(ex)[:300], flush=True)
print('пик памяти GPU %.2f ГБ' % (torch.cuda.max_memory_allocated() / 2**30))
