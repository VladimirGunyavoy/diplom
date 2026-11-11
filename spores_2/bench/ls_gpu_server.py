"""b18: LSGPU — сервер пересчёта V_Б Якоби на GPU (torch.compile). Запуск в sb3-venv (torch есть, numba нет):
LSDTN=.. LSWTHR=.. python ls_gpu_server.py <zoned2.npz> <unix-сокет> [проходов=20]. Стенсилы ВСЕГО атласа (зона = атлас, ZFULL) — на карте один раз; оболочка — булева маска inreg
(вне её V = V0 закреплены), запрет forb -> BIG, цель -> 0; старт всегда от BIG (не от прошлой V_Б). step — как gj3_r32.py (W f16, индексы int32). Клиент (envs_ls.refresh) шлёт (inreg, forb), получает V float32."""
import sys, os, time, threading, numpy as np, torch
from multiprocessing.connection import Listener
dev = 'cuda'; zf, addr = sys.argv[1], sys.argv[2]; NPASS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BIG = float(os.environ.get('LSBIG', 1e9)); DTN = float(os.environ['LSDTN']); WTHR = float(os.environ['LSWTHR']); pe = .3   # G.BIG, G.DTN, G.WTHR — из клиентского скрипта
d = np.load(zf); T = lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(dev)
nz = int(d['nz']); t0 = time.time()
Xi = T(d['X']).clamp_(min=0); W = T(d['W']).half(); nd = T(d['nd']).long()
own = torch.repeat_interleave(nd, T(np.diff(d['st'])).long())
Vgn = torch.full((nz,), BIG, device=dev); Vgn[nd] = T(d['Vg_n']).float(); goal = T(d['goal']); V0 = T(d['V0']).float(); Wpos = W > 1e-7
torch.cuda.synchronize(); print('на GPU: рёбер %d, K %d, перенос %.1f с, память %.2f ГБ' % (Xi.shape[0], Xi.shape[1], time.time() - t0, torch.cuda.memory_allocated() / 2**30), flush=True)
def step(V, inreg, forb):
    G = V[Xi]; ok = (G < BIG / 2) & Wpos
    Wk = torch.where(ok, W, 0).float(); s = Wk.mul(G).sum(1); sm = Wk.sum(1)
    mx = torch.where(ok, G, -1e30).amax(1)
    val = torch.where(sm >= WTHR, DTN + s + (1 - sm) * (mx + pe), BIG)
    Vn = Vgn.clone().scatter_reduce_(0, own, val, 'amin'); Vn = torch.where(forb, BIG, Vn); Vn = torch.where(goal, 0., Vn)
    return torch.where(inreg, Vn, V0)
cstep = torch.compile(step, dynamic=False)
def solve(inreg, forb):
    V = torch.where(inreg, BIG, V0); V = torch.where(inreg & goal, 0., V)
    for _ in range(NPASS): V = cstep(V, inreg, forb)
    return V
lock = threading.Lock()
t0 = time.time(); ir = torch.zeros(nz, dtype=torch.bool, device=dev); ir[:1000] = True; solve(ir, torch.zeros_like(ir)); solve(ir, torch.zeros_like(ir)); torch.cuda.synchronize()
print('компиляция+прогрев %.1f с' % (time.time() - t0), flush=True)
def serve(c):
    try:
        while True:
            inreg, forb = c.recv()
            with lock:
                t0 = time.perf_counter(); a = torch.from_numpy(inreg).to(dev); b = torch.from_numpy(forb).to(dev); V = solve(a, b).cpu().numpy(); tg = time.perf_counter() - t0
            c.send((V, tg))
    except EOFError: pass
L = Listener(addr, family='AF_UNIX'); print('ГОТОВ', flush=True)
while True: threading.Thread(target=serve, args=(L.accept(),), daemon=True).start()
