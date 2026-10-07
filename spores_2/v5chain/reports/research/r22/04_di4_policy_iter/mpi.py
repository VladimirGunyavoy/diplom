"""research-22, эксп. 04: solve в 4D модифицированной итерацией по политике (Howard). Причина каскада вёдер (эксп. 03): лучшее ребро-стенсил
в 92% зависит от вершины с V ≥ V узла (клетка M 3 поперёк крупнее шага DTN по V) ⇒ V — решение неявной системы, порядок не спасает.
Идея: полный проход (все 6.6M рёбер) нужен только чтобы выбрать лучшее ребро узла; между ними — K дешёвых проходов по одному ребру на узел.
Рёбра и V_ref — из эксп. 03 (SBCAUS 1, SBLAY 0, di4 L400). KIN — внутренних проходов (0 = чистый Якоби), env."""
import os, time, pickle, numpy as np
from tqdm import tqdm
E = os.environ['SAVEE']; I = np.load(E + '_I.npy'); IDX = np.load(E + '_IDX.npy'); W = np.load(E + '_W.npy').astype(np.float64); Vg = np.load(E + '_Vg.npy'); Vref = pickle.load(open(os.environ['VREFP'], 'rb'))
BIG = 1e3; DTN = .1; PESS = 1.; WTHR = .5; KIN = int(os.environ.get('KIN', 20)); TOLIN = float(os.environ.get('TOLIN', 1e-10)); N = len(Vref); goal = Vref == 0.
def interp(W, VI):
    ok = VI < BIG / 2; w = W * ok; sm = w.sum(1)
    with np.errstate(invalid='ignore'): v = (w * np.where(ok, VI, 0.)).sum(1) + (1 - sm) * (np.where(ok, VI, -np.inf).max(1) + PESS)
    v[sm < WTHR] = BIG; return v
st = np.flatnonzero(np.r_[True, I[1:] != I[:-1]]); nd = I[st]; cnt = np.diff(np.r_[st, len(I)]); V = np.full(N, BIG); V[goal] = 0.; V = np.minimum(V, Vg); t0 = time.time(); nfull = 0; nin = 0; ein = 0
bar = tqdm(desc='MPI KIN %d' % KIN, mininterval=10)
while True:
    val = np.empty(len(I))
    for a in range(0, len(I), 1 << 21): b = a + (1 << 21); val[a:b] = DTN + interp(W[a:b], V[IDX[a:b]])
    gm = np.minimum.reduceat(val, st); new = V.copy(); new[nd] = np.minimum(V[nd], gm); new[goal] = 0.; d = np.max(np.abs(new - V)); V = new; nfull += 1; bar.update(1); bar.set_postfix(d='%.2e' % d, inner=nin)
    if d < 1e-9: break
    if KIN:
        hit = np.flatnonzero((val == np.repeat(gm, cnt)) & (val < BIG / 2)); _, f = np.unique(I[hit], return_index=True); pe = hit[f]; pn = I[pe]; pI = IDX[pe]; pW = W[pe]; m = ~goal[pn]; pn, pI, pW = pn[m], pI[m], pW[m]
        for k in range(KIN):
            v = np.minimum(V[pn], DTN + interp(pW, V[pI])); di = np.max(np.abs(v - V[pn])); V[pn] = v; nin += 1; ein += len(pn)
            if di < TOLIN: break
dt = time.time() - t0; f = (Vref < BIG / 2) & (V < BIG / 2)
print('RESULT KIN %d | полных проходов %d, внутренних %d | вычислений рёбер %.3g (Якоби: %.3g) | solve %.0f с | max|V − Vref| %.3g mean %.3g | конечных %d / %d' % (KIN, nfull, nin, nfull * len(I) + ein, 175 * len(I), dt, np.abs(V[f] - Vref[f]).max(), np.abs(V[f] - Vref[f]).mean(), (V < BIG / 2).sum(), (Vref < BIG / 2).sum()), flush=True)
