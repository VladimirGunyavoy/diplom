"""research-37 (graph_to_cells шаг 2 → агент на модели; маятник, сжатие ×35–110): V̂(q) = квадратичная модель области ближайшего узла
(области — graph_cells.py, метки gc_lab_*.npy; модель переподгоняется по узлам области, центр — узел с наименьшей W). Узлы-одиночки и
вне областей — фолбэк LP-интерполяция графа (их число входит в «хранимые числа»). Парно: тот же граф с LP-интерполяцией (база).
Запуск: S100=1 RNDN=1 RB=.03 TAU=.2 TAUS=.05,.1,.2,.4,.8 KLP=12 VCAP=40 SRC=~/spore_v5/r35/refP_0.npz LAB=gc_lab_pend_112735_0.01_0.npy MINR=8 python3 model_agent.py pend 0 8 0"""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
import tvg_vt as T
from tvg_thin import mkag, env
from tvg_dl import run_agent
from graph_cells import feats, NFEAT
d = T.d; MINR = int(os.environ.get('MINR', NFEAT + 2))

class AgentModel:
    def __init__(self, base, S, P, Wg, lab):
        self.b, self.S, self.P = base, S, P; n = len(P); self.lab = lab.copy(); regs = np.unique(lab[lab >= 0])
        self.c = {}; self.x0 = {}; small = 0
        for r in regs:
            R = np.flatnonzero(lab == r)
            if len(R) < MINR: self.lab[R] = -1; small += len(R); continue
            s = R[np.argmin(Wg[R])]; D = S.wrap(P[R] - P[s]) / S.scale; self.c[r] = np.linalg.lstsq(feats(D, True), Wg[R], rcond=None)[0]; self.x0[r] = P[s]
        self.nreg = len(self.c); self.nfb = int((self.lab < 0).sum()); print('  моделей %d (областей < %d узлов → фолбэк: %d узлов); узлов фолбэка всего %d' % (self.nreg, MINR, small, self.nfb), flush=True)
    def vh(self, Q):
        Q = np.asarray(Q, float); _, j = self.b.tree.query(self.S.wrap(Q) / self.S.scale, k=1); j = j % len(self.P); r = self.lab[j]; out = np.empty(len(Q)); m = r >= 0
        for rr in np.unique(r[m]):
            ii = np.flatnonzero(r == rr); D = self.S.wrap(Q[ii] - self.x0[rr]) / self.S.scale; out[ii] = feats(D, True) @ self.c[rr]
        if (~m).any(): out[~m] = self.fb(Q[~m])
        return out

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts); z = np.load(os.path.expanduser(os.environ['SRC'])); P = z['P']; lab = np.load(os.path.expanduser(os.environ['LAB']))
    base = mkag(S, P); Wg = np.minimum(base.W, float(os.environ.get('VCAP', 40)))
    Tb, sb = run_agent(base, ns); print('ГРАФ (LP): %s' % sb, flush=True)
    orig = base.vh; am = AgentModel(base, S, P, Wg, lab); am.fb = orig; base.vh = am.vh
    Tm, sm = run_agent(base, ns); base.vh = orig
    stored = am.nreg * (d + NFEAT) + am.nfb * (d + 1); full = len(P) * (d + 1)
    print('МОДЕЛЬ: %s | хранимых чисел %d против %d (×%.1f)' % (sm, stored, full, full / stored), flush=True)
    json.dump(dict(graph=dict(T=Tb, **sb), model=dict(T=Tm, **sm), nreg=am.nreg, nfb=am.nfb, stored=stored, full=full),
              open(os.path.expanduser('~/spore_v5/r37/magent_%s_%s_%d.json' % (T.ENV, os.path.basename(os.environ['LAB']).replace('.npy', ''), MINR)), 'w'), default=float)

if __name__ == '__main__': main()
