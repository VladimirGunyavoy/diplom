"""research-37 (условия senior-2 (а)(б)): агент на модели с ЧЕСТНОЙ памятью — граф выброшен.
Область с ≥ MINR узлами хранит: центр μ (d), ковариацию C (d(d+1)/2) масштабированных координат, коэффициенты квадратичной W вокруг μ (NFEAT).
Поиск области точки q: KNC ближайших центров → min расстояния Махаланобиса d_M² = δᵀC⁻¹δ; если min d_M² > DM2 — фолбэк IDW по KU
ближайшим НЕпокрытым узлам (они хранятся: d+1 чисел). Хранимые числа = области × (d + d(d+1)/2 + NFEAT) + непокрытые × (d+1).
Парно: тот же граф с LP-интерполяцией. Запуск: как model_agent.py (+ DM2=9 KNC=8 KU=4)."""
import os, sys, time, json
sys.path.insert(0, os.path.expanduser('~/spore_v5/r37'))
import numpy as np
import tvg_vt as T
from tvg_thin import mkag, env
from tvg_dl import run_agent
from graph_cells import feats, NFEAT
d = T.d; MINR = int(os.environ.get('MINR', NFEAT + 2)); DM2 = float(os.environ.get('DM2', 9)); KNC = int(os.environ.get('KNC', 8)); KU = int(os.environ.get('KU', 4))

class ModelV:
    def __init__(self, S, P, Wg, lab):
        self.S = S; regs = np.unique(lab[lab >= 0]); mu, Ci, Cf, cov = [], [], [], []; unc = np.zeros(len(P), bool); unc[lab < 0] = True
        for r in regs:
            R = np.flatnonzero(lab == r)
            if len(R) < MINR: unc[R] = True; continue
            x0 = P[R[0]]; D0 = S.wrap(P[R] - x0) / S.scale; m = D0.mean(0); D = D0 - m; C = D.T @ D / len(R) + 1e-8 * np.eye(d)
            mu.append(S.wrap(x0 + m * S.scale)); Ci.append(np.linalg.inv(C)); Cf.append(np.linalg.lstsq(feats(D, True), Wg[R], rcond=None)[0])
        self.mu, self.Ci, self.Cf = np.array(mu), np.array(Ci), np.array(Cf); self.nreg = len(mu)
        self.ctree = T.cKDTree(T._scaled(S, T._images(S, self.mu))); self.PU, self.WU = P[unc], Wg[unc]; self.nU = int(unc.sum())
        self.utree = T.cKDTree(T._scaled(S, T._images(S, self.PU))) if self.nU else None
        self.stored = self.nreg * (d + d * (d + 1) // 2 + NFEAT) + self.nU * (d + 1); self.nmiss = 0; self.nq = 0
        print('  областей-моделей %d, непокрытых узлов %d; хранимых чисел %d' % (self.nreg, self.nU, self.stored), flush=True)
    def __call__(self, Q):
        Q = np.asarray(Q, float); S = self.S; _, jc = self.ctree.query(S.wrap(Q) / S.scale, k=min(KNC, self.nreg)); jc = jc % self.nreg
        dl = S.wrap(Q[:, None, :] - self.mu[jc]) / S.scale; dm = np.einsum('nki,nkij,nkj->nk', dl, self.Ci[jc], dl); b = np.argmin(dm, 1)
        r = jc[np.arange(len(Q)), b]; D = dl[np.arange(len(Q)), b]; out = np.einsum('nf,nf->n', feats(D, True), self.Cf[r])
        miss = dm[np.arange(len(Q)), b] > DM2; self.nq += len(Q); self.nmiss += int(miss.sum())
        if miss.any() and self.nU:
            dd, ju = self.utree.query(S.wrap(Q[miss]) / S.scale, k=min(KU, self.nU)); ju = ju % self.nU; w = 1. / (dd + 1e-6); out[miss] = (w * self.WU[ju]).sum(1) / w.sum(1)
        return out

def main():
    ts = T.TS; ns = len(env.X0); S = T.mkS(ts); z = np.load(os.path.expanduser(os.environ['SRC'])); P = z['P']; lab = np.load(os.path.expanduser(os.environ['LAB']))
    base = mkag(S, P); Wg = np.minimum(base.W, float(os.environ.get('VCAP', 40)))
    Tb, sb = run_agent(base, ns); print('ГРАФ (LP): %s' % sb, flush=True)
    mv = ModelV(S, P, Wg, lab); orig = base.vh; base.vh = mv; Tm, sm = run_agent(base, ns); base.vh = orig
    full = len(P) * (d + 1); atl = 34800 * (d + 1)
    print('МОДЕЛЬ: %s | хранимых %d (граф %d, ×%.1f; атлас 34.8k узлов %d) | вне эллипсоидов %.3f запросов' % (sm, mv.stored, full, full / mv.stored, atl, mv.nmiss / max(1, mv.nq)), flush=True)
    json.dump(dict(graph=dict(T=Tb, **sb), model=dict(T=Tm, **sm), nreg=mv.nreg, nU=mv.nU, stored=mv.stored, full=full, miss=mv.nmiss / max(1, mv.nq)),
              open(os.path.expanduser('~/spore_v5/r37/magent2_%s_%s_%d_%d.json' % (T.ENV, os.path.basename(os.environ['LAB']).replace('.npy', ''), MINR, ts)), 'w'), default=float)

if __name__ == '__main__': main()
