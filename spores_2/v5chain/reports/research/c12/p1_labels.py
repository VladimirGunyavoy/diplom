"""research-38, П-1 (bundles_scheme.md): метка узла = последовательность управлений вдоль оптимального пути графа (без длительностей).
Путь: из узла i — argmin-ребро VI-LP (как graph_ctrl.py), преемник = узел с наибольшим LP-весом в конце ребра; до цели (GL>0) или LMAX шагов.
Метка raw = схлопнутые подряд одинаковые u*; clean = то же после выброса прогонов длины 1 внутри пути (дребезг).
Числа: меток на 95% активных узлов (raw/clean), доля узлов с дребезгом, длины меток, доля незавершённых путей; проверка согласованности W.
Запуск: как graph_ctrl.py (SRC=npz с P,W; argv env NB KN seed)."""
import os, sys, json, collections
sys.path.insert(0, os.path.expanduser('~/spore_v5/c12/spores_2/w39'))
import numpy as np
import tvg_vt as T
d = T.d; VCAP = float(os.environ.get('VCAP', 40)); LMAX = int(os.environ.get('LMAX', 400))

def k95(cnt, tot):
    c = np.sort(np.array(list(cnt.values())))[::-1]; return int(np.searchsorted(np.cumsum(c) / tot, .95) + 1), int(len(c)), int((c == 1).sum())

def main():
    ts = T.TS; S = T.mkS(ts); z = np.load(os.path.expanduser(os.environ['SRC'])); P, W = z['P'], np.minimum(z['W'], VCAP); n = len(P)
    from tvg_lp import InterpLP
    tree = T.cKDTree(T._scaled(S, T._images(S, P))); ip = InterpLP(S, tree, n)
    E, OK, GL = T.edges(P, S); L2 = OK.shape[1]; J, lam, ins = ip.weights(E.reshape(-1, d)); K = J.shape[1]
    J = J.reshape(n, L2, K) % n; lam = lam.reshape(n, L2, K)
    sub = (lam * W[J]).sum(2); cand = np.where(GL > 0, GL, T.TAUV[None] + sub); cand[~OK | ((GL <= 0) & ~ins.reshape(n, L2))] = 1e9
    a = np.argmin(cand, 1); u = T.SIDX[a]; cm = cand[np.arange(n), a]
    act = (W < VCAP - 1) & ~T.in_goal(P) & (cm < 1e8)
    cons = np.abs(cm[act] - W[act]); print('согласованность W=min cand: мед |Δ| %.2e, p99 %.2e' % (np.median(cons), np.quantile(cons, .99)), flush=True)
    if os.environ.get('SIM', '1') == '1':   # честный путь: точка летит по argmin-ребру из СВОЕЙ позиции (W в конце — LP-интерполяция)
        rng = np.random.default_rng(ts); ia = np.where(act)[0]; ia = rng.choice(ia, min(int(os.environ.get('NS', 5000)), len(ia)), replace=False)
        act = np.zeros(n, bool); act[ia] = True; seq = np.full((n, LMAX), -1, np.int16); X = P[ia].copy(); alive = np.ones(len(ia), bool); uu = u[ia].copy(); aa = a[ia]
        Eall = E; HY = float(os.environ.get('HYST', 0)); Tp = np.zeros(len(ia))
        for t in range(LMAX):
            seq[ia[alive], t] = uu[alive]; idx = np.where(alive)[0]
            gh = (GL[ia[idx], aa[idx]] if t == 0 else gl[idx, aa[idx]]); Tp[idx] += np.where(gh > 0, gh, T.TAUV[aa[idx]])
            g = (GL[ia[idx], aa[idx]] if t == 0 else gl[idx, aa[idx]]) > 0
            Xn = Ecur[idx, aa[idx]] if t > 0 else Eall[ia[idx], aa[idx]]
            X[idx] = Xn; alive[idx[g]] = False; alive &= ~T.in_goal(X)
            if not alive.any(): break
            Ecur, OKc, glc = T.edges(X, S); Jc, lc, insc = ip.weights(Ecur.reshape(-1, d)); Jc = Jc % n
            sc = (lc * W[Jc]).sum(1).reshape(len(X), L2); cc = np.where(glc > 0, glc, T.TAUV[None] + sc); cc[~OKc | ((glc <= 0) & ~insc.reshape(len(X), L2))] = 1e9
            an = np.argmin(cc, 1)
            if HY > 0:   # гистерезис: прежнее базовое управление, если его лучший τ в пределах HY от минимума
                cp = np.where(T.SIDX[None, :] == uu[:, None], cc, 1e9); ap = np.argmin(cp, 1); bst = cc[np.arange(len(X)), an]
                an = np.where(cp[np.arange(len(X)), ap] <= bst + HY, ap, an)
            aa = an; uu = T.SIDX[aa]; gl = glc
            alive &= cc[np.arange(len(X)), aa] < 1e8
        unfin = np.zeros(n, bool); unfin[ia] = alive; alive = unfin
        rt = Tp[~unfin[ia]] / np.maximum(W[ia][~unfin[ia]], 1e-9); print('путь/W(старт): мед %.4f p90 %.4f max %.3f (HYST %g)' % (np.median(rt), np.quantile(rt, .9), rt.max(), HY), flush=True)
    else:
      term = (GL[np.arange(n), a] > 0) | ~act
      jj = J[np.arange(n), a]; ll = lam[np.arange(n), a]; succ = jj[np.arange(n), np.argmax(ll, 1)]; succ[term] = -1
      seq = np.full((n, LMAX), -1, np.int16); cur = np.arange(n); alive = act.copy()
      for t in range(LMAX):
        seq[alive, t] = u[cur[alive]]; nxt = succ[cur]; alive = alive & (nxt >= 0); cur = np.where(nxt >= 0, nxt, cur)
        if not alive.any(): break
    unfinished = alive & act
    raw = collections.Counter(); cln = collections.Counter(); chat = 0; Lr = []; Lc = []
    for i in np.where(act)[0]:
        s = seq[i]; s = s[s >= 0]
        ch = np.flatnonzero(np.diff(s)) + 1; st = np.r_[0, ch]; ln = np.diff(np.r_[st, len(s)]); v = s[st]
        r = tuple(v.tolist()); raw[r] += 1; Lr.append(len(r))
        keep = (ln > 1); keep[0] = keep[-1] = True
        if not keep.all(): chat += 1
        vc = v[keep]; vc = vc[np.r_[True, vc[1:] != vc[:-1]]]; c = tuple(vc.tolist()); cln[c] += 1; Lc.append(len(c))
    na = int(act.sum()); kr = k95(raw, na); kc = k95(cln, na)
    MINRUN = int(os.environ.get('MINRUN', 3)); A = np.asarray(T.ACT); m = A.shape[1] if A.ndim > 1 else 1; A = A.reshape(len(A), m)
    def runs(v):   # схлопнуть, выбросить внутренние прогоны короче MINRUN шагов, снова схлопнуть
        ch = np.flatnonzero(np.diff(v)) + 1; st = np.r_[0, ch]; ln = np.diff(np.r_[st, len(v)]); w = v[st]
        k = ln >= MINRUN; k[0] = k[-1] = True; w = w[k]; return tuple(w[np.r_[True, w[1:] != w[:-1]]].tolist())
    jnt = collections.Counter(); ax = [collections.Counter() for _ in range(m)]; prod = collections.Counter()
    np.savez_compressed(os.path.expanduser('~/spore_v5/c12/out/p1seq_%s_%d_%d%s.npz' % (T.ENV, n, ts, os.environ.get('TAG', ''))), seq=seq[act], idx=np.where(act)[0])
    for i in np.where(act)[0]:
        s_ = seq[i]; s_ = s_[s_ >= 0]; jnt[runs(s_)] += 1; Av = A[s_]
        lab = tuple(runs(np.sign(Av[:, k]).astype(int)) for k in range(m)); prod[lab] += 1
        for k in range(m): ax[k][lab[k]] += 1
    res_r = dict(minrun=MINRUN, joint_k95=k95(jnt, na)[0], joint_labels=len(jnt), prod_k95=k95(prod, na)[0], prod_labels=len(prod),
                 axis_k95=[k95(c, na)[0] for c in ax], axis_top=[[(list(k), v) for k, v in c.most_common(5)] for c in ax])
    print('ПО ОСЯМ %s' % res_r, flush=True)
    res = dict(env=T.ENV, nodes=n, active=na, controls=int(len(np.unique(u[act]))), unfinished=float(unfinished.sum() / na),
               raw_k95=kr[0], raw_labels=kr[1], raw_single=kr[2], clean_k95=kc[0], clean_labels=kc[1], clean_single=kc[2],
               frac_chatter=chat / na, len_raw_med=float(np.median(Lr)), len_raw_p90=float(np.quantile(Lr, .9)), len_clean_med=float(np.median(Lc)),
               len_clean_p90=float(np.quantile(Lc, .9)), runs=res_r, top_clean=[(list(k), v) for k, v in cln.most_common(8)])
    print('ИТОГ %s' % res, flush=True)
    json.dump(res, open(os.path.expanduser('~/spore_v5/c12/out/p1_%s_%d_%d%s.json' % (T.ENV, n, ts, os.environ.get('TAG', ''))), 'w'), default=int)

if __name__ == '__main__': main()
