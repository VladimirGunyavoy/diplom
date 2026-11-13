"""c13 блок 4: стыковка прямого и обратного деревьев коридора v6 — диагностика и руки (общий алгоритм: только S=SysN, flow, g, miss; ничего про ДИ).
cand()  — копия corridor_nd.candidates с тем же результатом + пары (i,b) и промах для каждого кандидата;
arm_local() — (б): K ближайших пар (прямой узел i, обратный b) соединяются локальным SLSQP (refine на коротком отрезке: хвост прямого пути + голова обратного, окно = шар tol_loc вокруг обратного предка), принять первую, у которой после глобального refine ответ допустим;
arm_nof()   — (в): без прямого дерева, от x0 к K ближайшим обратным узлам: жадный соединитель (q_c дуг по τ, слой минимизирует расстояние до узла) + локальный SLSQP + глобальный refine."""
import numpy as np
from src.atlas6.adaptive_nd import _tree, _fpath, tree_query
from src.atlas6.corridor_nd import refine, merge

def cand(S, x0, back, tau, NF, rho, miss, kn=8, dt=None):
    dt = tau / 2 if dt is None else dt; m = int(round(tau / dt))
    Pb, Gb, parb, layb, seqs, tree = back; Pf, Gf, parf, layf = _tree(S, tau, [x0], NF, rho, 10.0, True); out = []; pi_, pb_ = [], []
    for i in range(len(Pf)):
        if S.in_goal(Pf[i]): out.append(dict(hit=0, score=Gf[i], path=_fpath(Pf, parf, layf, i, tau), i=i, b=-1, miss=0.)); continue
        _, j = tree_query(S, tree, Pb, Pf[i], kn); pi_ += [i] * len(j); pb_ += list(j)
    if pi_:
        pi_ = np.array(pi_); pb_ = np.array(pb_); X = np.array([Pf[i] for i in pi_]); G0 = np.array([Gf[i] for i in pi_]); n = len(pi_)
        ln = np.array([len(seqs[b]) for b in pb_]); Sm = np.full((n, max(ln.max(), 1)), -1, int)
        for r, b in enumerate(pb_): Sm[r, :ln[r]] = seqs[b]
        dead = np.zeros(n, bool); hit = np.zeros(n, bool); hit_k = np.zeros(n, int); m0 = miss(X).astype(float); cut_k = np.zeros(n, int); k = 0
        for step in range(Sm.shape[1]):
            for sub in range(m):
                act = ~hit & ~dead & (step < ln)
                if not act.any(): break
                for s_ in range(S.L):
                    sel = act & (Sm[:, step] == s_)
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s_, dt))
                k += 1; dead |= S.blocked(X) & act; act = act & ~dead; ing = S.in_goal(X) & act; hit_k[ing] = k; hit |= ing
                mm = np.where(act & ~ing, miss(X), np.inf); better = mm < m0; m0[better] = mm[better]; cut_k[better] = k
        for r in range(n):
            if dead[r]: continue
            steps = [(Sm[r, q], dt) for q in range(ln[r]) for _ in range(m)]; fp = _fpath(Pf, parf, layf, pi_[r], tau)
            if hit[r]: kk = hit_k[r]; out.append(dict(hit=0, score=G0[r] + kk * dt, path=fp + steps[:kk], i=int(pi_[r]), b=int(pb_[r]), miss=0.))
            else: kk = cut_k[r]; out.append(dict(hit=1, score=G0[r] + kk * dt + m0[r], path=fp + steps[:kk], i=int(pi_[r]), b=int(pb_[r]), miss=float(m0[r])))
    out.sort(key=lambda c: (c['hit'], c['score'])); return out, sum(1 for c in out if c['hit'] == 0), (Pf, Gf, parf, layf)

def fdist(S, back, Pf):
    """расстояние (масштабированное) от каждого узла прямого дерева до ближайшего обратного узла и индекс обратного"""
    tree = back[5]; d, j = tree.query(np.asarray(Pf) / S.scale, k=1); return d, np.asarray(j) % len(back[0])

def _gloc(S, target, tol):
    sc = S.scale; return lambda x: tol ** 2 - ((np.asarray(x) - target) / sc) ** 2

def _try(flow, x0, path, g, tries=1, wlim=None):
    T, sq, d, ok = refine(flow, x0, path, g, tries=tries, wlim=wlim); return T, sq, d, ok

def arm_local(S, flow, x0, back, tau, NF, rho, g, miss, K=20, qf=2, qb=2, tol=None, wlim=None, glob_tries=2):
    """(б) → (T, seq, dts) или None; доп. поля статистики возвращаются во втором значении"""
    tol = .3 * rho if tol is None else tol
    C, nh, (Pf, Gf, parf, layf) = cand(S, x0, back, tau, NF, rho, miss, 8); Pb, Gb, parb, layb, seqs, tree = back
    d, jb = fdist(S, back, Pf); order = np.argsort(d); info = dict(pairs=0, local_ok=0, hits_before=nh, dmin=float(d.min()) if len(d) else None)
    for i in order[:K]:
        i = int(i); b = int(jb[i]); info['pairs'] += 1
        j = i; tail = []
        for _ in range(qf):
            if parf[j] < 0: break
            tail.append((layf[j], tau)); j = parf[j]
        tail = tail[::-1]; a = b; nb = 0
        for _ in range(qb):
            if parb[a] < 0: break
            a = parb[a]; nb += 1
        head = [(seqs[b][q], tau) for q in range(nb)]; conn = tail + head
        if not conn: continue
        prefix = _fpath(Pf, parf, layf, j, tau); rest = [(s, tau) for s in seqs[b][nb:]]
        T, sq, dd, ok = _try(flow, Pf[j], conn, _gloc(S, Pb[a], tol), tries=2)
        if not ok: continue
        info['local_ok'] += 1; full = prefix + list(zip(sq, dd)) + rest
        T2, sq2, d2, ok2 = refine(flow, x0, full, g, tries=glob_tries, wlim=wlim)
        if ok2: info['pair_i'] = i; return (T2, sq2, d2), info
    return None, info

def arm_nof(S, flow, x0, back, tau, g, K=20, qc=8, tol=None, rho=.15, wlim=None, glob_tries=2):
    """(в): без прямого дерева"""
    tol = .3 * rho if tol is None else tol
    Pb, Gb, parb, layb, seqs, tree = back; kk = min(K, len(Pb)); d, j = tree.query(np.asarray(x0)[None] / S.scale, k=kk); j = np.atleast_1d(j[0]) % len(Pb); info = dict(pairs=0, local_ok=0, dmin=float(np.atleast_1d(d[0])[0]))
    for b in j:
        b = int(b); info['pairs'] += 1; x = np.array(x0, float); path = []
        for _ in range(qc):
            sc = [np.linalg.norm((flow(x, s, tau) - Pb[b]) / S.scale) for s in range(S.L)]; s = int(np.argmin(sc)); path.append((s, tau)); x = S.wrap(flow(x, s, tau))
        T, sq, dd, ok = _try(flow, x0, path, _gloc(S, Pb[b], tol), tries=2)
        if not ok: continue
        info['local_ok'] += 1; full = list(zip(sq, dd)) + [(s, tau) for s in seqs[b]]
        T2, sq2, d2, ok2 = refine(flow, x0, full, g, tries=glob_tries, wlim=wlim)
        if ok2: return (T2, sq2, d2), info
    return None, info
