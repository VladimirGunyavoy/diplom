"""research-12: смена класса доведённого пути — удаление дуг (локальный поиск топологий, как «соседи» corridor_nd v6).
Доведённый путь (ocp_arcs.py OUT=*.npy) → сжать в bang-bang дуги (соседние куски с тем же знаком u сливаются) → для каждого i: удалить дугу i
или пару (i, i+1) → тёплый старт IPOPT (куски ≤ HMAX, состояния — из прежнего пути в моменты начала дуг) → лучший допустимый; повторять, пока улучшает.
Запуск (aida): PYTHONPATH=~/spore_v5/r5/pylib G=2 python3 dp_arc_drop.py ref_X.npy [ROUNDS]"""
import os, sys, json, time
import numpy as np
import ocp_arcs as O

M = int(os.environ.get('M', 40)); INS = int(os.environ.get('INS', 0)); O.HMAX = min(O.HMAX, M * float(os.environ.get('DTS', .0125))); HM = O.HMAX


def unpack(r):
    K = (len(r) - 1) // 3; return r[1:1 + 2 * K].reshape(K, 2), r[1 + 2 * K:]


def compress(U, H):
    """Слить соседние куски с одинаковым знаком u (|u| < .5 — свой «знак» 0)."""
    sg = np.where(np.abs(U) < .5, 0, np.sign(U)); out = []
    for s, u, h in zip(sg, U, H):
        if out and (out[-1][0] == s).all(): out[-1][2] += h; out[-1][1] = out[-1][1]
        else: out.append([s, u.copy(), h])
    return np.array([o[1] for o in out]), np.array([o[2] for o in out])


def starts(U, H):
    z = O.X0.copy(); Y = []
    for u, h in zip(U, H):
        Y.append(z.copy()); n = max(1, int(np.ceil(h / .005)))
        for _ in range(n): z = O.rk4(O.fnp, z, u, h / n)
    return np.array(Y)


def split(U, H, Y):
    Us, Hs, Ys = [], [], []
    for u, h, y in zip(U, H, Y):
        n = int(np.ceil(h / HM - 1e-9)); z = y.copy()
        for _ in range(n):
            Us.append(u); Hs.append(h / n); Ys.append(z.copy())
            for _ in range(20): z = O.rk4(O.fnp, z, u, h / n / 20)
    return np.array(Us), np.maximum(np.array(Hs), O.HMIN), np.array(Ys)


CACHE = {}
def solve(U, H, Y):
    Us, Hs, Ys = split(U, H, Y); K = len(Hs)
    if K not in CACHE: CACHE[K] = O.make(K, M)
    S, lbx, ubx, lbg, ubg, nx = CACHE[K]
    Xg = O.guess(Us, Hs, M, Ys); end = Xg[-1, :2]; tgt = end + ((O.C3 - end + np.pi) % (2 * np.pi) - np.pi); Xg[:, 2:] = np.clip(Xg[:, 2:], -O.WM + .02, O.WM - .02)
    r = S(x0=np.concatenate([Xg.ravel(), Us.ravel(), Hs]), p=tgt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg); st = S.stats()['return_status']
    w = np.array(r['x']).ravel(); U2 = w[nx:nx + 2 * K].reshape(K, 2); H2 = w[nx + 2 * K:]; ze, wm = O.simulate(U2, H2)
    dq = (ze[:2] - O.C3 + np.pi) % (2 * np.pi) - np.pi
    ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level') and np.all(np.abs(dq) <= O.RQ + 1e-3) and np.all(np.abs(ze[2:]) <= O.RW + 1e-3) and wm <= O.WM + 1e-3
    return (H2.sum() if ok else np.inf), U2, H2


def main():
    r = np.load(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 4; U, H = unpack(r); best = H.sum(); t0 = time.time()
    print(json.dumps(dict(start=sys.argv[1], T=round(best, 4))), flush=True)
    for rd in range(R):
        Uc, Hc = compress(U, H); Yc = starts(Uc, Hc); cand = []
        for i in range(len(Hc)):
            for drop in ((i,), (i, i + 1)):
                if max(drop) >= len(Hc): continue
                keep = [j for j in range(len(Hc)) if j not in drop]
                T, U2, H2 = solve(Uc[keep], Hc[keep], Yc[keep]); cand.append((T, drop, U2, H2))
            if INS:                                                                          # игла: короткая дуга другого угла u в середине дуги i
                for uc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    if np.allclose(np.sign(Uc[i]), uc): continue
                    h1 = Hc[i] / 2; z = Yc[i].copy(); n = max(1, int(np.ceil(h1 / .005)))
                    for _ in range(n): z = O.rk4(O.fnp, z, Uc[i], h1 / n)
                    Un = np.r_[Uc[:i], [Uc[i], uc, Uc[i]], Uc[i + 1:]]; Hn = np.r_[Hc[:i], [h1, .05, h1], Hc[i + 1:]]
                    zn = z.copy()
                    for _ in range(10): zn = O.rk4(O.fnp, zn, np.array(uc, float), .005)
                    Yn = np.r_[Yc[:i], [Yc[i], z, zn], Yc[i + 1:]]
                    T, U2, H2 = solve(Un, Hn, Yn); cand.append((T, ('ins', i) + tuple(uc), U2, H2))
        cand.sort(key=lambda c: c[0]); Tb = cand[0][0]
        print(json.dumps(dict(round=rd, arcs=len(Hc), best_drop=[str(x) for x in cand[0][1]], T=round(float(Tb), 4), top=[round(float(c[0]), 3) for c in cand[:5]],
                              nok=int(np.isfinite([c[0] for c in cand]).sum()), ncand=len(cand), sec=round(time.time() - t0))), flush=True)
        if not Tb < best - 1e-3: break
        best, U, H = Tb, cand[0][2], cand[0][3]
        np.save(sys.argv[1].replace('.npy', '_drop.npy'), np.concatenate([[best], U.ravel(), H]))
    print('RESULT ' + json.dumps(dict(start=sys.argv[1], T=round(float(best), 4), ratio=round(float(best) / float(os.environ.get('TREF', 7.636)), 4))), flush=True)


if __name__ == '__main__':
    main()
