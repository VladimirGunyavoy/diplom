"""research hub-research-7: маятник faces — лечение застоя фронта: неизвестное = большое КОНЕЧНОЕ (линейная интерполяция без строгости), итерация сверху вниз."""
import sys, json, time; sys.argv = ['x']
exec(open('/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/pend_faces_diag.py').read().split("d0 = float")[0])
def solve_big(F, BIGF=1e3, tol=1e-9, it=200000):
    N = len(F.P); G = F.goal_mask(F.P[:, 0], F.P[:, 1]); V = np.full(N, BIGF); V[G] = 0
    E = []
    for k in (0, 1):
        t, e, kd, out = F.hit(F.P, k); i0, i1, w = F._interp(np.where(out[:, None], 0.0, e), kd); E.append((t, i0, i1, w, out | ~np.isfinite(t)))
    for n in range(it):
        Vn = np.full(N, BIGF)
        for t, i0, i1, w, out in E: Vn = np.minimum(Vn, np.where(out, BIGF, t + (1 - w) * V[i0] + w * V[i1]))
        Vn[G] = 0; Vn = np.minimum(Vn, BIGF); d = np.max(np.abs(Vn - V)); V = Vn
        if d < tol: break
    F.V = np.where(V > BIGF / 2, np.inf, V); F.iters = n; return F.V
for wmax, d0 in ((4.0, .05), (4.0, .03), (6.0, .05)):
    S = pend_top(wmax=wmax); Q = Qphys / np.array([np.pi, wmax]); ln = lines_graded(.02, 1.0, 400, d0=d0)
    t0 = time.time(); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0); V0 = F.solve(); T0, _ = F.rollout(Q); t1 = time.time() - t0
    t0 = time.time(); V = solve_big(F); T, sw = F.rollout(Q); ok = np.isfinite(T) & np.isfinite(T0)
    print(json.dumps(dict(wmax=wmax, d0=d0, probes=len(F.P), strict=dict(finV=round(float(np.isfinite(V0).mean()), 3), reach=round(float(np.isfinite(T0).mean()), 3), sec=round(t1)),
          bigfin=dict(finV=round(float(np.isfinite(V).mean()), 3), reach=round(float(np.isfinite(T).mean()), 3), iters=F.iters, sec=round(time.time() - t0),
          T_med=round(float(np.median(T[np.isfinite(T)])), 2), T_vs_strict_on_common=round(float(np.mean(T[ok] / T0[ok])), 4) if ok.any() else None))), flush=True)
