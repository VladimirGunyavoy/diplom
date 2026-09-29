"""research: τ своё на слой для дифдрайва (ромб-U, h = .25). Солвер worker'а (v6 dd_atlas.solve_dd) с подменой τ → τ_k по слою.
Гипотеза: поворот на месте с τ = hθ (ровно один шаг θ-сетки) — без интерполяции по θ; прямые — τ = h. Сверка — как tests/check_dd_vs_ref.py."""
import sys, inspect, importlib.util, numpy as np
sys.path.insert(0, '/home/rl/claude-work/projects/spore/spores_2/v6')
import src.atlas6.dd_atlas as dd
sp = importlib.util.spec_from_file_location('ref', '/home/rl/claude-work/projects/spore/spores_2/v5chain/reports/research/dd_rhombus_ref.py')
ref = importlib.util.module_from_spec(sp); sp.loader.exec_module(ref)
src = inspect.getsource(dd.solve_dd).replace('def solve_dd(', 'def solve_tk(taus, ')
for a, b in [("ends = [flow(P, vw, tau) for vw in layers]", "ends = [flow(P, vw, tk) for vw, tk in zip(layers, taus)]"),
             ("bad = [hit(P) | hit(flow(P, vw, tau / 2)) | hit(flow(P, vw, tau)) for vw in layers]",
              "bad = [hit(P) | hit(flow(P, vw, tk / 2)) | hit(flow(P, vw, tk)) for vw, tk in zip(layers, taus)]"),
             ("for E, b in zip(ends, bad):\n            Vn = np.minimum(Vn, np.where(b, K, tau + interp(V, E)))",
              "for E, b, tk in zip(ends, bad, taus):\n            Vn = np.minimum(Vn, np.where(b, K, tk + interp(V, E)))")]:
    assert a in src, a; src = src.replace(a, b)
ns = dict(vars(dd)); exec(src, ns); solve_tk = ns['solve_tk']
rng = np.random.default_rng(3); poses = [np.array([*rng.uniform(-2, 2, 2), rng.uniform(-np.pi, np.pi)]) for _ in range(40)]
R = np.array([min(ref.tgt(*p), ref.tgtgt(*p)) for p in poses])
h = 0.25; hth = 2*np.pi/(4*int(round(np.pi/(2*h))))
for name, ts, tr in [("все τ=h", h, h), ("поворот τ=hθ, прямая h", h, hth), ("поворот τ=hθ/2, прямая h", h, hth/2),
                     ("поворот 2hθ, прямая h", h, 2*hth), ("поворот hθ, прямая 2h", 2*h, hth)]:
    A = solve_tk([ts, ts, tr, tr], h=h); d = np.array([dd.V_at(A, p) for p in poses]) / R
    print(f"{name:28s} (τ_прям={ts:.3f}, τ_пов={tr:.3f}): V/эталон mean {d.mean():.3f} max {d.max():.3f} min {d.min():.3f}", flush=True)
