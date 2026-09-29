"""Хвост атласа DI: ошибка запросов у линии насыщения |v|→vmax. Полосы по |v|/vmax, методы: билинейная, hybrid(c,v), bellman. Сетка 81×81 на [−4,4]. Запуск: python3 reports/atlas_vmax_query.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas.improve import Atlas, make_grid
from src.atlas.reference import T_star
C = make_grid(-4, 4, 81); D = C.copy()
rng = np.random.default_rng(0)
for vm in (1.0, 0.5):
    A_ = Atlas(C, D, vmax=vm)
    ref = lambda x, v: T_star(x, v, vmax=vm)
    bands = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0)]
    meths = {"bilinear": A_.bilinear, "hybrid": A_.hybrid, "bellman": A_.bellman}
    print(f"vmax={vm}")
    for lo, hi in bands:
        pts = [(rng.uniform(-1.5, 1.5), np.sign(rng.uniform(-1, 1)) * rng.uniform(lo, hi) * vm) for _ in range(400)]
        row = []
        for nm, f in meths.items():
            e = [abs(f(x, v) - ref(x, v)) for x, v in pts]
            e = [z for z in e if not np.isnan(z)]
            row.append(f"{nm}: n={len(e)}" + (f" mean {np.mean(e):.4f} max {np.max(e):.3f}" if e else " (nan везде)"))
        print(f"  |v|/vmax∈[{lo},{hi}): " + " | ".join(row), flush=True)
