import sys, importlib.util; sys.path.insert(0, '.')
import numpy as np
sp = importlib.util.spec_from_file_location('mref', '../v5chain/reports/research/manip_kin_ref.py'); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
from src.atlas6.manip2 import solve_manip2
obs = [((1.2, 0.8), 0.3), ((-0.5, 1.4), 0.25)]; n = 48
A = solve_manip2(h=2 * np.pi / n, obstacles=obs); g, T = m.grid_T((0.0, 0.0), obs, n)
V = A['V']; fin = np.isfinite(T) & ~A['occ']; d = (V - np.where(np.isfinite(T), T, 0))[fin]
print('занято %.1f%%, достижимо(эталон) %.1f%%, V<K %.1f%%' % (100 * A['occ'].mean(), 100 * fin.mean(), 100 * (V < 29).mean()))
print('V − Дейкстра: mean %+.3f max %.3f min %.3f (цель: V=Rg-сдвиг; ребро с проверкой середины ⇒ V ≥ T)' % (d.mean(), d.max(), d.min()))
print('несовпадение достижимости:', int(((V < 29) != fin).sum()))
