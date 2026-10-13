"""Копия v7-источника с блоком SYS=maze4 (v7 не правим, он чужой): python3 mk_maze4_src.py SRC_DIR DST_DIR. Блок: дд с инерцией по скорости (x, y, θ, v), см. maze4.py."""
import sys, shutil, os
src, dst = map(os.path.expanduser, sys.argv[1:3]); shutil.copytree(src, dst, dirs_exist_ok=True); fn = os.path.join(dst, 'growN.py'); t = open(fn).read()
BLOCK = """elif SYS == 'maze4':                                                                                        # c6 (CA-MAZE 4D, bench/maze4.py): дд с инерцией (x, y, θ, v), v̇ = a − v, u = (a, ω) ромб |a|+|ω| ≤ 1, |v| ≤ 1, цель: ящик ±RHO по (x, y), θ и v любые
    N = 4; PERIOD = np.array([0, 0, PI2, 0]); RHO = float(E('RHO', .08)); RHOV = np.array([RHO, RHO, np.pi, 1.]); XL = float(E('XL', 6.)); XLV = np.array([XL, XL, 0., 1.]); US = ((1., 0.), (-1., 0.), (0., 1.), (0., -1.))
    def f(y, u): th = y[..., 2]; v = y[..., 3]; return np.stack([v * np.cos(th), v * np.sin(th), u[1] + 0 * th, u[0] - v], -1)
    def Bq(y): return np.diag([0., 0., 1., 1.])
"""
key = "else: raise SystemExit('SYS?')"; assert key in t and 'maze4' not in t; open(fn, 'w').write(t.replace(key, BLOCK + key)); print('ok', fn)
