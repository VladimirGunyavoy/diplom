"""Клетка DI v6 против аналитики (source_doc §3). Запуск: python3 tests/test_atlas6_cell.py (pytest на хабе нет)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.cell import Cell, flow

def test_center0_walls_parabolas():
    c = Cell((0, 0), +1, r=0.3, tau=1.0)                   # сегмент горизонтален, стены x = ±r + t²/2
    assert np.allclose(c.d, [-1, 0]) or np.allclose(c.d, [1, 0])
    t = np.linspace(-1, 1, 9)
    for sg in (-1, 1):
        w = c.wall(sg, t)
        assert np.allclose(np.abs(w[:, 0] - t ** 2 / 2), 0.3) and np.allclose(w[:, 1], t)
    assert np.isclose(c.stretch(), 1.0)                     # длина торца = длине сегмента

def test_stretch_doc_numbers():
    c = Cell((0, 2), +1, r=0.1, tau=1.0)                    # §3: 0.89 / 1.0 / 1.61 при t = 0.5 / 1 / 2
    got = [c.stretch(t, physical=True) for t in (0.5, 1.0, 2.0)]
    assert np.allclose(got, [0.894, 1.0, 1.612], atol=2e-3), got

def test_exit_entry_straight():
    c = Cell((0, 2), -1, r=0.2, tau=0.7)
    for e, t in ((c.exit(), c.tau), (c.entry(), -c.tau)):    # торец — образ отрезка при аффинном потоке = отрезок
        ss = np.linspace(-1, 1, 5) * c.r
        pts = flow(c.segment(ss), c.u, t)
        a, b = pts[1:] - pts[:-1], pts[-1] - pts[0]
        cr = a[:, 0] * b[1] - a[:, 1] * b[0]
        assert np.allclose(cr, 0, atol=1e-12)

def test_coords_roundtrip():
    rng = np.random.default_rng(0)
    for (cen, u, Lx, Lv) in (((0, 0), 1, 1, 1), ((0, 2), 1, 4, 2), ((-1, -1.5), -1, 4, 2)):
        c = Cell(cen, u, r=0.3, tau=1.0, Lx=Lx, Lv=Lv)
        for _ in range(50):
            s, t = rng.uniform(-c.r, c.r), rng.uniform(-0.8, 0.8)
            q = flow(c.segment(np.array(s)), u, t)
            st = c.coords(q)
            assert st is not None and np.isclose(st[0], s, atol=1e-8) and np.isclose(st[1], t, atol=1e-8), (cen, s, t, st)
            assert c.contains(q)
    assert not Cell((0, 0), 1, 0.3, 1.0).contains((3.0, 0.0))

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
