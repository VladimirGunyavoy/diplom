"""Показ атласа v6 (numpy, src/atlas6) в Ursina: границы клеток линиями (стены, вход/выход, сегмент). Клавиша 9 — вкл/выкл. Точка мира — [x, Y, v] как в ortho_grid."""
import numpy as np
from ursina import color as ucolor
from src.atlas6.atlas import Atlas, seed_lattice

Y_NODES = 0.03


class AtlasView:
    def __init__(self, ctx, h=1.0, lim=3.0, r=0.3, tau=0.5, n_wall=6, name='atlas'):
        self._lm = ctx.line_manager
        self._names = []
        self._visible = True
        self.atlas = Atlas(seed_lattice((-lim, lim), (-lim, lim), h, h), r=r, tau=tau, alpha=0.7)
        pos, neg = ucolor.azure, ucolor.orange
        for k, cell in enumerate(self.atlas.cells):
            col = pos if cell.u > 0 else neg
            t = np.linspace(-cell.tau, cell.tau, n_wall + 1)
            polys = [cell.wall(-1, t), cell.wall(+1, t), cell.entry(), cell.exit(), cell.segment(np.array([-cell.r, cell.r]))]
            for pi, P in enumerate(polys):
                W = np.c_[P[:, 0], np.full(len(P), Y_NODES), P[:, 1]]
                for i in range(len(W) - 1):
                    ln = f'{name}_{k}_{pi}_{i}'
                    self._lm.create(ln, W[i], W[i + 1], col, 0.6 if pi < 2 else 0.9)
                    self._names.append(ln)
        print(f"[AtlasView] клеток {len(self.atlas.cells)}, линий {len(self._names)}")

    def toggle(self):
        self._visible = not self._visible
        for ln in self._names:
            self._lm.set_visible(ln, self._visible)
        print(f"[AtlasView] {'показан' if self._visible else 'скрыт'}")

    def tick(self):
        pass
