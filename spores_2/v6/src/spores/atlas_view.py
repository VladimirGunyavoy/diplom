"""Показ атласа v6 (numpy, src/atlas6) в Ursina: границы клеток линиями (стены, вход/выход, сегмент). Клавиша 9 — вкл/выкл. Точка мира — [x, Y, v] как в ortho_grid."""
import numpy as np
from ursina import color as ucolor
from src.atlas6.atlas import Atlas, seed_lattice

Y_NODES = 0.03


class AtlasView:
    def __init__(self, ctx, h=1.0, lim=3.0, r=0.1, tau=0.5, n_wall=6, name='atlas', Lx=4.0, Lv=2.0):
        self._lm = ctx.line_manager
        self._names = []
        self._visible = True
        self._gen = 0
        self._name, self._h, self._lim, self._r, self._tau, self._n_wall = name, h, lim, r, tau, n_wall
        self.Lx, self.Lv = Lx, Lv          # нормировка §7: x/Lx, v/Lv (DI, поле 4, a=1: x/4, v/2)
        self._build()

    def _build(self):
        for ln in self._names:
            self._lm.disable(ln)
        self._names = []
        self.atlas = Atlas(seed_lattice((-self._lim, self._lim), (-self._lim, self._lim), self._h, self._h),
                           r=self._r, tau=self._tau, alpha=0.7, Lx=self.Lx, Lv=self.Lv)
        pos, neg = ucolor.azure, ucolor.orange
        for k, cell in enumerate(self.atlas.cells):
            col = pos if cell.u > 0 else neg
            t = np.linspace(-cell.tau, cell.tau, self._n_wall + 1)
            polys = [cell.wall(-1, t), cell.wall(+1, t), cell.entry(), cell.exit(), cell.segment(np.array([-cell.r, cell.r]))]
            for pi, P in enumerate(polys):
                W = np.c_[P[:, 0], np.full(len(P), Y_NODES), P[:, 1]]
                for i in range(len(W) - 1):
                    ln = f'{self._name}_g{self._gen}_{k}_{pi}_{i}'
                    self._lm.create(ln, W[i], W[i + 1], col, 0.6 if pi < 2 else 0.9)
                    self._lm.set_visible(ln, self._visible)
                    self._names.append(ln)
        self._gen += 1
        print(f"[AtlasView] Lx={self.Lx:.2f} Lv={self.Lv:.2f}: клеток {len(self.atlas.cells)}, линий {len(self._names)}")

    def rescale(self, sign):
        """Клавиша масштаба оси скорости: Lv ×1.25^sign, атлас пересобирается (перпендикуляр к полю зависит от масштаба осей)."""
        self.Lv = min(max(self.Lv * 1.25 ** sign, 0.1), 20.0)
        self._build()

    def toggle(self):
        self._visible = not self._visible
        for ln in self._names:
            self._lm.set_visible(ln, self._visible)
        print(f"[AtlasView] {'показан' if self._visible else 'скрыт'}")

    def tick(self):
        pass
