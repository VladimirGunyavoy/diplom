"""GrowView без Ursina: геометрия снимка, подпись, таблица времени, живой рост ДИ через LiveStepper (v8, PLAN п.49)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.stepper.growview import build_geometry, caption, TimeTable
from src.stepper.demo import DemoStepper
from src.stepper.live import LiveStepper


def test_geometry_demo():
    d = DemoStepper(); d.key('C'); g = build_geometry(d.snapshot)
    assert g['points_done'].shape[1] == 3 and len(g['segs_rows_done']) > 0 and np.allclose(g['points_done'][:, 1], .01)
    assert 'STOP' not in caption(d.snapshot) or d.snapshot['reason']


def test_live_steps_and_final():
    L = LiveStepper(seed=(1., 0.)); assert L.snapshot['phase'] == 'seed'
    L.key('N'); assert L.snapshot['phase'] == 'section' and len(build_geometry(L.snapshot)['points_cur']) > 0
    L.key('C'); assert L.snapshot['phase'] == 'done' and len(L.snapshot['cells']) > 0
    g = build_geometry(L.snapshot); assert len(g['segs_core_done']) > 0 and len(g['points_done']) > 0


def test_timetable():
    t = TimeTable(); t.add_draw('row', .002); assert 'row' in t.text({'row': .1})


def test_outline_core_halo_and_segments():
    from src.stepper.growview import cell_outline, seg_index, HALO
    rr = .02; j = (np.arange(5) - 2) / 2                      # узлы поперёк: ±(1+HALO)·r, как Cell.build
    G = np.zeros((3, 5, 2)); G[:, :, 0] = (np.arange(3) * .05)[:, None]; G[:, :, 1] = j[None] * (1 + HALO) * rr
    for halo, w in ((True, (1 + HALO) * rr), (False, rr)):
        o = cell_outline(G, (0, 1), halo); z = o[..., 2]
        assert o.shape == (6, 2, 3) and abs(z.max() - w) < 1e-9 and abs(z.min() + w) < 1e-9     # 2·rows отрезков, ширина
    assert seg_index(3) == [(0, 1), (2, 3), (4, 5)]


def test_history_back_forward():
    L = LiveStepper(seed=None); steps = []
    for _ in range(5): L.key('N'); steps.append(L.snapshot['step'])
    assert L.hist_info is None
    L.back(); L.back()
    assert L.snapshot['step'] == steps[2] and L.hist_info == (4, 6), (L.snapshot['step'], steps, L.hist_info)       # 3-й шаг (история: 0-й, N1..N5)
    L.key('N'); L.key('N')
    assert L.snapshot['step'] == steps[4] and L.hist_info is None                                                    # снова живой 5-й
    L.key('N'); assert L.snapshot['step'] > steps[4] and L.hist_info is None                                         # 6-й — живой
    L.back(); L.key('M'); assert L.hist_info is None and L.snapshot['step'] >= steps[4]                                # M после Z: догнали живой и пошли дальше
    L.seed_at(None); assert L.history and len(L.history) == 1                                                        # seed_at очищает историю
