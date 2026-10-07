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
    assert 'СТОП' not in caption(d.snapshot) or d.snapshot['reason']


def test_live_steps_and_final():
    L = LiveStepper(seed=(1., 0.)); assert L.snapshot['phase'] == 'seed'
    L.key('N'); assert L.snapshot['phase'] == 'section' and len(build_geometry(L.snapshot)['points_cur']) > 0
    L.key('C'); assert L.snapshot['phase'] == 'done' and len(L.snapshot['cells']) > 0
    g = build_geometry(L.snapshot); assert len(g['segs_core_done']) > 0 and len(g['points_done']) > 0


def test_timetable():
    t = TimeTable(); t.add_draw('row', .002); assert 'row' in t.text({'row': .1})
