"""Рост ДИ 2D (SYS=di) в v8/src/algo/growN.py без Ursina: рост одного слоя с паузами и без — одинаковый результат; контракт снимка."""
import sys, os
os.environ.update(SYS='di', M='3', KF='21', MAXC='4', NFAIL='20', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.algo import growN as g
from src.stepper import stepper as S

REASONS = {'поле', 'изгиб', 'сосед', 'цель', 'TMAX', 'RMAX'}


def layer(): return g.build_layer(g.US[1], np.random.default_rng(0))[0]


def test_plain_grows():
    cells = layer(); assert 1 <= len(cells) <= 4
    for c in cells: assert c.G.shape[-1] == 2 and c.nf + c.nb >= 1


def test_stepped_same_and_contract():
    ref = layer(); st = S.Stepper(max_lvl=3).start(layer); seen = []; n = 0
    while st.wait_paused(30):
        v, s = st.poll(); seen.append(s); n += 1; st.cmd('n')
    st._th.join(5); assert st.error is None and st.finished and len(st.result) == len(ref)
    for a, b in zip(ref, st.result): assert np.allclose(a.c, b.c) and a.nf == b.nf and a.nb == b.nb
    ph = {s['phase'] for s in seen}; assert {'seed', 'section', 'row', 'stop', 'cell'} <= ph, ph
    for s in seen:
        assert s['u'] == g.US[1] and s['seed'].shape == (2,)
        if s['phase'] == 'stop': assert s['reason'] in REASONS, s['reason']
        if 'cell' in s: assert {'G', 'c', 'r', 'e', 'nb', 'nf'} <= set(s['cell'])
        if s['phase'] in ('seed', 'cell'): assert s['queue'].shape[1] == 2 and isinstance(s['cells'], list)
    assert seen[-1]['phase'] == 'cell' and len(seen[-1]['cells']) == len(ref)
    assert st.table()[0][1] >= 1; print('пауз', n, 'время по именам', [(r[0], r[1], round(r[2], 3)) for r in st.table()])


def test_levels_filter():
    st = S.Stepper(max_lvl=1).start(layer); ph = []
    while st.wait_paused(30): ph.append(st.poll()[1]['phase']); st.cmd('n')
    assert set(ph) <= {'seed', 'cell'} and ph[0] == 'seed'


def test_own_seed():
    cells = g.build_layer(g.US[1], np.random.default_rng(0), seeds=[[1., 0.]], only_queue=True)[0]
    assert len(cells) >= 1 and np.allclose(cells[0].c[0], 1., atol=.6)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
