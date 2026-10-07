"""Тест stepper без Ursina: фоновая функция с паузами, N/M/C программно, выключение пауз, время, снимок-копия."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.stepper import stepper as S


def work(arr):
    S.pause('start', 1, a=arr)
    for i in range(3):
        S.pause('row', 3, i=i, a=lambda: arr * i)
        time.sleep(.02)
        if i == 1: S.pause('mid', 2, i=i)
    S.pause('done', 1, must=True, total=int(arr.sum()))
    return 'ok'


def test_nmc():
    arr = np.arange(3.); st = S.Stepper().start(work, arr)
    assert st.wait_paused(); v, s = st.poll(); assert s['phase'] == 'start' and s['lvl'] == 1
    arr[0] = 99; assert s['a'][0] == 0                                        # снимок — копия
    assert st.cmd('n') and st.wait_paused(); assert st.poll()[1]['phase'] == 'row'
    assert st.cmd('m') and st.wait_paused(); s = st.poll()[1]; assert s['phase'] == 'row' and s['i'] == 1   # M с lvl 3: следующая той же глубины
    st.cmd('c'); st.wait_paused(); assert st.poll()[1]['phase'] == 'done'       # C — до обязательной
    st.cmd('c'); st._th.join(2.); assert st.finished and st.result == 'ok'


def test_m_skips_fine():
    st = S.Stepper().start(work, np.zeros(3))
    st.wait_paused()
    for ph in ('row', 'row', 'mid'): st.cmd('n'); st.wait_paused(); assert st.poll()[1]['phase'] == ph
    st.cmd('m'); st.wait_paused(); assert st.poll()[1]['phase'] == 'done'    # M при lvl 2: row (3) пропущена; дальше done (1 ≤ 2)
    assert st.poll()[1]['total'] == 0
    st.cmd('c'); st.thread_done = st._th.join(2.); assert st.finished and st.result == 'ok'


def test_c_runs_to_must_and_cfg():
    st = S.Stepper(max_lvl=2, off={'mid'}, start_paused=True).start(work, np.ones(3))
    st.wait_paused(); assert st.poll()[1]['phase'] == 'start'
    st.cmd('n'); st.wait_paused(); assert st.poll()[1]['phase'] == 'done'    # row (lvl 3) и mid (off) выключены конфигом
    st.cmd('c'); st._th.join(2.); assert st.finished and st.error is None
    assert 'row' not in st.times and st.counts == {'start': 1, 'done': 1}


def test_times_table_and_abort():
    st = S.Stepper().start(work, np.ones(3)); st.wait_paused(); st.note_render(.001)
    for _ in range(3): st.cmd('n'); st.wait_paused()
    assert st.times['row'] >= .015 or st.counts['row'] >= 1
    assert st.table()[-1][0].startswith('render'); st.abort(); assert st.finished and st.error is None
    assert S.ACTIVE is None


def test_noop_without_stepper():
    S.pause('x', 1, a=1); assert S.ACTIVE is None


def test_console_log():
    sys.modules.pop('src.algo.growN', None)
    os.environ.update(SYS='di', M='3', KF='21', MAXC='3', NFAIL='20', DELTA='.03', RMAX='.5', TMAX='3', TQDM_MI='1000')
    from src.algo import growN as g
    layer = lambda: g.build_layer(g.US[1], np.random.default_rng(0))[0]
    out = []; st = S.Stepper(log_actions=False); st.echo = out.append; st.start(layer); st.cmd('c') if st.wait_paused(30) else None
    st._th.join(60); assert st.finished and st.error is None
    assert any(l.startswith('[stop] spore #1 ') and 'reason=' in l for l in out) and any(l.startswith('[cell] spore #1 rows') for l in out), out
    out = []; st = S.Stepper(); st.echo = out.append; st.start(layer)
    while st.wait_paused(30): st.cmd('n')
    st._th.join(5)
    for key in ('spore #1: seed at (', 'section built, 21 points', 'row ', 'grew sideways side', 'stopped:', 'done: rows', 'layer finished:'):
        assert any(key in l for l in out), (key, out)
    print('\n'.join(out[:12]))


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('ok', k)
