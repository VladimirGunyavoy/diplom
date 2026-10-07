"""Headless-просмотр (PLAN п.49): тот же поток пауз без Ursina, каждый снимок → PNG (matplotlib) + таблица времени по паузам в txt.
Запуск: python3 src/stepper/frames.py [шагов=30] [max_lvl=3] [MAXC=3] [папка=reports/frames/fine]   (ДИ 2D, слой u=+1, затравка случайная или SEED=x,v)"""
import sys, os, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(snap, g, ax):
    """снимок → оси: готовые клетки серым (контур + строки), текущая клетка (контур с гало пунктиром, ядро сплошным, строки), сечение, затравка, очередь, грань, причина"""
    XL = float(g.XLV[0]); ax.set_xlim(-XL, XL); ax.set_ylim(-g.XLV[1], g.XLV[1]); ax.set_aspect('auto'); ax.grid(alpha=.2)
    ax.add_patch(plt.Ellipse((0, 0), *(2 * g.RHOV), fc='g', alpha=.3, lw=0) if getattr(g, 'GOALSHAPE', 'box') == 'ball' else plt.Rectangle(-g.RHOV, *(2 * g.RHOV), fc='g', alpha=.3, lw=0))
    def outline(c, col, ls, lw, rows=False, ker=False):
        G = c['G'].reshape(c['G'].shape[0], -1, 2)                                                  # (nt, M·m, n); для ДИ m=1 → (nt, M, 2)
        if ker: mid = G[:, G.shape[1] // 2:G.shape[1] // 2 + 1]; G = mid + (G - mid) / (1 + g.HALO)
        ring = np.concatenate([G[0], G[:, -1], G[-1][::-1], G[:, 0][::-1]]); ax.plot(*np.append(ring, ring[:1], 0).T, color=col, ls=ls, lw=lw)
        if rows:
            for r_ in G: ax.plot(*r_.T, color=col, lw=.5, alpha=.6)
    for c in snap.get('cells', []): outline(c, '0.55', '-', .8, rows=True)
    c = snap.get('cell')
    if c is not None:
        outline(c, 'C0', '--', 1., rows=True); outline(c, 'C0', '-', 1.8, ker=True)
    s = snap.get('section')
    if s is not None: ax.plot(*s.reshape(-1, 2).T, 'o-', color='C1', ms=3, lw=1, label='section (KF points)')
    q = snap.get('queue')
    if q is not None and len(q): ax.plot(*q.T, '^', color='C4', ms=5, label='queue (%d)' % len(q))
    f = snap.get('face')
    if f is not None: ax.plot(*np.asarray(f).reshape(-1, 2).T, 'x', color='r', ms=6, label='face')
    ax.plot(*snap['seed'], '*', color='k', ms=13, label='seed')
    ttl = '%d · %s (lvl %d) · u=%g' % (snap['step'], snap['phase'], snap['lvl'], snap['u'])
    d = snap.get('dir'); dn = None if d is None else {'F': 'forward', 'B': 'backward'}.get(d[0], 'sideways axis %d %s' % (d[1], '+' if d[2] > 0 else '−') if d[0] == 'a' else str(d))
    if snap.get('reason'): ttl += ' · STOP: %s (%s)' % (snap['reason'], dn)
    elif dn: ttl += ' · %s' % dn
    ax.set_title(ttl, fontsize=9); ax.set_xlabel('x'); ax.set_ylabel('v'); ax.legend(fontsize=7, loc='upper right')


def run(steps=30, max_lvl=3, maxc=3, out=None, seed=None):
    os.environ.update(SYS='di', M='3', KF='21', MAXC=str(maxc), NFAIL='20', RMAX='.5', TMAX='3', TQDM_MI='1000')
    from src.algo import growN as g
    from src.stepper.stepper import Stepper
    out = out or os.path.join(ROOT, 'reports', 'frames', 'fine'); os.makedirs(out, exist_ok=True)
    kw = dict(seeds=[seed], only_queue=True) if seed is not None else {}
    st = Stepper(max_lvl=max_lvl).start(lambda: g.build_layer(g.US[1], np.random.default_rng(0), **kw)); log = []
    for k in range(steps):
        if not st.wait_paused(60): break
        v, s = st.poll(); t0 = time.perf_counter(); fig, ax = plt.subplots(figsize=(7, 5)); draw(s, g, ax); fig.tight_layout(); fig.savefig(os.path.join(out, 'f%03d_%s.png' % (s['step'], s['phase'])), dpi=80); plt.close(fig)
        st.note_render(time.perf_counter() - t0); log.append('%3d %-8s lvl%d %s %s' % (s['step'], s['phase'], s['lvl'], s.get('reason', ''), sorted(k_ for k_ in s if k_ not in ('phase', 'lvl', 'must', 'step', 'dt')))); st.cmd('n')
    st.abort()
    with open(os.path.join(out, 'timing.txt'), 'w') as fh:
        fh.write('steps %d, max_lvl %d, MAXC %d\n' % (len(log), max_lvl, maxc)); fh.write('%-22s %6s %9s %9s\n' % ('pause', 'n', 'total,s', 'mean,ms'))
        for r in st.table(): fh.write('%-22s %6d %9.3f %9.2f\n' % r)
        fh.write('\n' + '\n'.join(log) + '\n')
    return out, log


if __name__ == '__main__':
    a = sys.argv[1:]; steps = int(a[0]) if a else 30; lvl = int(a[1]) if len(a) > 1 else 3; mc = int(a[2]) if len(a) > 2 else 3
    seed = [float(x) for x in os.environ['SEED'].split(',')] if os.environ.get('SEED') else None
    o, l = run(steps, lvl, mc, a[3] if len(a) > 3 else None, seed); print(o, len(l), 'frames')
