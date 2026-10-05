"""«Евал»: читает data/<метка>/ и рисует pics/<метка>.png. С --watch перерисовывает, когда данные меняются (открыть PNG в VS Code — он обновляется сам).
Запуск из этой папки: python3 plot.py [метка] [--watch]"""
import numpy as np, sys, os, json, time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); args = [a for a in sys.argv[1:] if not a.startswith('--')]; tag = args[0] if args else 'run'; WATCH = '--watch' in sys.argv
D = os.path.join(HERE, 'data', tag); OUT = os.path.join(HERE, 'pics', tag + '.png'); os.makedirs(os.path.dirname(OUT), exist_ok=True)
L, RHO, BIG = 2.5, .1, 1e3; INK, MUT = '#1f2328', '#6b7280'; COL = {-1: '#1f5fbf', 0: '#5f6b7a', 1: '#d9480f'}; BX = dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6)
def flow(y, u, t): return np.stack([y[..., 0] + y[..., 1] * t + u * t * t / 2, y[..., 1] + u * t], -1)
def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    try: return json.load(open(p)) if name.endswith('.json') else dict(np.load(p))
    except Exception: return None                                                             # файл пишется прямо сейчас — возьмём на следующем круге
def draw():
    global RHO
    st, cells, val, ag, paths = load('status.json') or {}, load('cells.json') or [], load('value.npz'), load('agent.npz'), load('paths.json'); dtn = (st.get('params') or {}).get('DTN', .06); RHO = (st.get('params') or {}).get('RHO', .1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 13.4), dpi=110); fig.patch.set_facecolor('white'); ax = ax.ravel(); vv = np.linspace(-L, L, 200)
    for a, u in zip(ax, (-1, 0, 1)):
        cl = [c for c in cells if c['u'] == u]
        for c in cl:
            cc, n, r = np.array(c['c']), np.array(c['n']), c['r']; tt = np.arange(-c['nb'], c['nf'] + 1) * dtn; e1 = flow(cc + r * n, u, tt); e2 = flow(cc - r * n, u, tt); pg = np.r_[e1, e2[::-1]]
            a.fill(pg[:, 0], pg[:, 1], fc=COL[u], alpha=.15, ec=COL[u], lw=.8); sg = cc + np.linspace(-r, r, 5)[:, None] * n; a.plot(sg[:, 0], sg[:, 1], '-', color=INK, lw=1.2); a.plot(sg[[0, 1, 3, 4], 0], sg[[0, 1, 3, 4], 1], 'o', color=INK, ms=2.2); a.plot(sg[2, 0], sg[2, 1], 'o', color='#8b2fc9', ms=7, mec='white', mew=.8, zorder=5)
        cov = (st.get('cover') or [None] * 3)[(-1, 0, 1).index(u)]
        con = (st.get('contact') or [None] * 3)[(-1, 0, 1).index(u)]
        a.set_title('атлас u = %+d: спор %d%s%s' % (u, len(cl), '' if cov is None else ', покрыто %.1f%%' % (100 * cov), '' if not con or con.get('all4') is None else '\nсоседи со всех 4 сторон у %.0f%% спор (бока %.0f%%, торцы %.0f%%)' % (100 * con['all4'], 100 * con['side'], 100 * con['end'])), color=INK, fontsize=11, loc='left')
        a.text(.02, .02, 'фиолетовая точка — спора, чёрные — её клоны на нормальном отрезке;\nзакрашено — клетка (отрезок, пронесённый потоком этого u вперёд и назад)', transform=a.transAxes, color=INK, fontsize=8.5, bbox=BX, va='bottom')
    a = ax[3]
    if val is not None:
        V = np.where(val['V'] >= BIG / 2, np.nan, val['V']); im = a.pcolormesh(val['g'], val['g'], V, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', plt.cm.Blues_r(np.linspace(0, .8, 64))), shading='auto')
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.02); cb.set_label('V — время до цели, с (белое — нет пути)', color=INK); cb.outline.set_visible(False)
    a.plot(-vv * np.abs(vv) / 2, vv, color=INK, lw=1, ls=(0, (4, 3)))
    for p in paths or []:
        P = np.array(p['p']); a.plot(P[:, 0], P[:, 1], color='#d9480f', lw=2); a.plot(P[0, 0], P[0, 1], 'o', color='#d9480f', ms=6, mec='white', mew=1.2)
        a.annotate(('T %.2f' % p['T'] if p['T'] else 'не дошёл') + ' / точное %.2f' % p['Ts'], P[0], (P[0, 0] + .08, P[0, 1] + (.16 if P[0, 1] > 0 else -.26)), color=INK, fontsize=8.5, bbox=BX)
    a.set_title('склейка трёх атласов: цена V и пути агента (пунктир — точная кривая переключения)', color=INK, fontsize=10.5, loc='left')
    for a in ax:
        a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_xlabel('положение x', color=INK); a.set_ylabel('скорость v', color=INK); a.tick_params(colors=MUT, labelsize=8); a.add_patch(plt.Rectangle((-RHO, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='#b42318', lw=1.4)); a.set_aspect('equal')
        for sp in a.spines.values(): sp.set_color('#d0d5dd')
    pr = st.get('params') or {}; head = 'ДИ, три атласа, растущие споры  [%s]   этап: %s   %.0f с' % (tag, st.get('stage', 'нет данных'), st.get('sec', 0))
    line2 = 'рост по времени ≤ %s, полуширина ≤ %s, цель ±%s, боковой заход %s, узлов поперёк %s, шаг %s с   |   спор %s, узлов %s' % ('без предела' if pr.get('TMAX') is None else pr.get('TMAX'), pr.get('RMAX'), pr.get('RHO', .1), pr.get('OVL', '—'), pr.get('M'), pr.get('DTN'), st.get('cells', '—'), st.get('nodes', '—'))
    line3 = ''
    if ag is not None:
        f = np.isfinite(ag['T']); r = ag['T'][f] / ag['Ts'][f]
        line3 = 'агент, %d стартов: дошли %.1f%%, T/T* среднее %.3f, медиана %.3f, макс %.2f, переключений (медиана) %.0f' % (len(f), 100 * f.mean(), r.mean() if f.any() else np.nan, np.median(r) if f.any() else np.nan, r.max() if f.any() else np.nan, np.median(ag['sw'][f]) if f.any() else np.nan)
    fig.suptitle(head + '\n' + line2 + ('\n' + line3 if line3 else ''), color=INK, fontsize=11, x=.02, ha='left', y=.995); fig.tight_layout(rect=(0, 0, 1, .955))
    tmp = OUT + '.tmp.png'; fig.savefig(tmp); plt.close(fig); os.replace(tmp, OUT)
def stamp(): return tuple(os.path.getmtime(os.path.join(D, f)) if os.path.exists(os.path.join(D, f)) else 0 for f in ('status.json', 'cells.json', 'value.npz', 'agent.npz', 'paths.json'))
last = None
while True:
    s = stamp()
    if s != last: draw(); last = s; print('нарисовано', OUT, time.strftime('%T'), flush=True)
    if not WATCH: break
    time.sleep(float(os.environ.get("EVERY", 10)))                                            # не чаще раза в EVERY секунд
