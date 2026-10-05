"""«Евал»: читает <эксперимент>/data/ и рисует <эксперимент>/pics/atlas.png; с --watch перерисовывает при изменении данных (не чаще раза в EVERY секунд, по умолчанию 60).
Запуск из этой папки: python3 plot.py 01_имя-эксперимента [--watch]"""
import numpy as np, sys, os, json, time, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); tag = [a for a in sys.argv[1:] if not a.startswith('--')][0]; WATCH = '--watch' in sys.argv; D = os.path.join(HERE, tag, 'data'); OUT = os.path.join(HERE, 'pics', tag + '.png'); os.makedirs(os.path.join(HERE, 'pics'), exist_ok=True)   # все картинки в одной папке, имя = номер и название эксперимента
BIG = 1e3; INK, MUT = '#1f2328', '#6b7280'; PAL = ['#1f5fbf', '#5f6b7a', '#d9480f']; SP = '#8b2fc9'; BX = dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6)
def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    try: return json.load(open(p)) if name.endswith('.json') else (pickle.load(open(p, 'rb')) if name.endswith('.pkl') else dict(np.load(p)))
    except Exception: return None
def draw():
    st, cells, val, ag, paths = load('status.json') or {}, load('cells.pkl') or [], load('value.npz'), load('agent.npz'), load('paths.pkl'); pr = st.get('params') or {}; US = pr.get('US', [-1, 0, 1]); XL, WL, PER, RHO = pr.get('XL', 3.14), pr.get('WL', 3.5), pr.get('PER'), pr.get('RHO', .1)
    sh = [0.] if not PER else [k * PER for k in (-1, 0, 1, 2)]; XV = (-XL, XL) if not PER else (-1., 7.); fig, ax = plt.subplots(2, 2, figsize=(14, 13.6), dpi=110); fig.patch.set_facecolor('white'); ax = ax.ravel(); xn = 'угол φ (0 и 2π — верх, π — низ; полоса шире периода, часть точек видна дважды)' if pr.get('SYS') == 'pend' else 'положение x'; yn = 'угловая скорость ω' if pr.get('SYS') == 'pend' else 'скорость v'
    for k, (a, u) in enumerate(zip(ax, US)):
        cl = [c for c in cells if abs(c['u'] - u) < 1e-9]
        for c in cl:
            g = c['G']; pg = np.r_[g[:, 0], g[::-1, -1]]; seg = g[[i for i in range(len(g)) if np.allclose(g[i, 2], c['c'], atol=1e-4)] or [0]][0]
            for s in sh:
                if pg[:, 0].max() + s < XV[0] or pg[:, 0].min() + s > XV[1]: continue
                a.fill(pg[:, 0] + s, pg[:, 1], fc=PAL[k], alpha=.15, ec=PAL[k], lw=.8); a.plot(seg[:, 0] + s, seg[:, 1], '-', color=INK, lw=1.1); a.plot(seg[[0, 1, 3, 4], 0] + s, seg[[0, 1, 3, 4], 1], 'o', color=INK, ms=2.2); a.plot(c['c'][0] + s, c['c'][1], 'o', color=SP, ms=6.5, mec='white', mew=.8, zorder=5)
        cov = (st.get('cover') or [None] * 3)[k]; con = (st.get('contact') or [None] * 3)[k]
        a.set_title('атлас u = %+g: спор %d%s%s' % (u, len(cl), '' if cov is None else ', покрыто %.1f%%' % (100 * cov), '' if not con or con.get('all4') is None else '\nсоседи со всех 4 сторон у %.0f%% спор (бока %.0f%%, торцы %.0f%%)' % (100 * con['all4'], 100 * con['side'], 100 * con['end'])), color=INK, fontsize=11, loc='left')
        a.text(.02, .02, 'фиолетовая точка — спора, чёрные — её клоны на нормальном отрезке;\nзакрашено — клетка (отрезок, пронесённый потоком этого u вперёд и назад)', transform=a.transAxes, color=INK, fontsize=8.5, bbox=BX, va='bottom')
    a = ax[3]
    if val is not None:
        V = np.where(val['V'] >= BIG / 2, np.nan, val['V']); cm = matplotlib.colors.LinearSegmentedColormap.from_list('b', plt.cm.Blues_r(np.linspace(0, .8, 64)))
        for s_ in sh: im = a.pcolormesh(val['gx'] + s_, val['gw'], V, cmap=cm, shading='auto', vmin=0, vmax=np.nanmax(V))
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.02); cb.set_label('V — время до цели, с (белое — нет пути)', color=INK); cb.outline.set_visible(False)
    for p in paths or []:
        P = p['p'].astype(float); first = True; col = '#d9480f' if P[-1, 0] >= P[0, 0] else '#0f8a5f'          # цвет по направлению: оранжевый — угол в итоге вырос, зелёный — уменьшился
        for s_ in sh:                                                                         # путь не рвём: рисуем целиком в каждой копии, где он виден
            if P[:, 0].max() + s_ < XV[0] or P[:, 0].min() + s_ > XV[1]: continue
            a.plot(P[:, 0] + s_, P[:, 1], color=col, lw=1.6); a.plot(P[0, 0] + s_, P[0, 1], 'o', color=col, ms=6, mec='white', mew=1.2)
            if first and XV[0] <= P[0, 0] + s_ <= XV[1]: a.annotate(('%.1f с' % p['T'] if np.isfinite(p['T']) else 'не дошёл') + ('' if p.get('ref') is None else ' / эталон %.1f' % p['ref']), (P[0, 0] + s_, P[0, 1]), (P[0, 0] + s_ + .06, P[0, 1] + .12), color=INK, fontsize=8, bbox=BX); first = False
    a.set_title('склейка трёх атласов: цена V и пути агента (старты и их зеркала; число — время до цели)\nоранжевый путь — угол в итоге вырос, зелёный — уменьшился', color=INK, fontsize=10.5, loc='left')
    for a in ax:
        a.set_xlim(*XV); a.set_ylim(-WL, WL); a.set_xlabel(xn, color=INK); a.set_ylabel(yn, color=INK); a.tick_params(colors=MUT, labelsize=8)
        for s_ in sh: a.add_patch(plt.Rectangle((-RHO + s_, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='#b42318', lw=1.4))
        if PER:
            for xb in (np.pi,): a.axvline(xb, color=MUT, lw=.8, ls=(0, (2, 3))); a.text(xb, WL * .97, ' низ (π)', color=MUT, fontsize=8, va='top')
            for xt in (0., 2 * np.pi): a.text(xt, -WL * .97, ' верх', color=MUT, fontsize=8, va='bottom')
        for sp in a.spines.values(): sp.set_color('#d0d5dd')
    head = '%s, три атласа, растущие споры  [%s]   этап: %s   %.0f с' % ({'pend': 'Маятник', 'di': 'Двойной интегратор'}.get(pr.get('SYS'), '?'), tag, st.get('stage', 'нет данных'), st.get('sec', 0))
    line2 = 'рост ≤ %s с в каждую сторону, полуширина ≤ %s, боковой заход %s, цель ±%s, узлов поперёк %s, шаг %s с   |   спор %s, узлов %s' % (pr.get('TMAX'), pr.get('RMAX'), pr.get('OVL'), RHO, pr.get('M'), pr.get('DTN'), st.get('cells', '—'), st.get('nodes', '—')); line3 = ''
    if ag is not None:
        fz = np.isfinite(ag['T']); r = ag['T'][fz] / ag['ref'][fz]; line3 = 'агент, %d стартов: дошли %.1f%%, T/эталон среднее %.3f, медиана %.3f, макс %.2f, переключений (медиана) %.0f' % (len(fz), 100 * fz.mean(), r.mean(), np.median(r), r.max(), np.median(ag['sw'][fz]))
    fig.suptitle(head + '\n' + line2 + ('\n' + line3 if line3 else ''), color=INK, fontsize=11, x=.02, ha='left', y=.995); fig.tight_layout(rect=(0, 0, 1, .95)); tmp = OUT + '.tmp.png'; fig.savefig(tmp); plt.close(fig); os.replace(tmp, OUT)
def stamp(): return tuple(os.path.getmtime(os.path.join(D, f)) if os.path.exists(os.path.join(D, f)) else 0 for f in ('status.json', 'cells.pkl', 'value.npz', 'agent.npz', 'paths.pkl'))
last = None
while True:
    s = stamp()
    if s != last: draw(); last = s; print('нарисовано', OUT, time.strftime('%T'), flush=True)
    if not WATCH: break
    time.sleep(float(os.environ.get('EVERY', 60)))
