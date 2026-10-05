"""«Евал»: читает <эксперимент>/data/ и рисует <эксперимент>/pics/atlas.png; с --watch перерисовывает при изменении данных (не чаще раза в EVERY секунд, по умолчанию 60).
Запуск из этой папки: python3 plot.py 01_имя-эксперимента [--watch]"""
import numpy as np, sys, os, json, time, pickle
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = os.path.dirname(os.path.abspath(__file__)); tag = [a for a in sys.argv[1:] if not a.startswith('--')][0]; WATCH = '--watch' in sys.argv; D = os.path.join(HERE, tag, 'data'); OUT = os.path.join(HERE, 'pics', tag + os.environ.get('PSUF', '') + '.png'); os.makedirs(os.path.join(HERE, 'pics'), exist_ok=True)   # все картинки в одной папке, имя = номер и название эксперимента
BIG = 1e3; INK, MUT = '#1f2328', '#6b7280'; PAL = ['#1f5fbf', '#5f6b7a', '#d9480f']; SP = '#8b2fc9'; BX = dict(boxstyle='round,pad=.25', fc='white', ec='#d0d5dd', lw=.6)
def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    try: return json.load(open(p)) if name.endswith('.json') else (pickle.load(open(p, 'rb')) if name.endswith('.pkl') else dict(np.load(p)))
    except Exception: return None
def draw():
    st, cells, val, ag, paths = load('status.json') or {}, load('cells.pkl') or [], load('value.npz'), load('agent.npz'), load('paths.pkl'); pr = st.get('params') or {}; US = pr.get('US', [-1, 0, 1]); XL, WL, PER, RHO = pr.get('XL', 3.14), pr.get('WL', 3.5), pr.get('PER'), pr.get('RHO', .1)
    sh = [0.] if not PER else [k * PER for k in (-1, 0, 1, 2)]; XV = (-XL, XL) if not PER else (-1., 7.); fig, ax = plt.subplots(2, 3, figsize=(22, 13.8), dpi=100); fig.patch.set_facecolor('white'); ax = ax.ravel(); xn = 'угол φ (0 и 2π — верх, π — низ; полоса шире периода, часть точек видна дважды)' if pr.get('SYS') == 'pend' else 'положение x'; yn = 'угловая скорость ω' if pr.get('SYS') == 'pend' else 'скорость v'
    for k, (a, u) in enumerate(zip(ax, US)):
        cl = [c for c in cells if abs(c['u'] - u) < 1e-9]
        SPMS, CLMS = float(os.environ.get('SPMS', 6.5)), float(os.environ.get('CLMS', 2.2)); msz = {}
        if int(os.environ.get('ADMS', 0)) and len(cl) > 1:                                    # ADMS=1 (слово пользователя 2026-10-06): размер точки по расстоянию до ближайшей споры на экране — густо → мелко
            from scipy.spatial import cKDTree; bb = a.get_position(); kx = bb.width * fig.get_figwidth() * 72 / (XV[1] - XV[0]); kw = bb.height * fig.get_figheight() * 72 / (2 * WL)
            C = np.array([c['c'] for c in cl], float); CC = np.concatenate([C + np.array([s_, 0.]) for s_ in ([0.] if not PER else [-PER, 0., PER])]) * np.array([kx, kw]); dd, _ = cKDTree(CC).query(C * np.array([kx, kw]), 2)
            for i, c in enumerate(cl): sp = float(np.clip(.6 * dd[i, 1], .8, SPMS)); msz[id(c)] = (sp, float(np.clip(.4 * sp, .5, CLMS)))
        for c in cl:
            g = c['G'].astype(float); g = g[:, g.shape[1] // 2:g.shape[1] // 2 + 1] + (g - g[:, g.shape[1] // 2:g.shape[1] // 2 + 1]) / 1.1; pg = np.r_[g[:, 0], g[::-1, -1]]; seg = g[[i for i in range(len(g)) if np.allclose(g[i, 2], c['c'], atol=1e-4)] or [0]][0]   # только ядро (без гало 10%)
            for s in sh:
                if pg[:, 0].max() + s < XV[0] or pg[:, 0].min() + s > XV[1]: continue
                a.fill(pg[:, 0] + s, pg[:, 1], fc=PAL[k], alpha=.15, ec=PAL[k], lw=.8); a.plot(seg[:, 0] + s, seg[:, 1], '-', color=INK, lw=1.1); a.plot(seg[[0, 1, 3, 4], 0] + s, seg[[0, 1, 3, 4], 1], 'o', color=INK, ms=msz.get(id(c), (SPMS, CLMS))[1]); a.plot(c['c'][0] + s, c['c'][1], 'o', color=SP, ms=msz.get(id(c), (SPMS, CLMS))[0], mec='white', mew=min(.8, .15 * msz.get(id(c), (SPMS, CLMS))[0]), zorder=5)
        cov = (st.get('cover') or [None] * 3)[k]; con = (st.get('contact') or [None] * 3)[k]
        a.set_title('атлас u = %+g: спор %d%s%s' % (u, len(cl), '' if cov is None else ', покрыто %.1f%%' % (100 * cov), '' if not con or con.get('all4') is None else '\nсоседи со всех 4 сторон у %.0f%% спор (бока %.0f%%, торцы %.0f%%)' % (100 * con['all4'], 100 * con['side'], 100 * con['end'])), color=INK, fontsize=11, loc='left')
        a.text(.02, .02, 'фиолетовая точка — спора, чёрные — её клоны на нормальном отрезке;\nзакрашено — клетка (отрезок, пронесённый потоком этого u вперёд и назад)', transform=a.transAxes, color=INK, fontsize=8.5, bbox=BX, va='bottom')
    a = ax[3]
    if val is not None:
        V = np.where(val['V'] >= BIG / 2, np.nan, val['V']); cm = matplotlib.colors.LinearSegmentedColormap.from_list('b', plt.cm.Greens_r(np.linspace(0, .85, 64)))
        for s_ in sh: im = a.pcolormesh(val['gx'] + s_, val['gw'], V, cmap=cm, shading='auto', vmin=0, vmax=np.nanmax(V))
        cb = fig.colorbar(im, ax=a, fraction=.046, pad=.02); cb.set_label('V — время до цели, с (белое — нет пути)', color=INK); cb.outline.set_visible(False)
    PSHOW, PALPHA = int(os.environ.get('PSHOW', 0)), float(os.environ.get('PALPHA', .6))         # показать PSHOW путей (равномерно: дальние точки по стартам), прозрачность PALPHA
    if paths and PSHOW and len(paths) > PSHOW:
        X0 = np.array([p['p'][0] for p in paths], float); sel = [0]; dmin = np.linalg.norm(X0 - X0[0], axis=1)
        for _ in range(PSHOW - 1): j = int(np.argmax(dmin)); sel.append(j); dmin = np.minimum(dmin, np.linalg.norm(X0 - X0[j], axis=1))
        paths = [paths[j] for j in sel]
    many = len(paths or []) > 20
    for p in paths or []:
        P = p['p'].astype(float); first = True; dtn = pr.get('DTN', .06)
        acc = np.diff(P[:, 1]) / dtn - (np.sin((P[1:, 0] + P[:-1, 0]) / 2) if pr.get('SYS') == 'pend' else 0.); cols = [PAL[k] for k in np.argmin(np.abs(acc[:, None] - np.array(US)[None]), 1)]   # управление на шаге — по приращению скорости
        for s_ in sh:                                                                         # путь не рвём: рисуем целиком в каждой копии, где он виден; цвет отрезка = управление
            if P[:, 0].max() + s_ < XV[0] or P[:, 0].min() + s_ > XV[1]: continue
            a.add_collection(LineCollection(np.stack([P[:-1] + np.array([s_, 0.]), P[1:] + np.array([s_, 0.])], 1), colors=cols, linewidths=1.2 if many else 2.2, linestyles=(0, (4, 2)), alpha=PALPHA, zorder=4)); a.plot(P[0, 0] + s_, P[0, 1], 'o', color=INK, ms=float(os.environ.get('STMS', 3)) if many else 6, mec='white', mew=.6 if many else 1.2, alpha=min(1., PALPHA * 1.5), zorder=5)
            if first and not many and XV[0] <= P[0, 0] + s_ <= XV[1]: a.annotate(('%.1f с' % p['T'] if np.isfinite(p['T']) else 'не дошёл') + ('' if p.get('ref') is None else ' / эталон %.1f' % p['ref']), (P[0, 0] + s_, P[0, 1]), (P[0, 0] + s_ + .06, P[0, 1] + .12), color=INK, fontsize=8, bbox=BX, zorder=6); first = False
    a.set_title('склейка трёх атласов: цена V (зелёная) и пути агента; число у старта — время до цели\nцвет пути = управление на этом участке: синий u = %+g, серый u = %+g, оранжевый u = %+g' % tuple(US), color=INK, fontsize=10.5, loc='left')
    for a in ax[:4]:
        a.set_xlim(*XV); a.set_ylim(-WL, WL); a.set_xlabel(xn, color=INK); a.set_ylabel(yn, color=INK); a.tick_params(colors=MUT, labelsize=8)
        for s_ in sh: a.add_patch(plt.Rectangle((-RHO + s_, -RHO), 2 * RHO, 2 * RHO, fill=False, ec='#b42318', lw=1.4))
        if PER:
            for xb in (np.pi,): a.axvline(xb, color=MUT, lw=.8, ls=(0, (2, 3))); a.text(xb, WL * .97, ' низ (π)', color=MUT, fontsize=8, va='top')
            for xt in (0., 2 * np.pi): a.text(xt, -WL * .97, ' верх', color=MUT, fontsize=8, va='bottom')
        for sp in a.spines.values(): sp.set_color('#d0d5dd')
    a = ax[4]                                                                                 # статистика 1: качество по стартам
    if ag is not None and np.isfinite(ag['T']).any():
        fz = np.isfinite(ag['T']); r = ag['T'][fz] / ag['ref'][fz]; bw = .025; cx = np.round(np.arange(.95, max(1.6, r.max() + .05), bw), 4); cnt = np.bincount(np.clip(np.round((r - cx[0]) / bw).astype(int), 0, len(cx) - 1), minlength=len(cx)); a.bar(cx, cnt, width=.7 * bw, align='center', color='#2f8f5b', ec='none')   # столбец по центру своего значения (слово пользователя: уже, центр под меткой x)
        for v, nm, ls in ((np.median(r), 'медиана %.3f' % np.median(r), '-'), (r.mean(), 'среднее %.3f' % r.mean(), (0, (4, 3)))): a.axvline(v, color=INK, lw=1.3, ls=ls); a.text(v, a.get_ylim()[1] * (.95 if ls == '-' else .85), ' ' + nm, color=INK, fontsize=9, va='top')
        a.axvline(1., color=MUT, lw=.8); a.set_title('качество агента: время / эталон, %d стартов (не дошли: %d)\nпереключений: медиана %.0f, макс %d' % (len(fz), int((~fz).sum()), np.median(ag['sw'][fz]), int(ag['sw'][fz].max())), color=INK, fontsize=10.5, loc='left')
        a.set_xlabel('время агента / эталонное время (1 — как эталон)', color=INK); a.set_ylabel('число стартов', color=INK)
    else: a.text(.5, .5, 'агент ещё не ехал', transform=a.transAxes, ha='center', color=MUT)
    a = ax[5]                                                                                 # статистика 2: размеры клеток
    if cells:
        dtn = pr.get('DTN', .06)
        for k, u in enumerate(US):
            cl = [c for c in cells if abs(c['u'] - u) < 1e-9]
            if cl: a.plot([(len(c['G']) - 1) * dtn for c in cl], [2 * c['r'] for c in cl], 'o', color=PAL[k], ms=4.5, alpha=.55, mec='white', mew=.4, label='u = %+g: %d спор, длительность медиана %.2f с' % (u, len(cl), np.median([(len(c['G']) - 1) * dtn for c in cl])))
        a.legend(fontsize=8.5, frameon=True, framealpha=.9, edgecolor='#d0d5dd', loc='lower right'); a.set_xlabel('длительность клетки, с (назад + вперёд от споры)', color=INK); a.set_ylabel('ширина отрезка споры', color=INK); a.set_ylim(0, None)
        a.set_title('размеры клеток: каждая точка — одна спора\nадаптация размера по форме: %s' % ('включена' if pr.get('ADAPT') else 'нет'), color=INK, fontsize=10.5, loc='left')
    for a in ax[4:]:
        a.tick_params(colors=MUT, labelsize=8); a.grid(color='#eef0f3', lw=.6); a.set_axisbelow(True)
        for sp in a.spines.values(): sp.set_color('#d0d5dd')
    head = '%s, три атласа, растущие споры  [%s]   этап: %s   %.0f с' % ({'pend': 'Маятник', 'di': 'Двойной интегратор'}.get(pr.get('SYS'), '?'), tag, st.get('stage', 'нет данных'), st.get('sec', 0))
    line2 = 'рост ≤ %s с в каждую сторону, полуширина ≤ %s, боковой заход %s, цель ±%s, узлов поперёк %s, шаг %s с   |   спор %s, узлов %s' % (pr.get('TMAX'), pr.get('RMAX'), pr.get('OVL'), RHO, pr.get('M'), pr.get('DTN'), st.get('cells', '—'), st.get('nodes', '—')); line3 = ''
    if ag is not None:
        fz = np.isfinite(ag['T']); r = ag['T'][fz] / ag['ref'][fz]; line3 = 'агент, %d стартов: дошли %.1f%%, T/эталон среднее %.3f, медиана %.3f, макс %.2f, переключений (медиана) %.0f' % (len(fz), 100 * fz.mean(), r.mean(), np.median(r), r.max(), np.median(ag['sw'][fz]))
    fig.suptitle(head + '\n' + line2 + ('\n' + line3 if line3 else ''), color=INK, fontsize=11, x=.02, ha='left', y=.995); fig.tight_layout(rect=(0, 0, 1, .945)); tmp = OUT + '.tmp.png'; fig.savefig(tmp); plt.close(fig); os.replace(tmp, OUT)
def stamp(): return tuple(os.path.getmtime(os.path.join(D, f)) if os.path.exists(os.path.join(D, f)) else 0 for f in ('status.json', 'cells.pkl', 'value.npz', 'agent.npz', 'paths.pkl'))
last = None
while True:
    s = stamp()
    if s != last: draw(); last = s; print('нарисовано', OUT, time.strftime('%T'), flush=True)
    if not WATCH: break
    time.sleep(float(os.environ.get('EVERY', 60)))
