"""Картинка «как считается V»: шесть сценариев на настоящем атласе, с числами в узлах. Запуск из этой папки (параметры атласа — окружением, как у compute.py):
TMAX=1.5 ORDER=1 RMIN=.01 VMIN=.2 OV=1 OVL=.05 python3 scenarios.py   → pics/scenarios.png"""
import numpy as np, sys, os, traceback
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '../../../../v5chain/reports/research'); sys.path.insert(0, R); os.chdir(R)
import di_grow_cells as G
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = G.Atlas().solve(); M, DT, BIG, RHO = G.M, G.DTN, G.BIG, G.RHO; cells = A.cells; offs = np.array([c.o for c in cells]); V = A.V; P = A.P
INK, MUT = '#1f2328', '#6b7280'; COL = {-1.: '#1f5fbf', 0.: '#5f6b7a', 1.: '#d9480f'}; BX = dict(boxstyle='round,pad=.3', fc='white', ec='#d0d5dd', lw=.6); HI = '#8b2fc9'
def cell_of(i): return cells[int(np.searchsorted(offs, i, 'right') - 1)]
def grid(c): return P[c.o:c.o + len(c.tt) * M].reshape(len(c.tt), M, 2), V[c.o:c.o + len(c.tt) * M].reshape(len(c.tt), M)
def fmt(v): return '?' if v >= BIG / 2 else '%.2f' % v
def draw_cell(a, c, nodes=True, lab=None, win=None, fs=6.5):
    g, v = grid(c); e1, e2 = g[:, 0], g[:, -1]; pg = np.r_[e1, e2[::-1]]; a.fill(pg[:, 0], pg[:, 1], fc=COL[c.u], alpha=.10, ec=COL[c.u], lw=1)
    for j in range(M): a.plot(g[:, j, 0], g[:, j, 1], '-', color=COL[c.u], lw=.5, alpha=.5)
    if nodes: a.plot(g[..., 0].ravel(), g[..., 1].ravel(), 'o', color=COL[c.u], ms=3, alpha=.8)
    if lab is not None:
        for p, x in zip(g.reshape(-1, 2), v.ravel()):
            if win is None or (abs(p[0] - win[0]) < win[2] and abs(p[1] - win[1]) < win[3]): a.text(p[0], p[1] + lab, fmt(x), color=INK, fontsize=fs, ha='center', va='bottom', clip_on=True)
def goal(a): a.add_patch(plt.Rectangle((-RHO, -RHO), 2 * RHO, 2 * RHO, fc='#b42318', alpha=.10, ec='#b42318', lw=1.6))
def arrow(a, p, q, col=INK): a.annotate('', q, p, arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8, shrinkA=3, shrinkB=3))
def win(a, c0, w, h): a.set_xlim(c0[0] - w, c0[0] + w); a.set_ylim(c0[1] - h, c0[1] + h)
def note(a, s, loc=(.02, .02), va='bottom'): a.text(loc[0], loc[1], s, transform=a.transAxes, color=INK, fontsize=9.5, bbox=BX, va=va)
ST = {u: A.stencils(G.flow(P, u, DT)) for u in G.US}                                           # пробы: узел + управление → (чужая клетка, 4 узла, 4 веса)
def rows(i, u):
    I, IDX, W = ST[u]; k = np.flatnonzero(I == i); out = []
    for kk in k: bad = ((W[kk] > 1e-6) & (V[IDX[kk]] >= BIG / 2)).any(); out.append((cell_of(IDX[kk][0]), IDX[kk], W[kk], BIG if bad else float((W[kk] * V[IDX[kk]]).sum())))
    return out
own = np.array([cell_of(c.o).u for c in cells for _ in range(len(c.tt) * M)]); fin = (V < BIG / 2) & ~A.goal & G.inbox(P)
fig, ax = plt.subplots(3, 2, figsize=(15, 20), dpi=110); fig.patch.set_facecolor('white'); ax = ax.ravel()
def panel(k, title, f):
    a = ax[k]; a.set_title(title, color=INK, fontsize=12, loc='left')
    try: f(a)
    except Exception as e: a.text(.5, .5, 'не нашлось примера:\n' + repr(e)[:120], transform=a.transAxes, ha='center', color='#b42318'); traceback.print_exc()
    a.set_xlabel('положение x', color=INK); a.set_ylabel('скорость v', color=INK); a.tick_params(colors=MUT, labelsize=8); a.grid(color='#eef0f3', lw=.6); a.set_axisbelow(True)
    for sp in a.spines.values(): sp.set_color('#d0d5dd')
cg = max([c for c in cells if c.u == -1.], key=lambda c: int(A.goal[c.o:c.o + len(c.tt) * M].sum()))   # клетка атласа u = −1, сильнее всех накрывшая цель
def p1(a):
    draw_cell(a, cg, lab=.004, win=(0, .12, .34, .46)); goal(a); win(a, (0, .12), .34, .46); a.plot(*cg.c, 'o', color=HI, ms=9, mec='white', mew=1, zorder=6)
    note(a, 'Клетка атласа u = −1 прошла через цель (красный квадрат).\nТочки — узлы таблицы: 5 траекторий × срезы через %.2f с.\nЧисла — время до цели; поток идёт сверху вниз (v убывает).\nВ цели 0. В столбцах, проходящих ЧЕРЕЗ цель, выше неё время растёт\nна ~%.2f за срез — езда по своей траектории. Остальные узлы (левые\nстолбцы, всё ниже цели) своей траекторией в цель не попадают —\nих числа получены переключением в другие атласы.' % (DT, DT), (.02, .98), 'top')
def p2(a):
    best = None
    for c in cells:
        g, v = grid(c)
        for j in range(M):
            for i in range(len(c.tt) - 1):
                if .3 < v[i, j] < 1.5 and v[i + 1, j] < BIG / 2 and abs(v[i, j] - DT - v[i + 1, j]) < 1e-9 and abs(g[i, j, 1]) > .5 and 3 < i < len(c.tt) - 4 and 0 < j < M - 1: best = (c, i, j); break
            if best: break
        if best: break
    c, it, j = best; g, v = grid(c); draw_cell(a, c, lab=.004, win=(g[it, j, 0], g[it, j, 1], .2, .24)); win(a, g[it, j], .2, .24); arrow(a, g[it, j], g[it + 1, j], HI); a.plot(*g[it, j], 's', color=INK, ms=9, zorder=7)
    note(a, 'Шаг по СВОЕЙ траектории: клетка атласа u = %+d, её управление, %.2f с.\nУзел (квадрат) попадает ровно в следующий узел своей траектории.\n\n   V = %.2f + %.3f = %.3f\n\nИнтерполяции нет: это точное время езды. Так число\nпередаётся вдоль всей траектории, срез за срезом.' % (c.u, DT, DT, v[it + 1, j], v[it, j]), (.02, .98), 'top')
def find_switch(two=False):
    for u in G.US:
        I, IDX, W = ST[u]; cand = np.flatnonzero(fin[I] & (own[I] != u) & (np.abs(P[I, 1]) > .7) & (np.abs(P[I, 0]) < 1.8))
        for kk in cand[::7]:
            i = I[kk]; rr = [r for r in rows(i, u) if r[3] < BIG / 2]
            if not rr: continue
            if not two and len(rr) == 1 and abs(DT + rr[0][3] - V[i]) < 1e-6 and min(rr[0][2]) > .12: return i, u, rr
            if two and len(rr) >= 2 and abs(rr[0][3] - rr[1][3]) > .01 and min(rr[0][2]) > .08 and min(rr[1][2]) > .08: return i, u, rr
    raise RuntimeError('нет подходящего узла')
def p3(a):
    i, u, rr = find_switch(); c0 = cell_of(i); ct, idx, w, val = rr[0]; q = G.flow(P[i], u, DT); draw_cell(a, c0, nodes=True); draw_cell(a, ct, lab=.004, win=(q[0], q[1], .16, .2)); win(a, q, .2, .24)
    a.fill(P[idx[[0, 1, 3, 2]], 0], P[idx[[0, 1, 3, 2]], 1], fc=HI, alpha=.18, ec=HI, lw=1.5); a.plot(P[idx, 0], P[idx, 1], 'o', color=HI, ms=9, mec='white', mew=1, zorder=6); arrow(a, P[i], q); a.plot(*P[i], 's', color=INK, ms=9, zorder=7); a.plot(*q, '*', color=INK, ms=14, zorder=7)
    note(a, 'ПЕРЕКЛЮЧЕНИЕ. Узел (квадрат) клетки атласа u = %+d пробует управление u = %+d.\nЧерез %.2f с он в точке ★ — она лежит в клетке ДРУГОГО атласа, внутри\nчетырёхугольника из 4 её узлов (фиолетовые). Время в ★ — среднее по ним с весами:\n\n   %s  =  %.3f\n\n   V узла = %.2f + %.3f = %.3f   (это его лучшая проба)' % (c0.u, u, DT, '  +  '.join('%.2f·%s' % (ww, fmt(V[k])) for ww, k in zip(w, idx)), val, DT, val, DT + val), (.02, .98), 'top')
def p4(a):
    i, u, rr = find_switch(two=True); q = G.flow(P[i], u, DT); win(a, q, .22, .26)
    for (ct, idx, w, val), col in zip(rr[:2], (HI, '#0f8a5f')):
        draw_cell(a, ct); a.fill(P[idx[[0, 1, 3, 2]], 0], P[idx[[0, 1, 3, 2]], 1], fc=col, alpha=.18, ec=col, lw=1.5); a.plot(P[idx, 0], P[idx, 1], 'o', color=col, ms=9, mec='white', mew=1, zorder=6)
        for k in idx: a.text(P[k, 0], P[k, 1] + .006, fmt(V[k]), color=INK, fontsize=7.5, ha='center', va='bottom')
    arrow(a, P[i], q); a.plot(*P[i], 's', color=INK, ms=9, zorder=7); a.plot(*q, '*', color=INK, ms=14, zorder=7)
    note(a, 'ТОЧКА В ДВУХ КЛЕТКАХ СРАЗУ (наложение соседей).\nТочка ★ лежит в двух клетках; у каждой свои 4 узла и своя оценка:\n\n   фиолетовая клетка: %.3f\n   зелёная клетка:    %.3f\n\nБерётся МЕНЬШАЯ: %.3f. Поэтому наложение не мешает,\nа щель между клетками — мешает (см. следующий сценарий).' % (rr[0][3], rr[1][3], min(rr[0][3], rr[1][3])), (.02, .98), 'top')
def p5(a):
    best = None
    for u in G.US:
        I, IDX, W = ST[u]; bad = ((W > 1e-6) & (V[IDX] >= BIG / 2)).any(1) & ((V[IDX] < BIG / 2).sum(1) >= 2) & (W.min(1) > .1); cand = np.flatnonzero(bad & fin[I] & (own[I] != u) & (np.abs(P[I]).max(1) < 2.2))
        if len(cand): k = cand[len(cand) // 2]; best = (I[k], u, IDX[k], W[k]); break
    i, u, idx, w = best; q = G.flow(P[i], u, DT); ct = cell_of(idx[0]); win(a, q, .2, .24); draw_cell(a, cell_of(i), nodes=False); draw_cell(a, ct, lab=.004, win=(q[0], q[1], .2, .24))
    a.fill(P[idx[[0, 1, 3, 2]], 0], P[idx[[0, 1, 3, 2]], 1], fc='#b42318', alpha=.12, ec='#b42318', lw=1.5); a.plot(P[idx, 0], P[idx, 1], 'o', color='#b42318', ms=9, mec='white', mew=1, zorder=6); arrow(a, P[i], q); a.plot(*P[i], 's', color=INK, ms=9, zorder=7); a.plot(*q, 'X', color='#b42318', ms=13, zorder=7)
    note(a, 'ПРОБА БЕЗ РЕЗУЛЬТАТА. Узел (квадрат) пробует u = %+d, точка прибытия ✕\nлежит в чужой клетке, но среди её 4 узлов (красные) есть «?» —\nузел, у которого пути к цели нет (например, поток уносит за край области).\nСреднее с «?» не считается, проба ничего не даёт.\nУзел берёт время из других проб: у него V = %s.\n\nЩелей «точка вообще ни в одной клетке» у узлов с известным временем\nв этом атласе не нашлось.' % (u, fmt(V[i])), (.02, .98), 'top')
def p6(a):
    best = None
    for u in G.US:
        tg = A.tgoal(P, u); cand = np.flatnonzero(np.isfinite(tg) & (tg < DT - 1e-9) & ~A.goal & (np.abs(V - tg) < 1e-9))
        if len(cand): i = cand[np.argmax(tg[cand])]; best = (i, u, tg[i]); break
    i, u, t = best; c0 = cell_of(i); draw_cell(a, c0, lab=.003, win=(0, 0, .3, .3)); goal(a); win(a, (P[i] + 0) / 2, .3, .3); tt = np.linspace(0, DT, 9); tr = G.flow(P[i], u, tt[:, None] * 0 + tt[:, None])[:, 0] if False else np.array([G.flow(P[i], u, x) for x in tt])
    a.plot(tr[:, 0], tr[:, 1], '-', color=INK, lw=1.5); a.plot(tr[:, 0], tr[:, 1], '.', color=INK, ms=5); a.plot(*P[i], 's', color=INK, ms=9, zorder=7); a.plot(*G.flow(P[i], u, t), '*', color='#b42318', ms=14, zorder=7)
    note(a, 'ВХОД В ЦЕЛЬ ПОСРЕДИ ШАГА. Узел (квадрат) под u = %+d\nвлетает в цель раньше, чем кончится шаг %.2f с.\nШаг проверяется в 8 промежуточных точках;\nвремя узла = время до первой точки внутри цели:\n\n   V = %.4f с' % (u, DT, t), (.02, .98), 'top')
panel(0, '1. Клетка налетела на цель: откуда берутся первые числа', p1)
panel(1, '2. Шаг по своей траектории — время растёт точно', p2)
panel(2, '3. Проба чужого управления: попадание в соседнюю клетку и 4 узла', p3)
panel(3, '4. Точка прибытия в двух клетках — берётся меньшее', p4)
panel(4, '5. Проба, которая ничего не даёт', p5)
panel(5, '6. Вход в цель посреди шага', p6)
fig.suptitle('Как считается время до цели V: шесть сценариев на настоящем атласе ДИ (спор %d, узлов %d)\nЦвет клетки — её атлас: синий u = −1, серый u = 0, оранжевый u = +1. Числа у узлов — время до цели, «?» — пути нет' % (len(cells), A.N), color=INK, fontsize=12, x=.02, ha='left', y=.995)
fig.tight_layout(rect=(0, 0, 1, .97)); out = os.path.join(HERE, 'pics', 'scenarios.png'); fig.savefig(out); print('saved', out)
