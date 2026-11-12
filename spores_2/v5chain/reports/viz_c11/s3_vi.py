"""Сюжет 3: итерация ценностей — W по итерациям (волна от цели), итоговая V с контурами. Узлы на панели (Делоне-VI, шаг 3 итерации на кадр)."""
from vizlib import *
from scipy.interpolate import griddata
S = get_S(); P = S['P']; hist = S['histd']; Wd = S['Wd']
fig, ax = plt.subplots(figsize=(10, 6.6)); fig.subplots_adjust(left=.08, right=.86, top=.88, bottom=.11)
cax = fig.add_axes([.89, .11, .025, .77]); frames = []
sel = list(range(0, len(hist), 1)); sel += [len(hist) - 1] * 1
for k, h in enumerate(sel):
    ax.clear(); cax.clear(); base_ax(ax); it = h * 3; W = hist[h]; fin = W < 1e2; last = (k == len(sel) - 1)
    if last: W = Wd; fin = W < 1e2; it = S['itd']
    sc = ax.scatter(P[fin, 0], P[fin, 1], c=np.minimum(W[fin], VMAX), s=5, cmap=CMAP, vmin=0, vmax=VMAX, marker='s', linewidths=0, zorder=2)
    ax.scatter(P[~fin, 0], P[~fin, 1], s=3, color='#dddddd', marker='s', linewidths=0, zorder=1)
    ax.plot([0], [0], marker='*', color='gold', ms=20, mec='k', zorder=9)
    if last:
        gx, gy, V = grid_V(S, 'dl'); ax.contour(gx, gy, V, levels=np.arange(1, 14, 1.), colors='k', linewidths=.6, zorder=6); tt = 'Готово: итоговая V с линиями уровня (каждые 1 с)'
    else: tt = 'Итерация %d: узлы, откуда цель уже достижима, получили оценку W' % it
    ax.set_title(tt, loc='left'); fig.colorbar(sc, cax=cax).set_label('W — время до цели, с')
    ax.text(.02, .96, 'W(узел) = min по u [ τ + W(конец дуги) ], цель: W = 0', transform=ax.transAxes, va='top', fontsize=12, bbox=dict(fc='w', ec='k', alpha=.9), zorder=20)
    frames.append(fig_to_img(fig))
frames[0].save('/tmp/c11_s3_first.png'); frames[-1].save('/tmp/c11_s3_last.png'); save_gif(frames, '/tmp/c11_method_vi.gif', fps=6, hold_last=14)
