"""Сюжет 5: дробление по невязке Беллмана (tvg_ref.py, DUMP из aida1 refP_0.npz: 70000 узлов + 5 раундов по +10%): добавленные узлы ложатся на кривые разрыва V."""
from vizlib import *
z = np.load('/tmp/c11_refP.npz'); P = z['P']; n0 = int(z['n0']); sizes = [n0]
for _ in range(5): sizes.append(sizes[-1] + int(.1 * sizes[-1]))
assert sizes[-1] == len(P), (sizes, len(P))
S = get_S(); G = grid_V(S, 'dl'); frames = []; fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw=dict(width_ratios=[1.5, 1])); fig.subplots_adjust(left=.05, right=.99, top=.86, bottom=.1, wspace=.12)
cx, cy, w = 1.4, -1.4, .6; rc = plt.get_cmap('autumn', 6)
for r in range(6):
    for k, ax in enumerate(axs):
        ax.clear(); base_ax(ax); heat(ax, *G, alpha=.55)
        for q in range(1, r + 1):
            A = P[sizes[q - 1]:sizes[q]]; ax.scatter(A[:, 0], A[:, 1], s=(5 if k == 0 else 14), color=plt.get_cmap('brg')(q / 5.) if False else ['#ffd700', '#ff9900', '#ff3300', '#cc0044', '#7a00cc'][q - 1], linewidths=0, zorder=3 + q, alpha=1 if q == r else .8)
        if k == 1: ax.set_xlim(cx - w, cx + w); ax.set_ylim(cy - w, cy + w); ax.set_title('лупа', loc='left'); ax.set_ylabel('')
    axs[0].plot([cx - w, cx + w, cx + w, cx - w, cx - w], [cy - w, cy - w, cy + w, cy + w, cy - w], 'k-', lw=1.5, zorder=15)
    axs[0].set_title('Раунд %d: узлов %d (+%d новых)' % (r, sizes[r], 0 if r == 0 else sizes[r] - sizes[r - 1]), loc='left')
    fig.suptitle('Дробление: новые узлы (цветные точки) добавляются там, где V не сходится с уравнением Беллмана — на кривых разрыва V', fontsize=13, y=.97)
    frames.append(fig_to_img(fig))
frames[0].save('/tmp/c11_s5_first.png'); frames[-1].save('/tmp/c11_s5_last.png'); save_gif(frames, '/tmp/c11_method_refine.gif', fps=1.2, hold_last=3)
