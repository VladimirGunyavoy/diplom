"""Сюжет 1: слева — случайные равномерные узлы появляются; справа — обратное дерево от цели растёт волной (упрощённая build_back: шаг назад по времени на τ тремя u, прореживание радиусом R)."""
from vizlib import *
R = .07; rng = np.random.default_rng(0)
G = rng.uniform(-GH + .02, GH - .02, (30, 2)); nodes = [G]; rnd = [np.zeros(len(G), int)]; front = G; allP = G.copy(); r = 0
while len(front) and r < 90:
    r += 1; C = np.vstack([wrap(flow(front, u, -TAU)) for u in ACT]); C = C[okv(C)]; C = C[rng.permutation(len(C))]
    d, _ = cKDTree(emb(allP)).query(emb(C)); C = C[d > R]
    if len(C) == 0: break
    t = cKDTree(emb(C)); keep = np.ones(len(C), bool)
    for a, b in sorted(t.query_pairs(R)):
        if keep[a] and keep[b]: keep[b] = False
    C = C[keep]; nodes.append(C); rnd.append(np.full(len(C), r)); front = C; allP = np.vstack([allP, C])
Pt = np.vstack(nodes); Rt = np.concatenate(rnd); print('дерево: узлов', len(Pt), 'раундов', r)
Pu = np.c_[rng.uniform(-PI, PI, len(Pt)), rng.uniform(-OMAX, OMAX, len(Pt))]
nf = 40; frames = []; fig, axs = plt.subplots(1, 2, figsize=(14, 6.4)); fig.subplots_adjust(left=.05, right=.99, top=.84, bottom=.1, wspace=.14)
for f in range(nf + 1):
    fr = f / nf; nu = int(fr * len(Pu)); rr = fr * Rt.max(); m = Rt <= rr
    for ax in axs: ax.clear(); base_ax(ax)
    axs[0].set_title('Вариант 1: случайные равномерные узлы', loc='left'); axs[0].scatter(Pu[:nu, 0], Pu[:nu, 1], s=3, color='#2c7fb8', linewidths=0, zorder=2)
    axs[0].text(.02, .96, 'узлов: %d' % nu, transform=axs[0].transAxes, va='top', fontsize=14, bbox=dict(fc='w', ec='k'), zorder=20)
    axs[1].set_title('Вариант 2: обратное дерево от цели', loc='left'); sc = axs[1].scatter(Pt[m, 0], Pt[m, 1], s=3, c=Rt[m], cmap=plt.get_cmap('plasma_r', 28), vmin=0, vmax=Rt.max(), linewidths=0, zorder=2)
    axs[1].text(.02, .96, 'узлов: %d   раунд волны: %d\n(от узла идём назад по времени на τ, тремя u)' % (m.sum(), int(rr)), transform=axs[1].transAxes, va='top', fontsize=13, bbox=dict(fc='w', ec='k'), zorder=20)
    fig.suptitle('Откуда берутся узлы графа (одинаковое число узлов к концу). ★ — цель', fontsize=15, y=.97); frames.append(fig_to_img(fig))
frames[0].save('/tmp/c11_s1_first.png'); frames[-1].save('/tmp/c11_s1_last.png'); save_gif(frames, '/tmp/c11_method_nodes.gif', fps=6, hold_last=14)
