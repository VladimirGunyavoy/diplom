"""Картинка для пользователя (hub-research-4): дифдрайв — 4 слоя ромба U и эталон TGT vs TGTGT для бокового сдвига 0.3."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B, O, INK, GR, MUT = '#2a78d6', '#eb6834', '#222222', '#1baf7a', '#8a8a85'
def robot(a, x, y, th, col, s=0.09, alpha=1.0):
    c, sn = np.cos(th), np.sin(th); R = np.array([[c, -sn], [sn, c]])
    P = (R @ np.array([[1.3, -0.7, -0.7, 1.3], [0, 0.7, -0.7, 0]]) * s).T + [x, y]
    a.fill(P[:3, 0], P[:3, 1], color=col, alpha=alpha, lw=0)
def run(seq, q):
    out = [q]; x, y, th = q
    for kind, d in seq:
        n = 30
        for k in range(1, n + 1):
            if kind == 'T': out.append((x, y, th + d * k / n))
            else: out.append((x + d * k / n * np.cos(th), y + d * k / n * np.sin(th), th))
        x, y, th = out[-1]
    return np.array(out)
fig, ax = plt.subplots(1, 2, figsize=(16, 6.6))
a = ax[0]; a.set_title('1. Состояние (x, y, θ); управление (v, ω) в ромбе |v| + |ω| ≤ 1\nвершины ромба = 4 слоя', loc='left')
for (dx, dy, lab, kind, col) in ((0, 0, 'G+: вперёд\n(v=1, ω=0)', 'G', B), (2.2, 0, 'G−: назад\n(v=−1, ω=0)', 'Gb', B), (0, -2.0, 'T+: поворот влево\nна месте (ω=+1)', 'T', O), (2.2, -2.0, 'T−: поворот вправо\nна месте (ω=−1)', 'Tr', O)):
    robot(a, dx, dy, 0.0, INK, s=0.25)
    if kind == 'G': a.annotate('', (dx + 0.9, dy), (dx + 0.35, dy), arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    if kind == 'Gb': a.annotate('', (dx - 0.9, dy), (dx - 0.25, dy), arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    if kind in ('T', 'Tr'):
        t = np.linspace(0.3, 2.4, 40) * (1 if kind == 'T' else -1); a.plot(dx + 0.45 * np.cos(t), dy + 0.45 * np.sin(t), color=col, lw=2.5)
        a.annotate('', (dx + 0.45 * np.cos(t[-1]), dy + 0.45 * np.sin(t[-1])), (dx + 0.45 * np.cos(t[-3]), dy + 0.45 * np.sin(t[-3])), arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
    a.text(dx - 0.6, dy - 0.95, lab, fontsize=10, color=INK)
a.text(-0.6, 1.0, 'Оптимум по Balkcom–Mason (2002):\nтолько повороты на месте (T) и прямые (G),\nне больше 5 кусков', fontsize=10, color=INK)
a.set_xlim(-1.0, 3.4); a.set_ylim(-3.3, 1.7); a.set_aspect('equal'); a.axis('off')
a = ax[1]; q0 = (0.0, 0.3, 0.0)
tgt = run([('T', -np.pi / 2), ('G', 0.3), ('T', np.pi / 2)], q0)
al = 0.27; s = 0.15 / np.sin(al); snake = run([('T', -al), ('G', s), ('T', 2 * al), ('G', -s), ('T', -al)], q0)
a.plot(tgt[:, 0], tgt[:, 1], color=O, lw=2.5, label='TGT: повернуться, проехать, повернуться — %.2f с' % (np.pi + 0.3))
a.plot(snake[:, 0], snake[:, 1], color=B, lw=2.5, label='TGTGT «змейка» (вперёд, затем назад) — %.2f с' % (4 * al + 0.3 / np.sin(al)))
for P, col in ((snake, B),):
    for k in range(0, len(P), 30): robot(a, *P[k], col, s=0.035, alpha=0.6)
robot(a, *q0, INK, s=0.05); robot(a, 0, 0, 0, GR, s=0.05)
a.text(0.02, 0.33, 'старт (0, 0.3, θ=0)', fontsize=10, color=INK); a.text(0.02, -0.06, 'цель (0, 0, θ=0)', fontsize=10, color=INK)
a.set_title('2. Эталон: боковой сдвиг на 0.3 при том же курсе\n«змейка» на 37% быстрее, чем повернуться и проехать', loc='left')
a.set_xlim(-0.3, 0.75); a.set_ylim(-0.15, 0.42); a.set_aspect('equal'); a.legend(loc='lower right', fontsize=9); a.set_xlabel('x'); a.set_ylabel('y')
for sp in ('top', 'right'): a.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig('reports/research/figs/explain_dd.png', dpi=100)
