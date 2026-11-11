"""b15: картинки для пользователя: (1) статик q8: путь агента А (carfree free1k, PAD 0) и эталона OCP, диски; (2) динамик fixed (старт i): 4 кадра по времени — диски, облако (R+κσ(τ=H)), путь агента, оракул не рисуем (только T). Данные: data/env_A_static.json (PAD 0), env_A_fixed_e.03_h2.json, env_loss.json"""
import json, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '.'); import envs_dyn as D
Dd = '/home/rl/claude-work/projects/spore/reports/bench/data/'; P = '/home/rl/claude-work/projects/spore/reports/bench/pics/'
st = json.load(open(Dd + 'env_A_static.json'))['res']; loss = json.load(open(Dd + 'env_loss.json')); fx = json.load(open(Dd + 'env_A_fixed_e.03_h2.json'))['res']
orc = {r['i']: r['T'] for r in json.load(open(Dd + 'env_oracle.json')) if r['mode'] == 'fixed'}
i = 8; fig, ax = plt.subplots(1, 1, figsize=(5, 5)); a = ax
for x, y in D.C0: a.add_patch(plt.Circle((x, y), .7, color='.8'))
a.add_patch(plt.Circle((4, 3.6), .45, color='g', alpha=.3)); Yr = np.array(loss[i]['Yr']); Ya = np.array(st[i]['Y'])
a.plot(Yr[:, 0], Yr[:, 1], 'b', label='OCP T=%.2f' % loss[i]['ref']['T']); a.plot(Ya[:, 0], Ya[:, 1], 'r', label='агент А T=%.2f' % st[i]['T']); a.set_aspect('equal'); a.set_xlim(-5, 5); a.set_ylim(-5, 5); a.legend(); a.set_title('статик, старт q8: carfree + А')
plt.tight_layout(); plt.savefig(P + 'agentA_static_q8.png', dpi=90); plt.close()
j = int(sys.argv[1]) if len(sys.argv) > 1 else 5; r = fx[j]; Y = np.array(r['Y']); H = D.Hazards('fixed', 0); T = r['T']; ts = np.linspace(0, T, 4)
fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
for a, t in zip(axs, ts):
    for c, ctr in enumerate(H.pos(t)):
        a.add_patch(plt.Circle(ctr, .7, color='.75')); w = H.vel(t)[c]; kap = np.sqrt(-2 * np.log(.03)); sg = np.sqrt(.005 * 2 ** 3 / 3); m = ctr + 2 * w; a.add_patch(plt.Circle(m, .7 + kap * sg, fill=False, ls='--', color='orange'))
    a.add_patch(plt.Circle((4, 3.6), .45, color='g', alpha=.3)); k = min(int(t / .16), len(Y) - 1); a.plot(Y[:k + 1, 0], Y[:k + 1, 1], 'r'); a.plot(*Y[k, :2], 'ro'); a.set_aspect('equal'); a.set_xlim(-5, 5); a.set_ylim(-5, 5); a.set_title('t=%.1f c' % t)
fig.suptitle('динамик fixed, старт q%d: агент А (T=%.2f, оракул %.2f); серое — диски, оранжевое — облако на 2 с вперёд' % (r['i'], T, orc[r['i']])); plt.tight_layout(); plt.savefig(P + 'agentA_dyn_fixed.png', dpi=80)
