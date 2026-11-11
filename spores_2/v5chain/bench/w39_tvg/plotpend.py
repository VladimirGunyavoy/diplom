"""w39: pend — горизонт дерева (max Gb) vs T_ref стартов; старты на фоне облака узлов (θ,ω). Запуск как tvg.py (KVAUTO=1)"""
import sys, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:5]
import tvg
from tvg import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = mkS(TS); back = build_back(S, TAU, NB, RB); P = np.array(back[0]); G = np.array(back[1]); ag = Agent(S, back); kv = tvg.KV
X = env.X0; dd, j = ag.tree.query(S.wrap(X) / S.scale, k=1); j = j % ag.nP
print('узлов %d, горизонт дерева max Gb %.2f, мед %.2f, p90 %.2f; max W %.2f' % (len(P), G.max(), np.median(G), np.percentile(G, 90), ag.W.max()))
res = []
for i in range(len(X)):
    T, why, _ = episode(ag, X[i]); res.append((i, T / env.Tref[i] if T else np.inf, env.Tref[i], G[j[i]], dd[i], X[i]))
res.sort(key=lambda r: -r[1]); print('i  T/Tref  Tref  G(ближ.узел)  d_near  (θ, ω)')
for r in res: print('%2d %6.3f %5.2f %5.2f %6.3f (%.2f, %.2f)' % (r[0], r[1], r[2], r[3], r[4], r[5][0], r[5][1]))
fig, ax = plt.subplots(1, 2, figsize=(13, 5)); sub = np.random.default_rng(0).choice(len(P), min(len(P), 30000), replace=False)
sc = ax[0].scatter(P[sub, 0], P[sub, 1], c=G[sub], s=1, cmap='viridis'); plt.colorbar(sc, ax=ax[0], label='G узла (время до цели), с')
tail = [r[0] for r in res[:6]]
for i in range(len(X)):
    ax[0].plot(X[i, 0], X[i, 1], 'o', mfc='none', mec='red' if i in tail else 'white', ms=9 if i in tail else 5)
    ax[0].annotate(str(i), X[i], color='red' if i in tail else 'w', fontsize=7)
ax[0].set_xlabel('θ'); ax[0].set_ylabel('ω'); ax[0].set_title('pend: узлы дерева (цвет G), старты (красные — хвост T/Tref)')
ax[1].scatter([r[2] for r in res], [r[1] for r in res], c='k'); ax[1].axvline(G.max(), color='r', ls='--', label='горизонт дерева max G'); ax[1].set_xlabel('T_ref старта, с'); ax[1].set_ylabel('T/T_ref агента'); ax[1].legend()
plt.tight_layout(); plt.savefig('pend_tree_horizon.png', dpi=110)
