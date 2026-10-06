import pickle, sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = pickle.load(open(sys.argv[1], 'rb')); V = D['V']; fig, axs = plt.subplots(1, 3, figsize=(21, 7))
for ax, uu in zip(axs, (-.3, 0., .3)):
    for G, u, o in zip(D['G'], D['u'], D['o']):
        if u != uu: continue
        nt, m = G.shape[:2]; v = V[o:o + nt * m].reshape(nt, m); fin = v < 500
        col = 'g' if fin.all() else ('orange' if fin.any() else 'r')
        ax.plot(G[[0, -1], :, 0].T, G[[0, -1], :, 1].T, '-', c=col, lw=.6); ax.plot(G[:, [0, -1], 0], G[:, [0, -1], 1], '-', c=col, lw=.6)
    ax.add_patch(plt.Rectangle((-.1, -.1), .2, .2, fill=False, ec='b')); ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.6, 3.6); ax.set_title('u=%g  green all V finite, orange part, red none' % uu)
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=70)
