"""Сюжет 2: один узел — дуги трёх u длиной τ, конец дуги; (а) старая сшивка: ближайший узел + KV·d, (б) новая: треугольник Делоне и веса λ."""
from vizlib import *
S = get_S(); P = S['P']; kv = S['kv']; ip = S['ip']; Wn, Wd = S['Wn'], S['Wd']
i = int(np.argmin(np.hypot(P[:, 0] - 1.3, P[:, 1] + 1.4))); x = P[i]
E, OK, GL = edges(x[None]); E = E[0]; best = int(np.argmin([TAU + (S['tree'].query(emb(wrap(e)), k=9)[0].min()) for e in E]))
sb = int(np.argmin([TAU + Pol(S, 'dl').vh(e)[0] for e in E])); e = wrap(E[sb]); w = .16
frames = []; fig, axs = plt.subplots(1, 3, figsize=(15, 6)); fig.subplots_adjust(left=.04, right=.99, top=.84, bottom=.1, wspace=.12)
def ctx(ax, stage):
    ax.clear(); ax.set_xlim(x[0] - .5, x[0] + .5); ax.set_ylim(x[1] - .5, x[1] + .5); ax.set_facecolor('#f4f4f4'); ax.set_xlabel('θ'); ax.set_ylabel('ω')
    m = (np.abs(P[:, 0] - x[0]) < .5) & (np.abs(P[:, 1] - x[1]) < .5); ax.plot(P[m, 0], P[m, 1], '.', color='#9aa', ms=3)
    ax.plot(*x, 'o', color='w', mec='k', ms=14, zorder=8); ax.set_title('Узел и его дуги', loc='left')
    for s in range(min(3, stage)):
        a = np.array(arc(x, s)); ax.plot(a[:, 0], a[:, 1], color=COL[s], lw=3.5, label=UNAME[s]); ax.plot(*a[-1], 'o', color=COL[s], mec='k', ms=9, zorder=7)
    if stage >= 1: ax.legend(loc='upper left', fontsize=11)
    ax.text(.03, .03, 'каждая дуга — поток τ = 0.2 с\nпри постоянном u', transform=ax.transAxes, fontsize=11)
def zoom(ax, title):
    ax.clear(); ax.set_xlim(e[0] - w, e[0] + w); ax.set_ylim(e[1] - w, e[1] + w); ax.set_facecolor('#f4f4f4'); ax.set_xlabel('θ'); ax.set_title(title, loc='left', fontsize=14)
    m = (np.abs(P[:, 0] - e[0]) < w + .05) & (np.abs(P[:, 1] - e[1]) < w + .05); ax.plot(P[m, 0], P[m, 1], 'o', color='#667', ms=5)
    ax.plot(*e, 'o', color=COL[sb], mec='k', ms=15, zorder=9); ax.text(e[0] + .008, e[1] + .012, ' конец дуги', fontsize=12, zorder=10)
dq, jq = S['tree'].query(emb(e[None]), k=9); jq = jq[0]; dq = dq[0]; jn = jq[0]
J, lam, ok, sm = ip.weights(e[None]); J = J[0]; lam = lam[0]; tri = ip.X[ip.tri.simplices[sm[0]]]
def draw(stage):
    ctx(axs[0], min(stage, 3)); zoom(axs[1], '(а) старая сшивка: ближайший + KV·d'); zoom(axs[2], '(б) новая: Делоне + веса λ')
    if stage >= 4:   # ближайший
        pn = P[jn]; axs[1].plot(*pn, 'o', color='#ff7f00', ms=11, mec='k', zorder=8); axs[1].annotate('', pn, e, arrowprops=dict(arrowstyle='->', color='#ff7f00', lw=2.5))
        axs[1].text(.03, .97, 'берём ближайший узел j:\nV(конец) ≈ W_j + KV·d\n= %.2f + %.2f·%.3f = %.2f' % (Wn[jn], kv, dq[0], Wn[jn] + kv * dq[0]), transform=axs[1].transAxes, va='top', fontsize=12, bbox=dict(fc='w', ec='k'))
        axs[1].text(.03, .03, 'KV — «страховка»: чем дальше\nузел, тем больше добавка', transform=axs[1].transAxes, fontsize=11)
    if stage >= 5:   # треугольник
        T3 = np.vstack([tri, tri[:1]]); axs[2].fill(tri[:, 0], tri[:, 1], color='#4daf4a', alpha=.2); axs[2].plot(T3[:, 0], T3[:, 1], '-', color='#4daf4a', lw=2.5)
        for k in range(3): axs[2].plot(*tri[k], 'o', color='#4daf4a', mec='k', ms=11, zorder=8); axs[2].annotate('λ=%.2f\nW=%.2f' % (lam[k], Wd[J[k]]), tri[k], textcoords='offset points', xytext=(8, 8), fontsize=11)
        axs[2].text(.03, .97, 'треугольник Делоне вокруг конца:\nV(конец) = Σ λ·W = %.2f' % float((lam * Wd[J]).sum()), transform=axs[2].transAxes, va='top', fontsize=12, bbox=dict(fc='w', ec='k'))
        axs[2].text(.03, .03, 'без KV: плавная интерполяция\nмежду тремя соседями', transform=axs[2].transAxes, fontsize=11)
    fig.suptitle('Как узел оценивает «куда приведёт дуга»: W(узел) = min по u [ τ + V(конец дуги) ]', fontsize=15, y=.97); frames.append(fig_to_img(fig))
for st in (0, 1, 2, 3, 3, 4, 4, 5, 5): draw(st)
frames[0].save('/tmp/c11_s2_first.png'); frames[-1].save('/tmp/c11_s2_last.png'); save_gif(frames, '/tmp/c11_method_node.gif', fps=1, hold_last=3)
