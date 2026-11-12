import numpy as np, io, os, pickle
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from core import *
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 13, 'font.family': 'DejaVu Sans'})
VMAX = 14.; CMAP = plt.get_cmap('GnBu', 28); COL = ['#e6194b', '#f58231', '#6a00c8']  # u=-.3, 0, +.3
UNAME = ['u = −0.3', 'u = 0', 'u = +0.3']
def get_S(path='/tmp/c11_S.pkl'):
    if os.path.exists(path): return pickle.load(open(path, 'rb'))
    S = solve_all(); pickle.dump(S, open(path, 'wb')); return S
def grid_V(S, mode, nx=300, ny=170):
    gx = np.linspace(-PI, PI, nx); gy = np.linspace(-OMAX, OMAX, ny); X, Y = np.meshgrid(gx, gy); Q = np.c_[X.ravel(), Y.ravel()]
    V = np.concatenate([Pol(S, mode).vh(Q[i:i + 20000]) for i in range(0, len(Q), 20000)]).reshape(X.shape); return gx, gy, np.where(V > 20, np.nan, V)
def base_ax(ax, title=None):
    ax.set_xlim(-PI, PI); ax.set_ylim(-OMAX, OMAX); ax.set_xlabel('угол θ  (0 = верх, ±π = низ)'); ax.set_ylabel('скорость ω')
    ax.set_xticks([-PI, -PI / 2, 0, PI / 2, PI]); ax.set_xticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
    if title: ax.set_title(title, loc='left')
    ax.plot([0], [0], marker='*', color='gold', ms=20, mec='k', zorder=9)
def heat(ax, gx, gy, V, alpha=1.):
    return ax.pcolormesh(gx, gy, V, cmap=CMAP, vmin=0, vmax=VMAX, shading='auto', alpha=alpha, rasterized=True)
def plot_path(ax, tr, **kw):
    tr = wrap(np.array(tr))
    if len(tr) < 2: return
    segs = [[tr[0]]]
    for a, b in zip(tr[:-1], tr[1:]):
        if abs(b[0] - a[0]) > PI: segs.append([b])
        else: segs[-1].append(b)
    for sg in segs:
        if len(sg) > 1: ax.plot(*np.array(sg).T, **kw)
def plot_arc(ax, x, s, **kw):
    a = np.array(arc(x, s)); a[:, 0] += wrap(np.array(x))[0] - x[0]
    for sh in (-2 * PI, 0, 2 * PI): ax.plot(a[:, 0] + sh, a[:, 1], **kw)
def fig_to_img(fig, ncol=256):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0); im = Image.open(buf).convert('RGB'); return im
def save_gif(frames, path, fps=6, hold_last=10, ncol=256):
    q = [f.quantize(colors=ncol, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    dur = [int(1000 / fps)] * len(q); dur[-1] = int(1000 / fps * hold_last)
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=True, disposal=1)
    print(path, os.path.getsize(path) / 1e6, 'MB', len(q), 'frames')
