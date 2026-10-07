"""research-22, эксп. 20: сколько клеток НА САМОМ ДЕЛЕ нужно для покрытия слоя — жадный отбор (set cover) из клеток Lfull по 300k пробным точкам поля ±2.5.
Если те же .90–.95 покрытия дают ~1500 клеток вместо 6700 — посев можно чинить отбором/порядком, а не новой геометрией. Выход: слои-подмножества для конвейера эксп. 18."""
import sys, os, time, pickle, heapq, numpy as np
sys.path.insert(0, '.')
import growN as G, sgpu
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
from tqdm import tqdm
G.HexIdx._test = sgpu.test_gpu; d_ = pickle.load(open(os.environ['LAYERS'], 'rb')); TG = [float(x) for x in os.environ.get('TARGETS', '.90,.99').split(',')]; out = {t: [] for t in TG}; NP = 300000
for li, L in enumerate(d_['layers']):
    rng = np.random.default_rng(10 + li); Y = rng.uniform(-2.5, 2.5, (NP, 4)); idx = G.HexIdx()
    for c in tqdm(L, desc='индекс слоя %d' % li, mininterval=10): idx.add(c)
    pi, hid, sc = idx.query(Y); R = np.array(idx.R); cid = idx.I[hid, 0]; J = idx.I[hid, 2:]; w = 2 * (1 + G.HALO) / (G.M - 1); Rc = R[cid] if R.ndim > 1 else R[cid][:, None]
    sl = -(1 + G.HALO) * Rc + (J + sc[:, 1:]) * w * Rc; core = (np.abs(sl) <= Rc + 1e-9).all(1); pi, cid = pi[core], cid[core]; full = len(np.unique(pi)) / NP
    o = np.argsort(cid, kind='stable'); cs = cid[o]; ps = pi[o]; st = np.searchsorted(cs, np.arange(len(L) + 1)); pts = [ps[st[k]:st[k + 1]] for k in range(len(L))]; cov = np.zeros(NP, bool); heap = [(-len(p), k) for k, p in enumerate(pts)]; heapq.heapify(heap); sel = []; done = set(TG); curve = []
    while heap and done:
        g, k = heapq.heappop(heap); gain = int((~cov[pts[k]]).sum())
        if heap and gain < -heap[0][0]: heapq.heappush(heap, (-gain, k)); continue
        if gain == 0: break
        cov[pts[k]] = True; sel.append(k); c_ = cov.mean()
        if len(sel) in (100, 200, 400, 800, 1200, 1600, 2400): curve.append('%d: %.3f' % (len(sel), c_))
        for t in sorted(done):
            if c_ >= t * (full if t > .95 else 1.): out[t].append([L[i] for i in sorted(sel)]); done.discard(t); curve.append('цель %.2f → %d клеток (покрытие %.3f)' % (t, len(sel), c_))
    for t in done: out[t].append([L[i] for i in sorted(sel)]); curve.append('цель %.2f не достигнута: %d клеток, %.3f' % (t, len(sel), cov.mean()))
    print('слой %d: всего клеток %d, покрытие ядром всех %.3f | жадный отбор — клеток: покрытие → %s' % (li, len(L), full, '; '.join(curve)), flush=True)
for t in TG:
    pickle.dump(dict(layers=out[t]), open('Lgreedy_%s.pkl' % str(t).replace('.', ''), 'wb')); print('сохранено Lgreedy_%s.pkl: клеток по слоям %s' % (str(t).replace('.', ''), [len(l) for l in out[t]]), flush=True)
