"""lr1 / bend: совпадает ли проверка изгиба (bend в growN) с реальной ошибкой модели клетки.
Код v8 не меняется: growN импортируется из каталога --algo (копия закоммитованной версии), pause подменяется хуком, который
через кадры (sys._getframe) берёт локальные growN: bend(), klo/khi/ilo/ihi, R0, Wt — так bend считается ровно тем же кодом.

Реальная ошибка клетки: точка внутри ядра в координатах клетки (строка ρ — дробный индекс в G, боковая s ∈ [−r, r]);
модельное положение = билинейная интерполяция узлов G (M узлов на строку; как в HexIdx), модельное время τ — та же интерполяция tau;
точное = поток rk4 (шаг ≤ .0125) от точки сечения c + s·e на время τ. Ошибка |точное − модельное|: евклид / ширина ядра (2r) и в метрике Грамиана (как bend:
Wc по формуле bend для строк клетки, sqrt(dᵀ W⁻¹ d)); max и среднее по 200 точкам. Доп. наборы: lat — точки на целых строках (чисто боковая ошибка), tim — на s=0 (чисто по времени).
По каждому стопу направления: bend текущей клетки и «пробное» расширение на один шаг в том же направлении: bend пробного ящика + построенная пробная клетка + её реальная ошибка
(это даёт точки за пределами DELTA, где у итоговых клеток bend ≤ DELTA по построению).

Запуск (Windows-питон с numpy/scipy/tqdm/matplotlib):
  python bend_check.py run    --algo DIR [--cells 200] [--par 6]   # все конфигурации → bend/*.json
  python bend_check.py report                                       # PNG + раздел «bend» в results.md
  python bend_check.py one --sys pend --nf 0 --delta .03 --algo DIR --out F.json    # одна конфигурация (её зовёт run)"""
import sys, os, json, time, argparse, subprocess, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'bend')
SYSS = ('pend', 'di'); NFS = (0, 1); DELTAS = (.01, .03, .1, .3)
PY = sys.executable


# ---------- одна конфигурация ----------
def one(a):
    os.environ.update(SYS=a.sys, NORMFRONT=str(a.nf), DELTA=str(a.delta), MAXC=str(a.cells), COVTOL='0', TQDM_MI='1000', M=os.environ.get('M', '3'), KF=os.environ.get('KF', '21'))   # как в live.py: M=3, KF=21
    sys.path.insert(0, a.algo)
    import growN as g
    rng_s = np.random.default_rng(a.seed + 1); u = g.US[-1]; Cell = g.Cell; HALO = g.HALO
    assert g.N == 2 and g.m_ == 1

    def wi_of(Wt, ilo, ihi):                                                # метрика bend для строк ilo..ihi
        Wc = Wt[ihi] if ihi >= -ilo else Wt[ilo]; T_ = (ihi - ilo) * g.DTN; Wc = Wc * (T_ / max(max(ihi, -ilo) * g.DTN, 1e-9)); lam, Q = np.linalg.eigh(Wc)
        return (Q / np.maximum(lam, 1e-12 * max(lam.max(), 1e-30))) @ Q.T

    def exact(y0, T):
        nsub = max(1, int(np.ceil(np.abs(T).max() / .0125))); return g._rkv(y0.copy(), u, T.astype(float), nsub)

    def cell_err(cd, Wi, K=200):
        G, tau = cd['G'], cd['tau']; r0 = float(cd['r'][0]); c = np.asarray(cd['c'], float); e0 = cd['e'][0]; hb, nb, nf = cd['hb'], cd['nb'], cd['nf']; nt = len(G); width = 2 * r0
        ax = np.linspace(-(1 + HALO) * r0, (1 + HALO) * r0, g.M)
        def ev(rho, s):
            i0 = np.clip(np.floor(rho).astype(int), 0, nt - 2); t = (rho - i0)[:, None]; j = np.clip(np.searchsorted(ax, s) - 1, 0, g.M - 2); b = ((s - ax[j]) / (ax[j + 1] - ax[j]))[:, None]
            bil = lambda X: (1 - t) * (1 - b) * X[i0, j] + t * (1 - b) * X[i0 + 1, j] + (1 - t) * b * X[i0, j + 1] + t * b * X[i0 + 1, j + 1]
            mod = bil(G); T = bil(tau[..., None])[:, 0]; d = exact(c + s[:, None] * e0, T) - mod
            return np.linalg.norm(d, axis=1) / width, np.sqrt(np.maximum(np.einsum('ki,ij,kj->k', d, Wi, d), 0))
        top = hb + nb + nf; S_ = lambda n: rng_s.uniform(-r0, r0, n)
        e1, g1 = ev(rng_s.uniform(hb, top, K), S_(K))
        e2, g2 = ev(hb + rng_s.integers(0, nb + nf + 1, 60).astype(float), S_(60))             # lat: целые строки
        e3, _ = ev(rng_s.uniform(hb, top, 60), np.zeros(60))                                   # tim: s = 0 (узел)
        return dict(eu_max=float(e1.max()), eu_mean=float(e1.mean()), gr_max=float(g1.max()), gr_mean=float(g1.mean()), lat_max=float(e2.max()), lat_gr=float(g2.max()), tim_max=float(e3.max()), width=width, nb=int(nb), nf=int(nf))

    state = dict(sp=None, rej=0, cells=[]); t0 = time.time()
    def hook(name, lvl=2, must=False, **st):
        sp = state['sp']
        if name == 'seed': state['sp'] = dict(seed=[float(x) for x in st['seed']], stops=[], Wt=None, bcur=None)
        elif name == 'section' and sp is not None: sp['Wt'] = sys._getframe(1).f_locals['Wt']
        elif name == 'stop' and sp is not None:
            f = sys._getframe(1)
            while f.f_code.co_name != 'growN': f = f.f_back
            L = f.f_locals; d = st['dir']; bend = L['bend']; klo, khi, ilo, ihi = list(L['klo']), list(L['khi']), L['ilo'], L['ihi']; S = L['S']
            rec = dict(dir=list(d), reason=st['reason'], b_cur=bend(klo, khi, ilo, ihi), b_tr=None, tr=None); sp['bcur'] = rec['b_cur']
            if d[0] == 'a':
                k = d[1]; nk = khi[k] + 1 if d[2] > 0 else klo[k] - 1; ok = 0 <= nk < len(S); tb = (list(klo), list(khi), ilo, ihi)
                if ok: tb[0][k] = min(klo[k], nk); tb[1][k] = max(khi[k], nk)
            else:
                i = ihi + 1 if d[0] == 'F' else ilo - 1; ok = L['imin'] <= i <= L['imax']; tb = (klo, khi, min(ilo, i), max(ihi, i))
            if ok:
                try:
                    rec['b_tr'] = bend(*tb); p, e = L['p'], L['e']
                    c_ = Cell(p + sum(e[k] * (S[tb[0][k]] + S[tb[1][k]]) / 2 for k in range(1)), u, np.array([(S[b] - S[a_]) / 2 for a_, b in zip(tb[0], tb[1])]) / (1 + HALO), e)
                    c_.p = p; c_.nf, c_.nb = tb[3], -tb[2]; c_.build(); cd = g.cell_dict(c_); rec['tr'] = cell_err(cd, wi_of(sp['Wt'], -cd['nb'], cd['nf']))
                except Exception as ex: rec['err'] = repr(ex)
            sp['stops'].append(rec)
        elif name == 'cell' and sp is not None:
            cd = st['cell'](); sp['final'] = cell_err(cd, wi_of(sp['Wt'], -cd['nb'], cd['nf'])); sp['final_bend'] = sp['bcur']; sp.pop('Wt'); sp.pop('bcur')
            state['cells'].append(sp); state['sp'] = None
        elif name == 'reject': state['rej'] += 1; state['sp'] = None
    g.pause = hook
    cells, _ = g.build_layer(u, np.random.default_rng(a.seed))
    res = dict(sys=a.sys, nf=a.nf, delta=a.delta, algo=open(os.path.join(a.algo, 'REV')).read().strip() if os.path.exists(os.path.join(a.algo, 'REV')) else a.algo, zacep=g.ZACEP, kf=g.KF,
               m=g.M, ncells=len(cells), rejected=state['rej'], sec=time.time() - t0, cells=state['cells'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, 'w'))
    print('%s nf=%d delta=%g: %d cells, %d records, %.0f s' % (a.sys, a.nf, a.delta, len(cells), len(state['cells']), time.time() - t0))


# ---------- запуск всех ----------
def run(a):
    from tqdm import tqdm
    jobs = [(s, nf, d) for s in SYSS for nf in NFS for d in DELTAS]; env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8'); procs = []; bar = tqdm(total=len(jobs), desc='bend_check')
    os.makedirs(OUT, exist_ok=True)
    while jobs or procs:
        while jobs and len(procs) < a.par:
            s, nf, d = jobs.pop(0); out = os.path.join(OUT, 'bend_%s_nf%d_d%g.json' % (s, nf, d))
            procs.append(subprocess.Popen([PY, os.path.abspath(__file__), 'one', '--sys', s, '--nf', str(nf), '--delta', str(d), '--cells', str(a.cells), '--algo', a.algo, '--out', out], env=env))
        time.sleep(1.)
        for p in list(procs):
            if p.poll() is not None: procs.remove(p); bar.update(1)
    bar.close()


# ---------- отчёт ----------
def load():
    R = {}
    for s, nf, d in itertools.product(SYSS, NFS, DELTAS):
        fn = os.path.join(OUT, 'bend_%s_nf%d_d%g.json' % (s, nf, d))
        if os.path.exists(fn): R[(s, nf, d)] = json.load(open(fn))
    return R


def report(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    R = load(); THR = (.01, .05); md = []; algo = next(iter(R.values()))['algo']
    pct = lambda x, q: float(np.percentile(x, q)) if len(x) else float('nan')
    md.append('## bend: совпадает ли проверка изгиба с реальной ошибкой клетки (worker-3)\n')
    md.append('Скрипт `bend_check.py` (`run` → `bend/*.json`, `report` → этот раздел и `bend_scatter.png`), код growN — коммит %s (в нём (в): bend по строкам одного времени, ZACEP по умолчанию для di), v8 не менялся; '
              'слой u = US[-1] (pend u=.3, di u=+1), 200 клеток (MAXC), COVTOL=0, сид 0, KF=%d, M=%d (как в live.py). Реальная ошибка — 200 случайных точек ядра, модель = билинейная интерполяция узлов G (+tau), точное = rk4 от точки сечения на модельное время; '
              '«%% ширины» = |Δ|/(2r ядра). bend здесь — в метрике Грамиана, как в growN; пробное расширение = один шаг в том же направлении, где росту сказали «стоп».\n' % (algo, next(iter(R.values()))['kf'], next(iter(R.values()))['m']))
    # ---- таблица по конфигурациям
    md.append('### Итоговые клетки по конфигурациям\n')
    md.append('| SYS | NF | DELTA | клеток | доля клеток со стопом по bend | bend итог, мед / макс | реальная ошибка макс-по-клетке, % ширины: мед / p95 / макс | lat / tim макс p95, % | ошибка ≤1% / ≤5% (доля клеток) |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for (s, nf, d), r in R.items():
        C = r['cells']
        if not C: continue
        eu = np.array([c['final']['eu_max'] for c in C]) * 100; fb = np.array([c['final_bend'] for c in C]); bs = np.mean([any(x['reason'] == 'bend' for x in c['stops']) for c in C])
        lat = np.array([c['final']['lat_max'] for c in C]) * 100; tim = np.array([c['final']['tim_max'] for c in C]) * 100
        md.append('| %s | %d | %g | %d | %.2f | %.4f / %.4f | %.3f / %.3f / %.3f | %.3f / %.3f | %.2f / %.2f |' % (s, nf, d, len(C), bs, np.median(fb), fb.max(), np.median(eu), pct(eu, 95), eu.max(), pct(lat, 95), pct(tim, 95), (eu <= 1).mean(), (eu <= 5).mean()))
    # ---- корреляции + (в)
    md.append('\n### (а) Корреляция bend ↔ реальная ошибка\n')
    md.append('Spearman по всем DELTA вместе. «итог» — итоговые клетки (у них bend ≤ DELTA, диапазон обрезан); «пробные» — пробные расширения на каждом стопе (bend за пределами DELTA тоже есть); ошибка — max по точкам клетки.\n')
    md.append('| SYS | NF | точек итог | ρ(bend, ошибка % ширины) итог | ρ(bend, ошибка Грам) итог | точек пробных | ρ(bend, полная % ширины) пробные | ρ(bend, полная Грам) пробные | ρ(bend, боковая % ширины) пробные | ρ(bend, боковая Грам) пробные | боковая ошибка Грам / bend (пробные с bend>1e-3): мед / p90 |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    pts = {}
    for s, nf in itertools.product(SYSS, NFS):
        fx, fe, fgm, tx, te, tg, tl, tlg = [], [], [], [], [], [], [], []
        for d in DELTAS:
            for c in R.get((s, nf, d), dict(cells=[]))['cells']:
                fx.append(c['final_bend']); fe.append(c['final']['eu_max']); fgm.append(c['final']['gr_max'])
                for x in c['stops']:
                    if x['tr'] is not None and x['b_tr'] is not None: tx.append(x['b_tr']); te.append(x['tr']['eu_max']); tg.append(x['tr']['gr_max']); tl.append(x['tr']['lat_max']); tlg.append(x['tr']['lat_gr'])
        fx, fe, fgm, tx, te, tg, tl, tlg = map(np.array, (fx, fe, fgm, tx, te, tg, tl, tlg)); pts[(s, nf)] = (fx, fe, fgm, tx, te, tg, tl, tlg)
        if len(fx) < 3 or len(tx) < 3: continue
        k_ = tx > 1e-3; rr = tlg[k_] / tx[k_]
        md.append('| %s | %d | %d | %.2f | %.2f | %d | %.2f | %.2f | %.2f | %.2f | %.3f / %.3f (n=%d) |' % (s, nf, len(fx), spearmanr(fx, fe)[0], spearmanr(fx, fgm)[0], len(tx), spearmanr(tx, te)[0], spearmanr(tx, tg)[0], spearmanr(tx, tl)[0], spearmanr(tx, tlg)[0], np.median(rr) if len(rr) else np.nan, pct(rr, 90), len(rr)))
    md.append('\n### (б)(в) При каком DELTA ошибка ≤ 1% / 5% ширины; «зря обрезал» / «пропустил кривую»\n')
    md.append('«Зря обрезал» = среди стопов по bend/tbend доля, где пробное расширение имело реальную ошибку ≤ порога (рост можно было продолжать). '
              '«Пропустил» = среди пробных расширений, где bend ≤ DELTA (проверка пропускает), доля с реальной ошибкой > порога. Столбец «итог>порога» — доля итоговых клеток с реальной ошибкой выше порога.\n')
    md.append('«полная» — ошибка по всей клетке (боковая + по времени); «боковая» — только на целых строках (это то, что в принципе видит bend).\n')
    md.append('| SYS | NF | DELTA | порог | итог > порога (полная / боковая) | стопов по bend | зря обрезал (полная / боковая) | пропусков (bend ≤ DELTA) | пропустил кривую (полная / боковая) | p95 итоговых, %: полная / боковая |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    best = {}
    for (s, nf, d), r in R.items():
        C = r['cells']
        if not C: continue
        eu = np.array([c['final']['eu_max'] for c in C]); la = np.array([c['final']['lat_max'] for c in C]); st_ = [x for c in C for x in c['stops'] if x['tr'] is not None and x['b_tr'] is not None]
        for thr in THR:
            bs = [x for x in st_ if x['reason'] in ('bend', 'tbend')]; ps = [x for x in st_ if x['b_tr'] <= d]
            m_ = lambda L, f: np.mean(L) if L else float('nan')
            fp = m_([x['tr']['eu_max'] <= thr for x in bs], 0); fn = m_([x['tr']['eu_max'] > thr for x in ps], 0); fpl = m_([x['tr']['lat_max'] <= thr for x in bs], 0); fnl = m_([x['tr']['lat_max'] > thr for x in ps], 0)
            md.append('| %s | %d | %g | %g%% | %.2f / %.2f | %d | %.2f / %.2f | %d | %.2f / %.2f | %.3f / %.3f |' % (s, nf, d, 100 * thr, (eu > thr).mean(), (la > thr).mean(), len(bs), fp, fpl, len(ps), fn, fnl, pct(eu * 100, 95), pct(la * 100, 95)))
            if pct(eu, 95) <= thr: best[(s, nf, thr, 'full')] = max(best.get((s, nf, thr, 'full'), 0), d)
            if pct(la, 95) <= thr: best[(s, nf, thr, 'lat')] = max(best.get((s, nf, thr, 'lat'), 0), d)
    md.append('\n**Наибольший из проверенных DELTA, при котором p95 максимальной ошибки итоговых клеток ≤ порога** (— = ни один):\n')
    md.append('| SYS | NF | полная ≤1% | полная ≤5% | боковая ≤1% | боковая ≤5% |'); md.append('|---|---|---|---|---|---|')
    for s, nf in itertools.product(SYSS, NFS): md.append('| %s | %d | %s | %s | %s | %s |' % (s, nf, best.get((s, nf, .01, 'full'), '—'), best.get((s, nf, .05, 'full'), '—'), best.get((s, nf, .01, 'lat'), '—'), best.get((s, nf, .05, 'lat'), '—')))
    # ---- график
    fig, ax = plt.subplots(3, 4, figsize=(20, 13))
    for j, (s, nf) in enumerate(itertools.product(SYSS, NFS)):
        fx, fe, fgm, tx, te, tg, tl, tlg = pts.get((s, nf), [np.zeros(0)] * 8)
        for i, (yf, yt, lab) in enumerate(((fe * 100, te * 100, 'ошибка, % ширины'), (fgm, tg, 'ошибка в метрике Грамиана'), (None, tlg, 'БОКОВАЯ ошибка (целые строки), Грам'))):
            q = ax[i, j]; q.scatter(tx, yt, s=6, c='tab:orange', alpha=.4, label='пробные расширения')
            if yf is not None: q.scatter(fx, yf, s=6, c='tab:blue', alpha=.5, label='итоговые клетки')
            for d in DELTAS: q.axvline(d, c='gray', lw=.5, ls=':')
            if i == 0:
                for t_ in (1, 5): q.axhline(t_, c='r', lw=.6, ls='--')
            q.set_xscale('log'); q.set_yscale('log'); q.set_xlabel('bend'); q.set_ylabel(lab); q.set_title('%s NORMFRONT=%d' % (s, nf)); q.grid(alpha=.3)
            if i >= 1: q.plot([1e-4, 10], [1e-4, 10], 'k-', lw=.5)
        if j == 0: ax[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, 'bend_scatter.png'), dpi=90)
    sec = '\n'.join(md) + '\n\nГрафик: `bend_scatter.png` (верх — полная ошибка % ширины, середина — полная в метрике Грамиана, низ — боковая в метрике Грамиана; серые пунктиры — значения DELTA, красные — 1% и 5%).\n'
    fn = os.path.join(HERE, 'results.md'); t = open(fn, encoding='utf-8').read(); i = t.find('## bend:'); t = (t[:i] if i >= 0 else t.rstrip() + '\n\n') + sec; open(fn, 'w', encoding='utf-8').write(t)
    print('\n'.join(md))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('one', 'run', 'report')); ap.add_argument('--sys', default='pend'); ap.add_argument('--nf', type=int, default=0); ap.add_argument('--delta', type=float, default=.03)
    ap.add_argument('--cells', type=int, default=200); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--algo', default=os.path.join(HERE, '../../../../v8/src/algo')); ap.add_argument('--out', default=''); ap.add_argument('--par', type=int, default=6)
    a = ap.parse_args(); {'one': one, 'run': run, 'report': report}[a.cmd](a)
