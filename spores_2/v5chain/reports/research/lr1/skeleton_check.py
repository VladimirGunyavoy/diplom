"""lr1 / skeleton: два скелета одной клетки — N (нормальные фронты, у узла своё время tau) и T (на тех же траекториях узлы через равный шаг DTN).
Вопрос: каким скелетом и какой интерполяцией точнее предсказывать (а) положение точки, (б) её ВРЕМЯ вдоль траектории (из этого складывается V).
Истина: точка (s, τ) — номер траектории в сечении клетки и время от сечения; x*(s, τ) = поток rk4 от c + s·e. Модель по узлам скелета: обратная интерполяция x* → (ŝ, τ̂) (Ньютон 2×2 по индексам (строка, столбец), старт от истинных значений);
ошибки |τ̂ − τ| (в долях DTN и в секундах), |ŝ − s| / ширина ядра (2r), и ПРЯМАЯ ошибка положения: по (s, τ) найти модельную строку, где tau_модель(ρ, s) = τ, и сравнить x̂ с x*, / ширина.
Варианты: A) N полилинейно; B) N квадратично поперёк (парабола через 3 узла строки) и по времени (3 соседние строки), tau тем же способом; C) T полилинейно; D) T линейно поперёк + квадратично по времени.
Точки: 200 на клетку, s ∈ [−r, r], τ между tau первой и последней строки ядра на этой траектории (линейно по s). Клетки: слой build_layer u = US[-1], NORMFRONT=1, 200 клеток. Код v8 не меняется, growN — копия из коммита (--algo).
  python skeleton_check.py run --algo DIR [--cells 200]   # pend и di → skeleton/*.json
  python skeleton_check.py report                          # раздел «skeleton» в results.md
  python skeleton_check.py one --sys pend --algo DIR --out F.json"""
import sys, os, json, time, argparse, subprocess, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'skeleton'); PY = sys.executable
VARS = (('A', 'N', 'lin', 'lin'), ('B', 'N', 'quad', 'quad'), ('C', 'T', 'lin', 'lin'), ('D', 'T', 'quad', 'lin'))     # (имя, скелет, по времени, поперёк)


def basis(kind, u, n, ref):
    """веса интерполяции по оси узлов 0..n-1 в точках u (K,): индексы (K,m), веса, производные весов; шаблон узлов выбран по ref (старт Ньютона) и не меняется — карта гладкая"""
    if kind == 'lin' or n < 3:
        i0 = np.clip(np.floor(ref).astype(int), 0, n - 2); a = u - i0
        return np.stack([i0, i0 + 1], 1), np.stack([1 - a, a], 1), np.stack([-np.ones_like(a), np.ones_like(a)], 1)
    j0 = np.clip(np.floor(ref + .5).astype(int) - 1, 0, n - 3); t = u - j0
    return j0[:, None] + np.arange(3), np.stack([(t - 1) * (t - 2) / 2, -t * (t - 2), t * (t - 1) / 2], 1), np.stack([(2 * t - 3) / 2, -(2 * t - 2), (2 * t - 1) / 2], 1)


def evalm(G, tau, kr, kc, rho, kap, rr, kk):
    """положение, время и их производные по (ρ, κ) тензорной интерполяцией; kr — по строкам (время), kc — по столбцам (поперёк)"""
    ir, wr, dwr = basis(kr, rho, G.shape[0], rr); ic, wc, dwc = basis(kc, kap, G.shape[1], kk); I = ir[:, :, None], ic[:, None, :]; Gs, Ts = G[I], tau[I]
    W = wr[:, :, None] * wc[:, None, :]; Wr = dwr[:, :, None] * wc[:, None, :]; Wc = wr[:, :, None] * dwc[:, None, :]
    return (np.einsum('kab,kabn->kn', W, Gs), np.einsum('kab,kabn->kn', Wr, Gs), np.einsum('kab,kabn->kn', Wc, Gs), np.einsum('kab,kab->k', W, Ts), np.einsum('kab,kab->k', Wr, Ts), np.einsum('kab,kab->k', Wc, Ts))


def invert(G, tau, kr, kc, xs, rho, kap):
    """x(ρ, κ) = x* Ньютоном 2×2 от старта (ρ, κ); возвращает ρ, κ, τ̂ и невязку"""
    rho, kap = rho.copy(), kap.copy(); nt, nc = G.shape[:2]; rr, kk = rho.copy(), kap.copy()
    for _ in range(15):
        x, xr, xc, t, tr, tc = evalm(G, tau, kr, kc, rho, kap, rr, kk); J = np.stack([xr, xc], 2); F = x - xs; det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]; det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        d0 = (J[:, 1, 1] * F[:, 0] - J[:, 0, 1] * F[:, 1]) / det; d1 = (-J[:, 1, 0] * F[:, 0] + J[:, 0, 0] * F[:, 1]) / det
        rho = rho - np.clip(d0, -1, 1); kap = kap - np.clip(d1, -1, 1)                  # шаг ≤ 1 узла: без разлёта
    x, xr, xc, t, tr, tc = evalm(G, tau, kr, kc, rho, kap, rr, kk); return rho, kap, t, np.linalg.norm(x - xs, axis=1)


def forward(G, tau, kr, kc, kap, tt, rho):
    """по (κ, τ) найти строку ρ с tau_модель(ρ, κ) = τ (Ньютон по ρ) и вернуть положение x̂"""
    rho = rho.copy(); rr = rho.copy(); kk = np.clip(kap, 0, G.shape[1] - 1)
    for _ in range(15):
        x, xr, xc, t, tr, tc = evalm(G, tau, kr, kc, rho, kap, rr, kk); tr = np.where(np.abs(tr) < 1e-9, 1e-9, tr); rho = rho - np.clip((t - tt) / tr, -1, 1)
    x, xr, xc, t, tr, tc = evalm(G, tau, kr, kc, rho, kap, rr, kk); return x, np.abs(t - tt)


def one(a):
    os.environ.update(SYS=a.sys, NORMFRONT='1', MAXC=str(a.cells), TQDM_MI='1000', M=os.environ.get('M', '3'), KF=os.environ.get('KF', '21'))
    if a.sys == 'pend': os.environ['COVTOL'] = '0'                                                      # pend: фиксированное число клеток
    else: os.environ.update(RMAX=a.rmax, TMAX=a.tmax)                                                   # di: слой с COVTOL кончается на ~20 больших клетках — меньший RMAX/TMAX даёт больше клеток
    sys.path.insert(0, a.algo)
    import growN as g
    assert g.N == 2 and g.M == 3
    u = g.US[-1]; rng = np.random.default_rng(a.seed + 1); H = g.HALO; DT = g.DTN; res = []; t0 = time.time()

    def exact(y0, T):
        nsub = max(1, int(np.ceil(np.abs(T).max() / .0125))); return g._rkv(y0.copy(), u, np.asarray(T, float), nsub)

    def skeleton_T(seg, tmin, tmax):
        k0, k1 = int(np.floor(tmin / DT)), int(np.ceil(tmax / DT)); fw, bw = [seg], []; y = seg
        for _ in range(k1): fw.append(g.step(fw[-1], u))
        for _ in range(-k0): y = g.step(y, u, -1.); bw.append(y)
        G = np.array(bw[::-1] + fw); ks = np.arange(k0, k1 + 1) * DT; return G[-(k1 - k0 + 1):] if len(G) > k1 - k0 + 1 else G, np.repeat(ks[:, None], seg.shape[0], 1)

    def cell_stats(cd, K=200):
        G, tau, c, e0 = cd['G'], cd['tau'], np.asarray(cd['c'], float), cd['e'][0]; r0 = float(cd['r'][0]); hb, nb, nf = cd['hb'], cd['nb'], cd['nf']; width = 2 * r0; ax = np.linspace(-(1 + H) * r0, (1 + H) * r0, g.M)
        s = rng.uniform(-r0, r0, K); lo = np.interp(s, ax, tau[hb]); hi = np.interp(s, ax, tau[hb + nb + nf]); tt = lo + rng.uniform(0, 1, K) * (hi - lo)
        xs = exact(c + s[:, None] * e0, tt); seg = G[hb + nb]                                                    # строка tau=0 — сечение (в N-скелете все tau=0 там)
        skT, tauT = skeleton_T(seg, tau.min(), tau.max()); kap = s / ((1 + H) * r0) + 1; dtmin = float(np.min(np.abs(np.diff(tau[hb:hb + nb + nf + 1], axis=0)) / DT)) if nb + nf > 0 else np.nan
        out = dict(nb=int(nb), nf=int(nf), r0=r0, dtau_min=dtmin, nrowsT=len(skT))
        for name, sk, kt, kx in VARS:
            Gm, Tm = (G, tau) if sk == 'N' else (skT, tauT); nt = len(Gm)
            if sk == 'N': tv = np.stack([np.interp(kap, np.arange(3), Tm[i]) for i in range(nt)], 1); rho0 = np.array([np.interp(tt[q], tv[q], np.arange(nt)) for q in range(K)])
            else: rho0 = (tt - Tm[0, 0]) / DT
            rho, kp, th, resid = invert(Gm, Tm, kt, kx, xs, rho0, kap); xh, rt = forward(Gm, Tm, kt, kx, kap, tt, rho0); mi = resid < 1e-6 * width; mf = rt < 1e-7 * DT
            et = np.abs(th - tt) / DT; es = np.abs((kp - 1) * (1 + H) * r0 - s) / width; ep = np.linalg.norm(xh - xs, axis=1) / width      # статистика — только по сошедшимся точкам; доли несошедшихся — отдельно
            mx = lambda v, m: float(v[m].max()) if m.any() else float('nan'); mn = lambda v, m: float(v[m].mean()) if m.any() else float('nan')
            out[name] = dict(t_max=mx(et, mi), t_mean=mn(et, mi), s_max=mx(es, mi), s_mean=mn(es, mi), p_max=mx(ep, mf), p_mean=mn(ep, mf), bad=float((~mi).mean()), badf=float((~mf).mean()))
        return out

    def hook(name, lvl=2, must=False, **st):
        if name == 'cell': res.append(cell_stats(st['cell']()))
    g.pause = hook
    cells, _ = g.build_layer(u, np.random.default_rng(a.seed))
    rev = open(os.path.join(a.algo, 'REV')).read().strip() if os.path.exists(os.path.join(a.algo, 'REV')) else a.algo
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(dict(sys=a.sys, algo=rev, ncells=len(cells), sec=time.time() - t0, cells=res), open(a.out, 'w'))
    print('%s: %d cells, %.0f s' % (a.sys, len(res), time.time() - t0))


def run(a):
    """конфигурации параллельно (≤ --par процессов): pend — сиды 0..3 по cells/4 клеток; di — четыре пары (RMAX, TMAX), по cells/4 клеток (слой di кончается по покрытию раньше — клеток может быть меньше)"""
    from tqdm import tqdm
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONIOENCODING='utf-8'); os.makedirs(OUT, exist_ok=True); n = a.cells // 4
    jobs = [('pend', 'seed%d' % k, ['--seed', str(k)]) for k in range(4)] + [('di', 'r%s_t%s' % (r, tm), ['--rmax', r, '--tmax', tm]) for r, tm in (('.05', '.6'), ('.08', '1'), ('.12', '1.5'), ('.2', '2.5'))]
    procs = []; bar = tqdm(total=len(jobs), desc='skeleton_check', mininterval=10)
    while jobs or procs:
        while jobs and len(procs) < a.par:
            s, tag, extra = jobs.pop(0); procs.append(subprocess.Popen([PY, os.path.abspath(__file__), 'one', '--sys', s, '--cells', str(n), '--algo', a.algo, '--out', os.path.join(OUT, 'skeleton_%s_%s.json' % (s, tag))] + extra, env=env))
        time.sleep(2)
        for p in list(procs):
            if p.poll() is not None: procs.remove(p); bar.update(1)
    bar.close()


def report(a):
    pct = lambda x, q: float(np.percentile(x, q)) if len(x) else float('nan')
    md = ['\n## skeleton: два скелета клетки — N (нормальные фронты) и T (равные шаги времени) (worker-3)\n']; algo = ''
    md.append('Скрипт `skeleton_check.py`. Клетка = те же траектории (M=3 столбца). Истина: точка (s, τ) → x*(s, τ) потоком rk4; модель восстанавливает (ŝ, τ̂) обратной интерполяцией по узлам скелета; прямая ошибка положения — по (s, τ) находим модельную строку с tau_модель = τ и сравниваем x̂ с x*. '
              'Варианты: **A** N полилинейно (сейчас); **B** N квадратично поперёк (3 узла) и по времени (3 соседние строки), tau так же; **C** T полилинейно; **D** T линейно поперёк + квадратично по времени. '
              'T-скелет: строки на каждом DTN от min до max tau клетки (с гало) на тех же 3 траекториях. 200 точек на клетку, в таблице — максимум по точкам клетки, затем медиана / p95 / макс по клеткам. Статистика по сошедшимся точкам (доли несошедшихся — в заголовке). «Веер» — клетки с min Δτ/DTN < .6 между соседними строками ядра у какого-либо клона. Секунды = доли DTN × 0.1.\n')
    for s in ('pend', 'di'):
        fns = sorted(glob.glob(os.path.join(OUT, 'skeleton_%s_*.json' % s)))
        if not fns: continue
        rs = [json.load(open(fn)) for fn in fns]; C = [c for r in rs for c in r['cells']]; algo = rs[0]['algo']; fan = [c for c in C if c['dtau_min'] < .6]
        md.append('### %s (u = US[-1], NORMFRONT=1): %d клеток, веер %d; строк ядра медиана %d, строк T-скелета медиана %d, доля точек, где Ньютон не сошёлся (обратная / прямая): %s\n' % (
            s, len(C), len(fan), np.median([c['nb'] + c['nf'] for c in C]), np.median([c['nrowsT'] for c in C]), ' '.join('%s %.3f/%.3f' % (v[0], np.mean([c[v[0]]['bad'] for c in C]), np.mean([c[v[0]]['badf'] for c in C])) for v in VARS)))
        md.append('| клетки | вар. | ошибка времени, DTN: мед / p95 / макс | время, с: p95 / макс | траектория |ŝ−s|, % ширины: мед / p95 / макс | положение (прямая), % ширины: мед / p95 / макс |'); md.append('|---|---|---|---|---|---|')
        for grp, G in (('все', C), ('веер', fan)):
            if not G: continue
            for v in VARS:
                f = lambda k, sc=1.: (lambda a_: a_[np.isfinite(a_)])(np.array([c[v[0]][k] for c in G]) * sc)
                t, sx, p = f('t_max'), f('s_max', 100), f('p_max', 100)
                md.append('| %s (%d) | %s | %.3f / %.3f / %.3f | %.4f / %.4f | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f |' % (grp, len(G), v[0], np.median(t), pct(t, 95), t.max(), pct(t, 95) * .1, t.max() * .1, np.median(sx), pct(sx, 95), sx.max(), np.median(p), pct(p, 95), p.max()))
        md.append('')
    fn = os.path.join(HERE, 'results.md'); t = open(fn, encoding='utf-8', newline='').read(); i = t.find('## skeleton:')
    rd = os.path.join(HERE, 'skeleton_conclusions.md'); tail = open(rd, encoding='utf-8').read() if os.path.exists(rd) else ''
    sec = '\n'.join(md) + '\nКод growN — коммит %s.\n' % algo + tail
    t = (t[:i].rstrip() + '\n' if i >= 0 else t.rstrip() + '\n') + sec; open(fn, 'w', encoding='utf-8', newline='\n').write(t.replace('\r\n', '\n')); print('\n'.join(md))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('one', 'run', 'report')); ap.add_argument('--sys', default='pend'); ap.add_argument('--cells', type=int, default=200); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--par', type=int, default=8)
    ap.add_argument('--rmax', default='.3'); ap.add_argument('--tmax', default='3'); ap.add_argument('--algo', default=''); ap.add_argument('--out', default=''); a = ap.parse_args(); {'one': one, 'run': run, 'report': report}[a.cmd](a)
