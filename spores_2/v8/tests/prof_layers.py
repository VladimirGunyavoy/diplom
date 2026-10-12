"""Профиль build_layer по трём слоям ДИ (u = +1, -1, 0) — без правок growN.py.
Окружение как у src/stepper/live.py (импорт live ставит те же env), rng(0) на каждый слой, без Stepper (паузы — пустышки).
  python tests/prof_layers.py [повторов=3]   → reports/prof_layers.md (+ reports/prof_layers.json)
1) повторы без профилировщика: время слоя = минимум из N (на нагруженной машине берите больше повторов);
2) один прогон под cProfile: топ-20 по cumtime / tottime, счётчики вызовов, доли блоков (а)–(ж).
Блоки считаются по cumtime функций growN (вложенные def'ы видит cProfile по имени: rowstep, nf_fill, bend, covcheck, build...):
  (а) решётка строк R   — вызовы adv/wstep/jac/rowstep/inbox_g прямо из growN (цикл по строкам ±2rm × nmax)
  (б) NORMFRONT         — nf_fill (внутри NFR = nf1_rows / nf2_rows, Ньютон по клонам)
  (в) bend              — bend()
  (г) зацеп/покрытие    — HexIdx.query/covered/inside/add без covcheck-проб
  (д) посев в непокрытое — covcheck() (COVP проб + idx.inside)
  (е) Cell.build / Gt   — Cell.build
  (ж) паузы stepper     — pause()
"""
import os, sys, time, json, cProfile, pstats, io
HERE = os.path.dirname(os.path.abspath(__file__)); V8 = os.path.abspath(os.path.join(HERE, '..'))
os.chdir(V8); sys.path.insert(0, V8)
import numpy as np
from src.stepper import live                                             # ставит env как в окне (SYS=di, M=3, KF=21, ...)
from src.algo import growN as g

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LAYERS = live.LAYERS


def run_layer(u):
    t = time.perf_counter(); res = g.build_layer(u, np.random.default_rng(0), None)[0]; return time.perf_counter() - t, res


def nodes(res): return int(sum(c.G.size // g.N for c in res))


def plain():
    out = []
    for u in LAYERS:
        ts = []
        for _ in range(REPS): dt, res = run_layer(u); ts.append(dt)
        out.append(dict(u=u, min_s=min(ts), all_s=ts, cells=len(res), nodes=nodes(res)))
        print('PLAIN u=%s min %.2f s (%s) cells %d nodes %d' % (u, min(ts), ', '.join('%.2f' % x for x in ts), len(res), nodes(res)), flush=True)
    return out


def profiled(u):
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); g.build_layer(u, np.random.default_rng(0), None); pr.disable()
    return pstats.Stats(pr), time.perf_counter() - t


def key_of(st, name, file_part='growN'):
    """все ключи (file, line, func) с данным именем функции"""
    return [k for k in st.stats if k[2] == name and file_part in k[0].replace('\\', '/')] if file_part else [k for k in st.stats if k[2] == name]


def cum(st, name, file_part='growN'):
    return sum(st.stats[k][3] for k in key_of(st, name, file_part))


def calls(st, name, file_part='growN'):
    return sum(st.stats[k][1] for k in key_of(st, name, file_part))


def from_caller(st, callee, caller, file_part='growN'):
    """cumtime вызовов callee прямо из caller (по именам)"""
    tot = 0.
    for k in key_of(st, callee, file_part):
        for ck, (cc, nc, tt, ct) in st.stats[k][4].items():
            if ck[2] == caller: tot += ct
    return tot


def method_keys(st, cls_method):
    """методы класса: в cProfile имя функции — просто 'query', 'covered'... — различаем по строкам HexIdx"""
    return [k for k in st.stats if k[2] == cls_method and 'growN' in k[0].replace('\\', '/')]


def table(st, key, n=20):
    rows = sorted(st.stats.items(), key=lambda kv: -kv[1][3 if key == 'cum' else 2])[:n]; out = []
    for (fn, ln, name), (cc, nc, tt, ct, _) in rows:
        out.append('| %s:%d `%s` | %d | %.3f | %.3f |' % (os.path.basename(fn), ln, name, nc, tt, ct))
    return out


def blocks_of(st):
    total = sum(st.stats[k][3] for k in key_of(st, 'build_layer')); c = lambda n: cum(st, n); b = {}
    b['(а) решётка строк R (adv/wstep/jac/rowstep/inbox_g из growN)'] = sum(from_caller(st, n, 'growN') for n in ('adv', 'wstep', 'jac', 'rowstep', 'inbox_g', 'Bq'))
    b['(б) NORMFRONT (nf_fill → NFR/nf1_rows, Ньютон по клонам)'] = c('nf_fill')
    b['(в) bend'] = c('bend')
    b['(г) зацеп/покрытие: HexIdx.covered/inside/add (query внутри них; без covcheck)'] = sum(c(n) for n in ('covered', 'inside', 'add')) - sum(from_caller(st, n, 'covcheck') for n in ('inside', 'covered'))
    b['(д) посев в непокрытое: covcheck (COVP проб + idx.inside)'] = c('covcheck')
    b['(е) Cell.build / Gt'] = c('build')
    b['(ж) паузы stepper (pause)'] = sum(st.stats[k][3] for k in st.stats if k[2] == 'pause')
    b['прочее (build_layer − сумма блоков; может быть < 0 при перекрытии блоков)'] = total - sum(b.values())
    return total, b


def main():
    pl = plain(); sts = []; walls = []
    for u in LAYERS: st, w = profiled(u); sts.append(st); walls.append(w); print('PROF u=%s wall %.1f s' % (u, w), flush=True)
    per = [blocks_of(st) for st in sts]; total_prof = sum(t for t, _ in per)
    blocks = {k: sum(b[k] for _, b in per) for k in per[0][1]}
    st = sts[0]
    for o in sts[1:]: st.add(o)                                                       # суммарная статистика трёх слоёв
    cnt = {n: calls(st, n) for n in ('f', 'rk4', 'adv', 'jac', 'wstep', 'basis', 'query', 'covered', 'inside', 'add', '_test', '_prep', 'nf1_rows', 'nf2_rows', 'nf_fill', 'bend', 'growN', 'covcheck', 'build', 'rand_seed')}
    cnt['np.linalg.solve (Ньютон/МНК)'] = sum(st.stats[k][1] for k in st.stats if k[2] in ('solve',) and 'linalg' in k[0].replace('\\', '/'))
    cnt['np.linalg.eigh'] = sum(st.stats[k][1] for k in st.stats if k[2] in ('eigh',) and 'linalg' in k[0].replace('\\', '/'))
    cnt['np.linalg.svd'] = sum(st.stats[k][1] for k in st.stats if k[2] in ('svd',) and 'linalg' in k[0].replace('\\', '/'))
    cnt['np.stack (f для di и др.)'] = sum(st.stats[k][1] for k in st.stats if k[2] == 'stack')
    L = ['# prof_layers — профиль build_layer по слоям ДИ (u = %s)' % ', '.join('%+g' % u for u in LAYERS), '',
         'Скрипт `tests/prof_layers.py` (перегенерирует этот файл; ручные выводы и предложения — `reports/prof_layers_notes.md`). Окружение как у `live.py`, rng(0) на каждый слой (в окне rng(k) — клетки чуть другие), паузы stepper — пустышки.',
         'Машина: %s, повторов %d (минимум). cProfile раздувает время на мелких вызовах, поэтому доли — по cumtime внутри профилированных прогонов, а секунды — из чистых прогонов. Разброс чистых прогонов велик, если машина занята (другие воркеры/окно).' % (os.environ.get('COMPUTERNAME', 'n/a'), REPS), '',
         '## Время по слоям (без профилировщика, минимум)', '', '| слой u | мин, с | все прогоны, с | клеток | узлов |', '|---|---|---|---|---|']
    for r in pl: L.append('| %+g | %.2f | %s | %d | %d |' % (r['u'], r['min_s'], ', '.join('%.2f' % x for x in r['all_s']), r['cells'], r['nodes']))
    L += ['| **сумма** | **%.2f** | | %d | %d |' % (sum(r['min_s'] for r in pl), sum(r['cells'] for r in pl), sum(r['nodes'] for r in pl)), '',
          '## Доли блоков (cumtime в профилированных прогонах; build_layer всего %.1f с)' % total_prof, '',
          '| блок | ' + ' | '.join('u=%+g, с (%%)' % u for u in LAYERS) + ' | три слоя, с | доля |', '|---|' + '---|' * len(LAYERS) + '---|---|']
    for k in blocks: L.append('| %s | ' % k + ' | '.join('%.2f (%.0f%%)' % (b[k], 100 * b[k] / max(t, 1e-9)) for t, b in per) + ' | %.2f | %.0f%% |' % (blocks[k], 100 * blocks[k] / total_prof))
    L += ['| **build_layer** | ' + ' | '.join('**%.1f**' % t for t, _ in per) + ' | **%.1f** | 100%% |' % total_prof, '',
          '## Число вызовов (три слоя вместе)', '', '| функция | вызовов |', '|---|---|'] + ['| `%s` | %d |' % (k, v) for k, v in cnt.items()] + ['',
          '## Топ-20 по cumtime (три слоя)', '', '| функция | вызовов | tottime, с | cumtime, с |', '|---|---|---|---|'] + table(st, 'cum') + ['',
          '## Топ-20 по tottime (три слоя)', '', '| функция | вызовов | tottime, с | cumtime, с |', '|---|---|---|---|'] + table(st, 'tot')
    os.makedirs(os.path.join(V8, 'reports'), exist_ok=True)
    open(os.path.join(V8, 'reports', 'prof_layers.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump(dict(plain=pl, blocks=blocks, per_layer=[dict(u=u, total=t, blocks=b) for u, (t, b) in zip(LAYERS, per)], total_prof=total_prof, counts=cnt), open(os.path.join(V8, 'reports', 'prof_layers.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('written reports/prof_layers.md', flush=True)


if __name__ == '__main__':
    main()
