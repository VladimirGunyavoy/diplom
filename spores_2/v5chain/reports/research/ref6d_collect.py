"""Сбор эталонов T 6D из логов ref6d_ocp.py (aida ~/spore_v5/r5/out/*.log) → ref6d_T.json: {набор: {WMP: [T_ref по запросам, None = не найдено]}}.
Запуск: python3 ref6d_collect.py <папка логов> [выход.json]"""
import sys, re, glob, json, os
d = sys.argv[1]; out = {}
for p in sorted(glob.glob(os.path.join(d, '*_W*_q*.log'))):
    name, W, q = re.match(r'(.+)_W([\d.]+)_q(\d+)\.log', os.path.basename(p)).groups()
    m = re.search(r'T_ref (\S+)\s+успешных (\d+)/(\d+)', open(p).read())
    if not m: continue
    L = out.setdefault(name, {}).setdefault('W' + W, {})
    L[int(q)] = dict(T=None if m.group(1) == 'None' else float(m.group(1)), ok=int(m.group(2)), tot=int(m.group(3)))
res = {n: {w: [v.get(q, {}).get('T') for q in range(1, max(v) + 1)] for w, v in ws.items()} for n, ws in out.items()}
json.dump(dict(doc='T_ref 6D (OCP CasADi/IPOPT N40 M4, мультистарт по 8 ветвям 2π; W3 — |w|≤3 вдоль пути, W50 — без предела, как refine коридора). '
               'Запросы = stats_manip6.py без OBST (rng(0), первые 4 = check_corridor_manip6.py). c4g3/c4g0 — 4 запроса G .3 вниз / G 0.',
               T=res), open(sys.argv[2] if len(sys.argv) > 2 else 'ref6d_T.json', 'w'), indent=1)
for n, ws in res.items():
    for w, T in ws.items(): print(n, w, len([t for t in T if t]), '/', len(T), [None if t is None else round(t, 3) for t in T[:8]], '…')
# best: лучшее известное T_ref (min по прогонам; решение W3 допустимо и для W50) — для сравнения с коридором брать best.<набор>.W50
mn = lambda *L: [min([t for t in ts if t is not None], default=None) for ts in zip(*L)]
J = json.load(open(sys.argv[2] if len(sys.argv) > 2 else 'ref6d_T.json')); T = J['T']; B = {}
if 's32down' in T: B['down'] = dict(W50=mn(T['s32down']['W50'], T['s32down']['W3'], T.get('c4g3', {}).get('W50', []) + [None] * 28), W3=T['s32down']['W3'])
if 's32up' in T: B['up'] = dict(W50=mn(T['s32up']['W50'], T['s32up']['W3'], T.get('s32upNS8', {}).get('W50', [None] * 32)), W3=T['s32up']['W3'])
J['best'] = B; J['doc'] += ' best — лучшее известное (вверх: мультистарт не сошёлся на 7/31, неопределённость до ~6%; вниз NS3=NS6).'
json.dump(J, open(sys.argv[2] if len(sys.argv) > 2 else 'ref6d_T.json', 'w'), indent=1)
