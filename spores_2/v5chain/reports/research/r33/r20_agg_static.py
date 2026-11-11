"""R20×100 статик кар: сводка по сидам (research-33). Запуск на aida 1: python3 r20_agg_static.py ~/spore_v5/a36/r20car
Читает env_A_static_r20s<k>.json (res[i].ok/T/touch/ms, Tref), s<k>_p1.log/p2.log (узлы, build, solve). Печатает строки md и итог по пулу."""
import json, os, re, sys, glob
import numpy as np
D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/spore_v5/a36/r20car')
rows, R, MS, TCH, OK, N = [], [], [], 0, 0, 0
for f in sorted(glob.glob(D + '/env_A_static_r20s*.json'), key=lambda s: int(re.findall(r's(\d+)\.json', s)[0])):
    k = int(re.findall(r's(\d+)\.json', f)[0]); d = json.load(open(f)); res, Tr = d['res'], np.array(d['Tref'], float)
    ok = [i for i, r in enumerate(res) if r['ok'] and not r['touch']]; tch = sum(bool(r['touch']) for r in res)
    rat = np.array([res[i]['T'] / Tr[i] for i in ok if np.isfinite(Tr[i])]); ms = np.array([r['ms'] for r in res])
    p1 = open(f'{D}/s{k}_p1.log', errors='ignore').read() if os.path.exists(f'{D}/s{k}_p1.log') else ''
    p2 = open(f'{D}/s{k}_p2.log', errors='ignore').read() if os.path.exists(f'{D}/s{k}_p2.log') else ''
    m1 = re.findall(r'построено \[([\d, ]+)\] узлов (\d+) (\d+) с', p1); m2 = re.findall(r't_solve ([\d.]+)', p2)
    cells = sum(int(x) for x in m1[-1][0].split(',')) if m1 else -1; nodes = int(m1[-1][1]) if m1 else -1
    tb = float(m1[-1][2]) if m1 else float('nan'); ts = float(m2[-1]) if m2 else float('nan')
    rows.append(f'| {k} | {len(ok)}/{len(res)} | {tch} | {np.median(rat):.3f} | {np.percentile(rat, 90):.3f} | {rat.max():.3f} | {cells} | {nodes/1e6:.2f}M | {tb:.0f} | {ts:.0f} | {np.median(ms):.0f} |')
    R += list(rat); MS += list(ms); TCH += tch; OK += len(ok); N += len(res)
print('| сид | дошли | касаний | мед | p90 | max | клеток | узлов | build с | solve с | мс/шаг |\n|---|---|---|---|---|---|---|---|---|---|---|')
print('\n'.join(rows))
R = np.array(R)
if len(R):
    print(f'\nПУЛ: сидов {len(rows)}, дошли {OK}/{N}, касаний {TCH}, T/T_ref мед {np.median(R):.3f} p90 {np.percentile(R, 90):.3f} max {R.max():.3f}, мс/шаг мед {np.median(MS):.0f}')
