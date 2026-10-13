"""c7: total(N)=заготовка+N·запрос по строкам table_<env>.md (для таблиц, где make_report не знает ручные строки). python plot_total_table.py di4 → reports/bench/total_di4.png"""
import sys, re, os
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
env = sys.argv[1]; D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports', 'bench')
rows = []
for l in open(f'{D}/table_{env}.md'):
    c = [x.strip() for x in l.strip().strip('|').split('|')]
    if len(c) < 8 or c[0].startswith(('метод', '---')): continue
    b = re.match(r'[\d.e+]+', c[2]); q = re.match(r'[\d.e+]+', c[3])
    if not (b and q): continue
    nm = re.sub(r'\*\*|\(.*', '', c[0]).strip()[:48] + ' [' + c[6].split('(')[0].strip() + ', ' + c[7].split('/')[0].strip() + ']'
    rows.append((nm, float(b.group()), float(q.group()), 'споры' in c[0]))
N = np.logspace(0, 5, 200); fig, ax = plt.subplots(figsize=(10, 6.5))
for nm, b, q, sp in rows: ax.loglog(N, np.maximum(b + N * q, 1e-3), lw=3 if sp else 1.5, label=nm)
ax.set_xlabel('число запросов N'); ax.set_ylabel('полное время, с = заготовка + N·запрос'); ax.set_title(f'{env} — total(N) (по table_{env}.md; спор b12 — заготовка 215 с)'); ax.grid(True, which='both', alpha=.3); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{D}/total_{env}.png', dpi=100); print(len(rows), 'строк')
