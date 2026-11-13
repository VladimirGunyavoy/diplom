import json, sys, numpy as np
D = '/home/random/spore_v5/c14/m8/reports/bench/data/m8_spores8d%s.json'
a = json.load(open(D % sys.argv[1])); b = json.load(open(D % sys.argv[2]))
d = []; 
for x, y in zip(a['per_start'], b['per_start']):
    tx, ty = x['T_verify'], y['T_verify']; d.append(None if tx is None and ty is None else 'DIFF' if (tx is None) != (ty is None) else abs(tx - ty))
num = [v for v in d if isinstance(v, float)]
print('maxΔT_verify', max(num) if num else None, 'DIFF', d.count('DIFF'), 'None-оба', d.count(None), 'reach', a['reached'], b['reached'])
for k in ('ratio_med', 'ratio_p90', 't_query_med', 't_query_p90', 't_query_sum', 't_wall_parallel', 't_build'): print(k, a[k], b[k])
