import json
D='/home/random/spore_v5/c14/m8/reports/bench/data/m8_spores8d%s.json'
a=json.load(open(D%'c14fast'))['per_start']; b=json.load(open(D%'c14es'))['per_start']
for i,(x,y) in enumerate(zip(a,b)):
    tx,ty=x['T_verify'],y['T_verify']
    if (tx is None)!=(ty is None) or (tx is not None and tx!=ty): print(i,tx,ty,'Tref',json.load(open(D%'c14fast'))['Tref'][i],'tq',round(x['t_query']),round(y['t_query']))
