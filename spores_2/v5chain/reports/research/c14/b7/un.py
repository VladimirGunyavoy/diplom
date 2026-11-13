import json, numpy as np
D='/home/random/spore_v5/c14/m8/reports/bench/data/m8_spores8d%s.json'
a=json.load(open(D%'c14ctl'))['per_start']; b=json.load(open(D%'c14loc'))['per_start']
ra=[x['T_verify'] is not None for x in a]; rb=[x['T_verify'] is not None for x in b]
print('base',sum(ra),'loc',sum(rb),'union',sum(x or y for x,y in zip(ra,rb)),'loc спасает',[i for i,(x,y) in enumerate(zip(ra,rb)) if y and not x],'base-only',sum(x and not y for x,y in zip(ra,rb)))
both=[(x['T_verify'],y['T_verify']) for x,y in zip(a,b) if x['T_verify'] is not None and y['T_verify'] is not None]; print('оба дошли',len(both),'T_loc/T_base мед',np.median([q/p for p,q in both]))
fails=[i for i,x in enumerate(ra) if not x]; tl=[b[i]['t_query'] for i in fails]; print('base-провалы',fails,'время (б) на них',[round(t) for t in tl],'loc_ok',[b[i]['info'].get('local_ok') for i in fails])
print('local_ok доля',np.mean([x['info'].get('local_ok',0)>0 for x in b]),'pairs мед',np.median([x['info']['pairs'] for x in b]))
tb=[x['t_query'] for x in a]; print('время base провалов',[round(tb[i]) for i in fails])
