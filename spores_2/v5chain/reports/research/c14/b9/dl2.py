import json, sys
D='/home/random/spore_v5/c14/m8/reports/bench/data/m8_spores8d%s.json'
A=json.load(open(D%sys.argv[1])); B=json.load(open(D%sys.argv[2])); Tr=A['Tref']
for i,(x,y) in enumerate(zip(A['per_start'],B['per_start'])):
    tx,ty=x['T_verify'],y['T_verify']
    if (tx is None)!=(ty is None) or (tx is not None and abs(tx-ty)>1e-9): print(i,'%.4f -> %s'%(tx,'None' if ty is None else '%.4f'%ty),'(%+.1f%%)'%(100*(ty-tx)/tx) if ty and tx else '')
