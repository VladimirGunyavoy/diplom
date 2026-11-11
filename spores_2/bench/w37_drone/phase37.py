import sys,os,time,importlib.util,json
sys.argv=['x','6000','1600','5','_ph']; sys.path.insert(0,os.path.expanduser('~/spore_v5/a37/spores_2/bench'))
spec=importlib.util.spec_from_file_location('sde',os.path.expanduser('~/spore_v5/a37/spores_2/bench/spores_drone_e.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import src.atlas6.corridor_nd as cn
LOG=[]; _c=cn.candidates; _r=cn.refine
def c2(S,x0,back,tau,NF,*a,**k):
    t=time.perf_counter(); r=_c(S,x0,back,tau,NF,*a,**k); LOG.append(('cand',NF,round(time.perf_counter()-t,2),r[1],len(r[0]))); return r
def r2(*a,**k):
    t=time.perf_counter(); r=_r(*a,**k); LOG.append(('ref',round(time.perf_counter()-t,3),bool(r[3]))); return r
cn.candidates=c2; cn.refine=r2
d=json.load(open(os.path.expanduser('~/spore_v5/a37/spores_2/bench/data/drone_spores6d_R0s%s.json'%os.environ['SD'])))
ids=[x['i'] for x in d['per_start'] if x['why']=='ok' and 2.5<x['t_query']<4][:3]; m.work(ids[0])
for i in ids:
    LOG.clear(); t=time.perf_counter(); r=m.work(i); w=time.perf_counter()-t
    cs=[l for l in LOG if l[0]=='cand']; rs=[l for l in LOG if l[0]=='ref']
    print('start',i,'wall %.2f'%w,'cand',[(l[1],l[2],l[3]) for l in cs],'refines %d (ok %d) refine-time %.2f'%(len(rs),sum(l[2] for l in rs),sum(l[1] for l in rs)),flush=True)
    seq=[];
    for l in LOG: seq.append('C%d'%l[1] if l[0]=='cand' else ('r+' if l[2] else 'r-'))
    print(' '.join(seq)[:300])
