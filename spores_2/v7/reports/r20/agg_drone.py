"""R20 дрон: сводка. python3 agg_drone.py <папка с drone_spores6d_r20s<k>.json> (эталон bench/data/ocp_all100.json)"""
import json,sys,numpy as np,collections,os
o=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../bench/data/ocp_all100.json'))); Tr=np.array([o[str(i)]['T'] for i in range(100)])
R=0;tc=0;rt=[];why=collections.Counter();tq=[];per=[];bad=collections.Counter()
for k in range(20):
    d=json.load(open(os.path.join(sys.argv[1],'drone_spores6d_r20s%d.json'%k))); R+=d['reached']; per.append(d['reached'])
    for x in d['per_start']:
        tq.append(x['t_query']); why[x['why']]+=1
        if x['T_verify'] is not None: rt.append(x['T_verify']/Tr[x['i']]); tc+=x['touch']
        else: bad[x['i']]+=1
rt=np.array(rt); tq=np.array(tq)
print('дошли %d/2000 (сиды %d–%d), касаний %d, T/T_OCP мед %.3f p90 %.3f max %.3f, запрос мед %.2f p90 %.2f max %.1f с, причины %s, чаще всего %s'%(R,min(per),max(per),tc,np.median(rt),np.percentile(rt,90),rt.max(),np.median(tq),np.percentile(tq,90),tq.max(),dict(why),bad.most_common(6)))
