exec(open('/tmp/claude-1000/p3.py').read().split("T,sw=V.rollout")[0])
for eps in (0.0,):
    T,sw=V.rollout(Q,dt=.06,eps=eps); ok=np.isfinite(T)&(Ta<500)&(Ta>1); r=T[ok]/Ta[ok]
    print('eps',eps,'дошли',np.isfinite(T).mean().round(3),'T/Ta mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'sw med',np.median(sw),flush=True)
