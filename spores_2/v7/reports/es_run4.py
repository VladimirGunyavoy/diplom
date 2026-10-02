import sys
exec(open('/tmp/claude-1000/p3.py').read().split("V=SporeV(")[0])
gn=goal
V=SporeV(S,cells,gn,hs=.008,ht=.02,dt_edge=.06); V.Vn=np.load('/tmp/claude-1000/pend_V6.npy'); print('V ok',flush=True)
Sp=S
exec(open('/tmp/claude-1000/es_head4.py').read())
class SV:
    def __init__(s,V): s.V=V
    def Vq(s,x,w):
        Q=np.stack([wrap(np.atleast_1d(x))/np.pi,np.atleast_1d(w)/4.0],1); return np.minimum(s.V.value(Q),1e3)
SVo=SV(V)
rng=np.random.default_rng(0); Q=np.stack([rng.uniform(-1,1,60),rng.uniform(-.8,.8,60)],1)[:int(os.environ.get('N',30))]
Ta=ref_T(Q); Qp=np.stack([Q[:,0]*np.pi,Q[:,1]*4.0],1)
CFG=[(3,.001,1.),(3,.01,1.),(3,.001,.95),(3,.01,.95),(1,.001,1.)]
for e,nz,g in CFG:
    EST=e; NOISE=nz; GAIN=g; EPS=.02; Ta_=Ta
    t0=time.time(); T,SW=run(SVo,Qp,'P3',R=.5); ok=np.isfinite(T)&(Ta<500)&(Ta>1); r=T[ok]/Ta[ok]
    print('EST',e,'noise',nz,'GAIN',g,'дошли',np.isfinite(T).mean().round(3),'T/Ta mean',r.mean().round(3),'med',np.median(r).round(3),'max',r.max().round(2),'sw med',np.median(SW),'max',SW.max(),'сек',round(time.time()-t0),flush=True)
