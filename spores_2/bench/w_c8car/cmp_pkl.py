import pickle, numpy as np, sys
class Dummy:
    def __init__(s,*a,**k): pass
    def __setstate__(s,st): s.__dict__.update(st if isinstance(st,dict) else {'_st':st})
class U(pickle.Unpickler):
    def find_class(s, m, n):
        if m == '__main__' or m.startswith('growCar') or m.startswith('finish_gen'):
            return type(n,(Dummy,),{})
        return super().find_class(m,n)
def load(f): return U(open(f,'rb')).load()
def flat(o,p=''):
    if isinstance(o,np.ndarray): yield p,o
    elif isinstance(o,dict):
        for k,v in o.items(): yield from flat(v,p+'/'+str(k))
    elif isinstance(o,(list,tuple)):
        for i,v in enumerate(o): yield from flat(v,p+'[%d]'%i)
    elif hasattr(o,'__dict__'): yield from flat(o.__dict__,p+'.'+type(o).__name__)
def cmp(a,b,name):
    A=dict(flat(load(a))); B=dict(flat(load(b))); n=0; bad=0; mx=0.; nonid=[]
    for k,v in A.items():
        w=B.get(k)
        if w is None or v.shape!=w.shape: bad+=1; nonid.append(k); continue
        n+=1
        if v.dtype.kind=='f':
            eq=np.array_equal(v,w,equal_nan=True)
            if not eq:
                d=np.abs(np.where(np.isfinite(v)&np.isfinite(w),v-w,0)); mx=max(mx,float(d.max())); nonid.append(k)
        elif not np.array_equal(v,w): nonid.append(k)
    print(name,'массивов',n,'форма/ключ не совпал',bad,'неидентичных',len(nonid),'max|Δ|',mx, nonid[:6])
d='/home/aida/spore_v5/c8/prof/'
cmp(d+'base_L.pkl',d+'nbf_L.pkl','слои'); cmp(d+'base.pkl',d+'nbf.pkl','атлас(V)')
