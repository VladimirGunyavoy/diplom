import sys,time,pickle; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('tests/run_faces_pend.py').read().split("S = pend_top()")[0])
S=pend_top(); t0=time.time(); cells=[]
from src.cells7.cover import cover_layer
for k in (0,1):
    c,P,cov=cover_layer(S,k,m=40); print('layer',k,len(c),'cov',cov.mean().round(3),round(time.time()-t0),flush=True); cells+=c
    pickle.dump([(c.k,c.c,c.r,c.tau) for c in cells],open('/tmp/claude-1000/pend_cells_m40.pkl','wb'))
