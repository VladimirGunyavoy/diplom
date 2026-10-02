import sys; sys.path.insert(0,'../v6/src')
import numpy as np
from atlas6.pend import PendAtlas
_A=None
def ref_T(Q):
    global _A
    if _A is None: _A=PendAtlas(n_th=360,n_w=241,wmax=4.0,tau=.05,umax=.3,R_goal=.15,goal_theta=np.pi); _A.solve(iters=3000)
    th=(Q[:,0]*np.pi+2*np.pi)%(2*np.pi)-np.pi; return np.array([_A.value(np.array([a,b*4])) for a,b in zip(th,Q[:,1])])
