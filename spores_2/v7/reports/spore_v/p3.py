import sys,time,pickle; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('tests/run_faces_pend.py').read().split("S = pend_top()")[0])
from src.cells7.cell import Cell
from src.cells7.spore_v import SporeV, rk4v, BIG
from atlas6_ref import ref_T
S=pend_top(); cells=[Cell(S,k,c,r,t) for k,c,r,t in pickle.load(open('/tmp/claude-1000/pend_cells_filled.pkl','rb'))]
goal=lambda Y:(np.abs(S.wrap(Y)[:,0])<=.02)&(np.abs(Y[:,1])<=.02)
V=SporeV(S,cells,goal,hs=.02,ht=.02); V.Vn=np.load('/tmp/claude-1000/pend_V.npy')
rng=np.random.default_rng(0); Q=np.stack([rng.uniform(-1,1,60),rng.uniform(-.8,.8,60)],1); Ta=ref_T(Q)
T,sw=V.rollout(Q,dt=.06); v0=V.value(Q); bad=~np.isfinite(T)
print('не дошли',bad.sum(),'из',len(Q),'| V*(старт) BIG у них:',(v0[bad]>=500).sum(),'| Ta у них med',np.median(Ta[bad]).round(2),'| V/Ta у дошедших med',np.median(v0[~bad]/Ta[~bad]).round(3))
# траектория одного недошедшего: где встаёт
i=np.nonzero(bad)[0][0]; y=Q[i:i+1].copy(); print('старт',y,'V',V.value(y),'Ta',Ta[i])
for step in range(2000):
    vs=[];ys=[]
    for k in (0,1): yn=rk4v(S,k,y,np.array([.06]),n=2); ys.append(yn); vs.append(V.value(yn)[0])
    kb=int(np.argmin(vs))
    if vs[kb]>=BIG/2: print('стоп шаг',step,'y',y,'Vcand',np.round(vs,1)); break
    y=ys[kb]
    if step%200==0: print(step,y.round(3),round(vs[kb],2))
