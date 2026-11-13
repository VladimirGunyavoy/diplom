import sys, runpy
sys.argv = ['spores_di.py','di4','12000','1000','3','dbg',sys.argv[1]]
import os; os.environ['NSTART']='1'
g = runpy.run_path('spores_di.py', run_name='lib')
import numpy as np
e=g['e']; S=g['S']; flow=g['flow']; US_=g['US_']
x=e.X0[0].copy(); print('x0',x,'Tref',e.Tref[0])
b=g['corridor_query'](S, flow, x, g['back'], g['tau'], 1000, g['rho'], g['g'], g['miss'], K=3, neigh=0)
print('T',b[0],'seq',b[1],'dts',b[2])
y=x.copy()
for s,dt in zip(b[1],b[2]): y=flow(y,s,dt); print(s,US_[s],dt,y)
print('goal?',g['ing'](y), 'verify', g['common'].verify(e,x,[US_[s] for s in b[1]],list(b[2])))
print('env step check', e.step(x[None].copy(), US_[b[1][0]][None], b[2][0])[0], flow(x,b[1][0],b[2][0]))
