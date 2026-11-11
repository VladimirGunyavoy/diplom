import sys,os; sys.argv=['x','300000','.3','8','sel']; os.environ['FM']='.1'; os.environ['HS']='.5'; os.environ['RB']='.3'
import numpy as np, canal_tree as T, canal as C
S=T.mkS(2); back=T.build_back(S,C.TAU,300000,.3); ag=T.Agent(S,back); env=C.mkenv(); t=env.unwrapped.task
for seed in (200081,):
    env.reset(seed=seed); print('start pos',np.round(t.agent.pos[:2],3),'rho',round(float(np.hypot(*t.agent.pos[:2])),3))
    for step in range(60):
        d=t.data;p=t.agent.pos;R_=t.agent.mat;th=np.arctan2(R_[1,0],R_[0,0]);th0=th-d.qpos[2];c0,s0=np.cos(th0),np.sin(th0)
        vx=c0*d.qvel[0]-s0*d.qvel[1];vy=s0*d.qvel[0]+c0*d.qvel[1]
        Q,RW,CL=T.cands(p[0],p[1],th,vx,vy,C.ACT)
        a=ag.act(p[0],p[1],th,vx,vy)
        if step%6==0: print(step,'pos',np.round(p[:2],2),'v',np.round([vx,vy],2),'a',C.ACT[a],'CLmin %.2f maxCL %.2f nOK %d'%(CL.min(),CL.max(),int(((Q[:,0]<=T.RMAXW)&(CL>=T.FM)).sum())))
        env.step(C.ACT[a])
