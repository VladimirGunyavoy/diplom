import sys; sys.argv=['x','one']; 
exec(open('canal.py').read().split("if __name__")[0])
S,B=succ_all(XS,VS,ACT); V,M=solve_(S,B,1.0,1.2,.1,.2,.2,XS,VS,400); print('M frac',M.mean(),'V<1e2',(V<1e2).mean())
env=mkenv(); t=env.unwrapped.task; env.reset(seed=100000)
for step in range(120):
    d=t.data; p=t.agent.pos; R_=t.agent.mat; x,y,th=p[0],p[1],np.arctan2(R_[1,0],R_[0,0]); v=d.qvel[0]*np.cos(d.qpos[2])+d.qvel[1]*np.sin(d.qpos[2])
    a=policy(V,x,y,th,v,.3); _,r,c,te,tr,_=env.step(ACT[a]); p=t.agent.pos
    if step%6==0 or max(abs(p[0]),abs(p[1]))>WALL: print(step,'pos',np.round([x,y],2),'th %.2f v %.2f'%(th,v),'a',ACT[a],'V %.2f'%interp(V,x,y,th,v),'vel',np.round(d.qvel[:3],2),'r %.3f'%r)
    if max(abs(p[0]),abs(p[1]))>WALL: break
