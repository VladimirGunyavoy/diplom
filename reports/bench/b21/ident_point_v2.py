import warnings; warnings.filterwarnings("ignore")
import safety_gymnasium, numpy as np
from scipy.optimize import least_squares
e=safety_gymnasium.make("SafetyPointCircle2-v0"); e.reset(seed=0); t=e.unwrapped.task
agent=t.agent; mdl=t.model; d=t.data
wrapped=lambda a:(a+np.pi)%(2*np.pi)-np.pi
e.reset(seed=1); T0=d.time; e.step(np.zeros(2)); print("env step dt",d.time-T0); print("dt",mdl.opt.timestep,"n_substeps",getattr(t,"action_repeat",None), "gear", mdl.actuator_gear[:,0])
import mujoco
def pose_w():
    R=t.agent.mat if hasattr(t.agent,"mat") else None
    return t.agent.pos[:2].copy(), np.arctan2(R[1,0],R[0,0])
def sim_mj(U):
    e.reset(seed=1); p0,th0=pose_w(); c,s_=np.cos(th0),np.sin(th0); out=[(0,0,0)]; t0=d.time if False else None
    for u in U:
        e.step(np.array(u,float)); p,th=pose_w(); q=p-p0; out.append((c*q[0]+s_*q[1],-s_*q[0]+c*q[1],wrapped(th-th0)))
    return np.array(out)
def sim_model(U,p,dt=.02):
    gv,cv,gw,cw,cy,kc,c2=p; x=y=th=vx=vy=w=0.; out=[(0,0,0)]
    n=4
    for u in U:
        h=dt/n
        for _ in range(n):
            vx+=h*(gv*u[0]-cv*vx-c2*abs(vx)*vx+kc*w*vy); vy+=h*(-cy*vy-kc*w*vx); w+=h*(gw*u[1]-cw*w)
            x+=h*(vx*np.cos(th)-vy*np.sin(th)); y+=h*(vx*np.sin(th)+vy*np.cos(th)); th+=h*w
        out.append((x,y,th))
    return np.array(out)
rng=np.random.default_rng(0)
def mk(n=100,hold=20):
    return np.repeat(rng.uniform(-1,1,(n//hold,2)),hold,0)   # 2 с = 25 шагов по .08
tr=[mk() for _ in range(12)]; te=[mk() for _ in range(12)]+[np.repeat([[1,0]],100,0),np.repeat([[1,1]],100,0),np.repeat([[1,-.5]],100,0),np.repeat([[-1,1]],100,0)]+[np.repeat(rng.uniform(-1,1,(25,2)),4,0) for _ in range(8)]
mj_tr=[sim_mj(U) for U in tr]
def res(p): return np.concatenate([(sim_model(U,p)-M)[:,:2].ravel() for U,M in zip(tr,mj_tr)]+[0.3*wrapped(sim_model(U,p)[:,2]-M[:,2]) for U,M in zip(tr,mj_tr)])
wrapped=lambda a:(a+np.pi)%(2*np.pi)-np.pi
r=least_squares(res,[15,8,20,7,5,1,0],bounds=([0,0,0,0,0,-3,0],[1e3]*7)); p=r.x; print("fit gv cv gw cw cy kc c2",np.round(p,3))
errs=[]
for U in te:
    M=sim_mj(U); Q=sim_model(U,p); errs.append(np.linalg.norm(Q[-1,:2]-M[-1,:2]))
    errs.append(0)  # placeholder
errs=np.array(errs[::2]); print("конечная ошибка положения за 2 с (м): мед %.3f p90 %.3f max %.3f"%(np.median(errs),np.percentile(errs,90),errs.max()))
print("по сценариям:",np.round(errs,3))
print("размер пути (мед, м):",np.median([np.linalg.norm(sim_mj(U)[-1,:2]) for U in te]))
print("sim_mj dist for [1,0] 2s:", np.round(sim_mj(np.repeat([[1,0]],100,0))[-1],3))
