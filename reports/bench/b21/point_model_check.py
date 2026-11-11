import warnings; warnings.filterwarnings("ignore")
import safety_gymnasium, numpy as np, mujoco
e=safety_gymnasium.make("SafetyPointCircle2-v0"); e.reset(seed=1); t=e.unwrapped.task; m=t.model; d=t.data
M0=np.zeros((3,3)); mujoco.mj_fullM(m,d,M0)
mass=M0[0,0]; a=M0[1,2]/mass; Icm=M0[2,2]-mass*a*a; D=m.dof_damping[:3].copy(); h=m.opt.timestep
print("m",mass,"a",a,"Icm",Icm,"D",D,"h",h)
def Mq(q):
    th=q[2]; return np.array([[mass,0,-mass*a*np.sin(th)],[0,mass,mass*a*np.cos(th)],[-mass*a*np.sin(th),mass*a*np.cos(th),Icm+mass*a*a]])
def step_model(q,qd,ctrl,sub=10):
    for _ in range(sub):
        th=q[2]; M=Mq(q)
        F0=np.clip(ctrl[0],-.05,.05); F1=np.clip(ctrl[1]-.3*qd[2],-.05,.05)
        tau=np.array([.3*F0*np.cos(th),.3*F0*np.sin(th),.3*F1])
        bias=np.array([-mass*a*np.cos(th)*qd[2]**2,-mass*a*np.sin(th)*qd[2]**2,0.])
        f=tau-bias-D*qd
        qdd=np.linalg.solve(M+h*np.diag(D),f)
        qd=qd+h*qdd; q=q+h*qd
    return q,qd
def run_mj(U):
    e.reset(seed=1); R0=t.agent.mat.copy(); th0=np.arctan2(R0[1,0],R0[0,0]); out=[]
    # state in generalized coords from qpos
    dd=t.data; out.append((dd.qpos[:3].copy(),dd.qvel[:3].copy()))
    for u in U:
        e.step(np.array(u,float)); dd=t.data; out.append((dd.qpos[:3].copy(),dd.qvel[:3].copy()))
    return out
def pos_world(q,q0w):  # world pos of body origin: hinge outer, origin at R(theta)(u,v)+hinge
    return q
rng=np.random.default_rng(5); E=[]; ET=[]; L=[]
for i in range(60):
    hold=[4,20,10][i%3]; n=100//hold; U=np.repeat(rng.uniform(-1,1,(n,2)),hold,0)
    if i%5==0: U=np.sign(U)
    ref=run_mj(U); q,qd=ref[0][0].copy(),ref[0][1].copy()
    for u in U: q,qd=step_model(q,qd,u)
    qm=ref[-1][0]; E.append(np.linalg.norm(q[:2]-qm[:2])); ET.append(abs(q[2]-qm[2])); L.append(max(np.linalg.norm(np.array([r[0][:2] for r in ref])-ref[0][0][:2],axis=1)))
E=np.array(E); print("60 послед. по 2 с: ошибка положения мед %.4f p90 %.4f max %.4f м; курс max %.3f рад; путь мед %.2f м"%(np.median(E),np.percentile(E,90),E.max(),max(ET),np.median(L)))
