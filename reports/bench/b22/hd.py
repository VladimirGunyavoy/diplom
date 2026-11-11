import warnings; warnings.filterwarnings('ignore')
import numpy as np, safety_gymnasium
e=safety_gymnasium.make('SafetyPointCircle2-v0'); e.reset(seed=100000); t=e.unwrapped.task
for n in range(3):
    R=t.agent.mat; print('mat',np.round(R,2).tolist(),'qpos',np.round(t.data.qpos[:3],2))
    for _ in range(15): e.step(np.array([1.,0.]))
    print(' vel dir',np.round(np.arctan2(t.data.qvel[1],t.data.qvel[0]),2),'|v|',np.round(np.hypot(*t.data.qvel[:2]),2),'candidates atan2(m10,m00)=%.2f atan2(m01,m00)=%.2f atan2(m21? ) qpos2=%.2f'%(np.arctan2(R[1,0],R[0,0]),np.arctan2(R[0,1],R[0,0]),t.data.qpos[2]))
    for _ in range(10): e.step(np.array([0.,1.]))
