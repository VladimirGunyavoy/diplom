# research-23: те же слои g50d1p1 (GS 50, DELTA .1, 4×3000) + SMEAN=1 (среднее по стенсилам группы в solve, research-22 п.43) — гипотеза «V < T* от min по перекрывающимся стенсилам в solve»
cd ~/spore_v5/r23/05_manip_pess; cp L_g50d1p1.pkl L_g50d1sm.pkl
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 RMAX=.6 TMAX=3 DELTA=.1 MAXC=3000 SBCAUS=1 SBLAY=0 SOLVEGPU=1 STGPU=0 SMEAN=1 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=g50d1sm flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > g50d1sm.log 2>&1
sed -e "s/A_g50d1p1.pkl/A_g50d1sm.pkl/" vstart_d1.py > vstart_sm.py; python3 vstart_sm.py > vstart_sm.out 2>&1
