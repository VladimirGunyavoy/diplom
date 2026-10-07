# research-23 r23/07: GOALX=1 — узлы цели не участвуют в интерполяции V (solve GPU и vstar); те же слои g50d1p1 (GS 50, DELTA .1, 4×3000), PESS 1; гипотеза: занижение V у границы цели (vtrace: V*+t растёт у цели) — от размазывания V = 0 цели
cd ~/spore_v5/r23/07_goalx
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 RMAX=.6 TMAX=3 DELTA=.1 MAXC=3000 SBCAUS=1 SBLAY=0 SOLVEGPU=1 STGPU=0 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 GOALX=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=g50gx flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > g50gx.log 2>&1
sed -e "s/A_g50d1p1.pkl/A_g50gx.pkl/" vstart_d1.py > vstart_gx.py; python3 vstart_gx.py > vstart_gx.out 2>&1
