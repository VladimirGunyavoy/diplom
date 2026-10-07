# research-23: solve manip на слоях b6 L_g50d3 (GS 50, GLIM 0, DELTA .3, 4×3000) с PESS=1 WTHR .5 — проверка гипотезы «V не растекается из-за PESS −1»; стенсилы CPU, solve под flock
D=~/spore_v5/r23/05_manip_pess; mkdir -p $D; cd $D; cp ~/spore_v5/r23/03_manip_pipeline/{growN.py,finish_gen.py,pipe.py} .; cp ~/spore_v5/wb5/m45/L_g50d3.pkl L_g50p1.pkl
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 RMAX=.6 TMAX=3 DELTA=.3 MAXC=3000 SBCAUS=1 SBLAY=0 STGPU=0 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=g50p1 flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > g50p1.log 2>&1
