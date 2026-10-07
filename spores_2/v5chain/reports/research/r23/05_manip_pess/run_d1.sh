# research-23: слои b6 L_d1c3000 (GS 0? — проверить узлы цели) нельзя: GS 0. Берём DELTA .1 c GS 50: строим слои сами (STGPU 0), solve PESS 1 под flock; затем vstart (V*/эт) — гипотеза «V занижена интерполяцией на крупных клетках»
cd ~/spore_v5/r23/05_manip_pess
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 RMAX=.6 TMAX=3 DELTA=.1 MAXC=3000 SBCAUS=1 SBLAY=0 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=g50d1p1 STGPU=0 OMP_NUM_THREADS=2 BUILDONLY=1 python3 pipe.py > g50d1p1_build.log 2>&1
TAG=g50d1p1 STGPU=0 flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > g50d1p1.log 2>&1
