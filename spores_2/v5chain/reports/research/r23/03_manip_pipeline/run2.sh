# research-23, протокол диспетчера 2026-10-07: стенсилы на CPU (STGPU=0), GPU-solve под замком flock /tmp/gpu.lock
cd ~/spore_v5/r23/03_manip_pipeline
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=0 GLIM=0 RMAX=.6 TMAX=3 SBCAUS=1 SBLAY=0 STGPU=0 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python
TAG=d3c3000 DELTA=.3 MAXC=3000 flock /tmp/gpu.lock $PY pipe.py > d3c3000.log 2>&1
until grep -aq "\[d3c6000\] слои" d3c6000_build.log; do sleep 30; done
TAG=d3c6000 DELTA=.3 MAXC=6000 flock /tmp/gpu.lock $PY pipe.py > d3c6000.log 2>&1
