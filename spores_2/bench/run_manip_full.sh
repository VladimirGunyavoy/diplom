#!/bin/bash
# Манипулятор 4D, полный тайминг (aida, ждёт конца PPO): слои CPU (run45c) → solve GPU SONE (run45h). Лог ~/spore_v5/c1/manip_full.log; времена — строки "[TAG] ... build N с" и "solve N с".
while pgrep -f "[p]po_run.py" > /dev/null; do sleep 20; done
W=~/spore_v5/c1; D=$W/manip_run; mkdir -p $D; cd $D; cp ~/spore_v5/wb5/growN.py ~/spore_v5/wb5/finish_gen.py ~/spore_v5/wb5/m45/pipe.py .
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GLIM=0 RMAX=.6 TMAX=3 SBCAUS=1 SBLAY=0 NOLATCH=1 FINGRID=2 VF=1 PYTHONUNBUFFERED=1 PESS=1 GPUMAX=1000000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TAG=g300d3 DELTA=.3 MAXC=3000 GS=300
STGPU=0 OMP_NUM_THREADS=2 BUILDONLY=1 python3 pipe.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a -E "^\[|Error|Trace"
cp L_g300d3.pkl L_g300so.pkl; export VF=2 TAG=g300so SONE=1
STGPU=1 SOLVEGPU=1 flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a -E "^\[|Error|Trace"
echo MANIPFULLDONE
