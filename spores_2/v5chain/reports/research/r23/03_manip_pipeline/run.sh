D=~/spore_v5/r23/03_manip_pipeline; mkdir -p $D; cd $D; cp ~/spore_v5/r23/02_manip_cover/growN.py ~/spore_v5/r23/02_manip_cover/finish_gen.py .
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=0 GLIM=0 RMAX=.6 TMAX=3 SBCAUS=1 SBLAY=0 STGPU=1 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
go() { export TAG=$1 DELTA=$2 MAXC=$3; OMP_NUM_THREADS=2 BUILDONLY=1 python3 pipe.py > $1_build.log 2>&1; ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > $1.log 2>&1; }
setsid nohup bash -c "$(declare -f go); go d3c3000 .3 3000" < /dev/null > /dev/null 2>&1 &
