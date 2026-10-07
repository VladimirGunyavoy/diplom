# research-23 r23/08: гипотеза «V занижается в гиперячейках, накрывающих излом V» — уже клетки поперёк: RMAX .3 (было .6), MAXC 6000, DELTA .1, GS 50, PESS 1; слои STGPU 0, solve под flock; затем vstart (V*/эт) и kink-проверка не нужна
D=~/spore_v5/r23/08_manip_rmax3; mkdir -p $D; cd $D; cp ../05_manip_pess/{growN.py,finish_gen.py,pipe.py,vstart_d1.py} .
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 RMAX=.3 TMAX=3 DELTA=.1 MAXC=6000 SBCAUS=1 SBLAY=0 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=r3c6 STGPU=0 OMP_NUM_THREADS=2 BUILDONLY=1 python3 pipe.py > r3c6_build.log 2>&1
TAG=r3c6 STGPU=0 flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > r3c6.log 2>&1
sed -e "s/A_g50d1p1.pkl/A_r3c6.pkl/" vstart_d1.py > vstart_r3.py; python3 vstart_r3.py > vstart_r3.out 2>&1
