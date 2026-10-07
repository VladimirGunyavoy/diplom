# research-23 r23/09 (слово пользователя: ресурсы, 4D быстро): M=2 поперёк (8 узлов в сечении вместо 27) — (а) RMAX .6 MAXC 3000, (б) RMAX .3 MAXC 6000; DELTA .1, GS 50, PESS 1; слои CPU, solve под flock, затем vstart
D=~/spore_v5/r23/09_manip_m2; mkdir -p $D; cd $D; cp ../05_manip_pess/{growN.py,finish_gen.py,pipe.py,vstart_d1.py} .
export SYS=manip KF=11 M=2 FRAC=.5 OVH=.5 MINROWS=3 GS=50 GLIM=0 TMAX=3 DELTA=.1 SBCAUS=1 SBLAY=0 SOLVEGPU=1 NOLATCH=1 FINGRID=2 VF=1 PESS=1 WTHR=.5 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for v in "m2r6 .6 3000" "m2r3 .3 6000"; do set -- $v; export TAG=$1 RMAX=$2 MAXC=$3
  STGPU=0 OMP_NUM_THREADS=2 BUILDONLY=1 python3 pipe.py > ${TAG}_build.log 2>&1 &
done; wait
for t in m2r6 m2r3; do export TAG=$t; [ $t = m2r6 ] && export RMAX=.6 MAXC=3000 || export RMAX=.3 MAXC=6000
  STGPU=0 flock /tmp/gpu.lock ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python pipe.py > $t.log 2>&1
  sed -e "s/A_g50d1p1.pkl/A_$t.pkl/" vstart_d1.py > vstart_$t.py; python3 vstart_$t.py > vstart_$t.out 2>&1
done
