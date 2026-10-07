cd ~/spore_v5/r23/02_manip_cover; cp ~/spore_v5/wb5/growN.py ~/spore_v5/wb5/finish_gen.py . 2>/dev/null
export SYS=manip KF=11 M=3 FRAC=.5 OVH=.5 MINROWS=3 MAXC=4000 NFAIL=400 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
run() { TAG=$1 setsid nohup env "${@:2}" python3 seed.py > s_$1.log 2>&1 < /dev/null & }
run base   GS=300 GLIM=.25 DELTA=.03 RMAX=.6 TMAX=3
run g0d03  GS=0 GLIM=0 DELTA=.03 RMAX=.6 TMAX=3
run g0d1   GS=0 GLIM=0 DELTA=.1  RMAX=.6 TMAX=3
run g0d3   GS=0 GLIM=0 DELTA=.3  RMAX=.6 TMAX=3
run g0d1r1 GS=0 GLIM=0 DELTA=.1  RMAX=1  TMAX=3
