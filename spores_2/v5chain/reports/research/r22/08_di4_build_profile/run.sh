D=~/spore_v5/r22/08_di4_build_profile; S=~/spore_v5/r22/01_p37_di4_V_vs_Tbox; cd $D; cp $S/{growN.py,finish_gen.py} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 MAXC=${MAXC:-200} PYTHONUNBUFFERED=1 TQDM_MI=10
python3 prof.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$" | cut -c1-210
