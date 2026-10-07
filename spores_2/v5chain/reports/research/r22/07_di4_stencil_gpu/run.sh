D=~/spore_v5/r22/07_di4_stencil_gpu; S=~/spore_v5/r22/01_p37_di4_V_vs_Tbox; cd $D; cp $S/{growN.py,finish_gen.py} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/w23/v7/reports/di4/L400_rho35.pkl PYTHONUNBUFFERED=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python sgpu.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$" | cut -c1-300
