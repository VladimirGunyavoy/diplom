D=~/spore_v5/r22/15_di4_mean_within_layer; S=~/spore_v5/r22/12_di4_full_layers_gpu; cd $D; cp $S/{growN.py,finish_gen.py,sgpu.py,di4_ref_60_rho35.npy} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/wb5/p38/L1600.pkl PYTHONUNBUFFERED=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python p15.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$" | cut -c1-420
