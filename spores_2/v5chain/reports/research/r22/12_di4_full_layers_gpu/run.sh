D=~/spore_v5/r22/12_di4_full_layers_gpu; cd $D; cp ~/spore_v5/r22/01_p37_di4_V_vs_Tbox/{growN.py,finish_gen.py} . ; cp ~/spore_v5/r22/07_di4_stencil_gpu/sgpu.py . ; cp ~/spore_v5/r22/10_di4_agent_nolatch/di4_ref_60_rho35.npy .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=${LAYERS:-~/spore_v5/wb5/p38/L1600.pkl} PYTHONUNBUFFERED=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python p38.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v "^$" | cut -c1-520
