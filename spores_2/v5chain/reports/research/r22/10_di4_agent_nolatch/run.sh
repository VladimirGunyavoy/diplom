D=~/spore_v5/r22/10_di4_agent_nolatch; S=~/spore_v5/r22/03_di4_edge_causality; cd $D; cp ~/spore_v5/r22/01_p37_di4_V_vs_Tbox/{growN.py,finish_gen.py} .
export SAVEE=$S/E VREFP=$S/V.pkl LBP=~/spore_v5/r22/01_p37_di4_V_vs_Tbox/LB.npy PYTHONUNBUFFERED=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python mkv.py 2>&1 | cut -c1-330
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 LAYERS=~/spore_v5/w23/v7/reports/di4/L400_rho35.pkl
for v in "latch_p1 1" "nolatch_p1 1" "nolatch_p0.3 .3" "nolatch_p0.1 .1" "nolatch_p0 0"; do set -- $v; VN=$1 PESS=$2 python3 roll.py 2>&1 | grep -v "^$" | cut -c1-330 & done; wait
