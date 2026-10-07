D=~/spore_v5/r22/09_di4_V_uniqueness; S=~/spore_v5/r22/03_di4_edge_causality; cd $D; export SAVEE=$S/E VREFP=$S/V.pkl LBP=~/spore_v5/r22/01_p37_di4_V_vs_Tbox/LB.npy PYTHONUNBUFFERED=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python uniq.py 2>&1 | cut -c1-400
