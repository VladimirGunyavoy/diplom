cd ~/spore_v5/r22/12_di4_full_layers_gpu; export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/wb5/p38/L1600.pkl PYTHONUNBUFFERED=1
for v in 0.6 1 1.5 2.5; do VF=$v python3 vf.py 2>&1 | grep --line-buffered -a "^VF\|Error\|Trace" | cut -c1-330 & done; wait
