D=~/spore_v5/r22/13_di4_fast_finish; S=~/spore_v5/r22/12_di4_full_layers_gpu; cd $D; cp $S/{growN.py,finish_gen.py,di4_ref_60_rho35.npy} .
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=~/spore_v5/wb5/p38/L1600.pkl VFILE=$S/V_sblay1.npy PYTHONUNBUFFERED=1
for v in "VF=1 FH=.05" "VF=1 FH=.025" "VF=.6 FH=.05" "VF=1.5 FH=.05"; do env $v python3 fin.py 2>&1 | grep --line-buffered -a "^VF\|Error\|Trace" | cut -c1-360 & done; wait
