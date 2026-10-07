# research-22, эксп. 17: ждать полные слои b5 (п.38а, wb5/p38/Lfull.pkl), затем конвейер GPU: стенсилы → SBCAUS + SBLAY 1, среднее в группе → V/LB → агент без финиша и с shoot_pol VF 1
D=~/spore_v5/r22/17_di4_Lfull_pipeline; S=~/spore_v5/r22/12_di4_full_layers_gpu; F=~/spore_v5/wb5/p38/Lfull.pkl; cd $D; cp $S/{growN.py,finish_gen.py,sgpu.py,di4_ref_60_rho35.npy} . ; cp ~/spore_v5/r22/13_di4_fast_finish/{fin.py,fin2.py} .
n=0; while [ $n -lt 360 ]; do if [ -f $F ]; then a=$(stat -c %s $F); sleep 45; b=$(stat -c %s $F); [ "$a" = "$b" ] && break; fi; n=$((n+1)); [ $((n % 10)) = 0 ] && echo "жду Lfull.pkl: $((n/2)) мин"; sleep 30; done
[ -f $F ] || { echo "Lfull.pkl не появился за 3 ч"; exit 1; }
ls -la $F
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=$F PYTHONUNBUFFERED=1 SBLS=1
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python p17.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a "слои\|^u \|пар \|стенсилы 4\|SBLAY\|Error\|Trace" | stdbuf -oL cut -c1-420
VF=1 FH=.1 VFILE=$D/V_mean_sblay1.npy python3 fin2.py 2>&1 | grep -a "^VF\|Error\|Trace" | cut -c1-330
