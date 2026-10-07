# повтор эксп. 17: GPU был занят прогоном b5 на тех же слоях (15.4 ГБ) — ждать освобождения карты (< 1.5 ГБ два замера подряд), затем конвейер
D=~/spore_v5/r22/17_di4_Lfull_pipeline; F=~/spore_v5/wb5/p38/Lfull.pkl; cd $D
n=0; k=0; while [ $n -lt 480 ]; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); if [ "$u" -lt 1500 ]; then k=$((k+1)); else k=0; fi; [ $k -ge 2 ] && break; n=$((n+1)); [ $((n % 20)) = 0 ] && echo "жду GPU: $((n/4)) мин, занято $u МБ"; sleep 15; done
[ $k -ge 2 ] || { echo "GPU не освободился за 2 ч"; exit 1; }
export SYS=di4 M=3 SIDE=0 GM=.25 GLIM=0 FRAC=1 RHO=.35 PESS=1 LAYERS=$F PYTHONUNBUFFERED=1 SBLS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python p17.py 2>&1 | stdbuf -oL tr '\r' '\n' | grep --line-buffered -a "слои\|^u \|пар \|стенсилы 4\|SBLAY\|Error\|Trace\|p17.py\", line" | stdbuf -oL cut -c1-420
VF=1 FH=.1 VFILE=$D/V_mean_sblay1.npy python3 fin2.py 2>&1 | grep -a "^VF\|Error\|Trace" | cut -c1-330
