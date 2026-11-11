#!/bin/bash
# research-32: этап 1 (numba, r31/venv) со сторожем ОЗУ, затем этап 2 (torch, sb3 venv, GPU)
cd ~/spore_v5/a31/spores_2/bench
# ждать окна по ОЗУ (машину грузят воркеры): старт при avail >= 17 ГБ, не дольше 3 ч
for i in $(seq 1 1080); do av=$(free -g | awk 'NR==2{print $7}'); [ "$av" -ge 17 ] && break; [ $((i % 30)) -eq 1 ] && echo "$(date +%T) жду ОЗУ avail=${av}G"; sleep 10; done
echo "$(date +%T) старт этапа 1"
DTN=.2 ~/spore_v5/r31/venv/bin/python -u ~/spore_v5/r32/gj1_r32.py ~/spore_v5/a34/cf/d2s1.pkl > ~/spore_v5/r32/gj1.txt 2>&1 &
P=$!; echo $P > ~/spore_v5/r32/gj1.pid
while kill -0 $P 2>/dev/null; do
  av=$(free -g | awk 'NR==2{print $7}')
  if [ "$av" -lt 6 ]; then kill $P; echo "$(date +%T) KILLED by guard avail=${av}G" >> ~/spore_v5/r32/gj1.txt; exit 1; fi
  sleep 5
done
~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python -u ~/spore_v5/r32/gj2_r32.py > ~/spore_v5/r32/gj2.txt 2>&1
echo "$(date +%T) EXIT" >> ~/spore_v5/r32/gj2.txt
