#!/bin/bash
# research-26: надзор за прогонами воркеров на aida — печатает только новые проблемы
SEEN=$(dirname $0)/aida_seen.txt; touch $SEEN
while true; do
  out=$(ssh -p 2222 -o ConnectTimeout=15 random@127.0.0.1 'bash -s' <<'R' 2>/dev/null
cd ~/spore_v5
# 1) свежие ошибки в логах, изменённых за 10 мин
for f in $(find . -name "*.log" -mmin -10 -size -200M 2>/dev/null); do
  tail -c 20000 "$f" | tr "\r" "\n" | grep -E "Traceback|Error|Killed|MemoryError|CUDA out of memory|nan%|NaN" | tail -2 | sed "s|^|ERR $f: |" | cut -c1-260
done
# 2) python-процессы старше 15 мин, у которых cwd-логи не обновлялись 15 мин
for p in $(pgrep -u random -f "python" ); do
  et=$(ps -o etimes= -p $p 2>/dev/null | tr -d " "); [ -z "$et" ] && continue; [ "$et" -lt 900 ] && continue
  cwd=$(readlink /proc/$p/cwd 2>/dev/null); case "$cwd" in /home/random/spore_v5*) ;; *) continue;; esac
  o=$(readlink /proc/$p/fd/1 2>/dev/null); case "$o" in /*) [ -f "$o" ] && cwd="$cwd $o";; esac
  fresh=$(find $cwd -maxdepth 1 -newermt "-15 min" -type f 2>/dev/null | head -1)
  # оболочки-обёртки (bash -c) не проверять
  case "$(ps -o comm= -p $p)" in bash|sh) continue;; esac
  # считает сам или его дети (пул) — не зависание
  cpu=$(ps -o pcpu= -p $p --ppid $p 2>/dev/null | awk '{s+=$1} END {print int(s)}'); [ "${cpu:-0}" -ge 20 ] && continue
  [ -z "$fresh" ] && echo "STALL pid $p (с $(ps -o lstart= -p $p | cut -c12-16)), в $cwd ничего не писалось 15 мин: $(ps -o args= -p $p | cut -c1-120)"
done
R
)
  [ -z "$out" ] && { sleep 300; continue; }
  while IFS= read -r l; do
    k=$(echo "$l" | md5sum | cut -c1-12); grep -q "$k" $SEEN && continue; echo "$k" >> $SEEN; echo "$(date +%H:%M) $l"
  done <<< "$out"
  sleep 300
done
