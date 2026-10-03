#!/bin/bash
# research-9: очередь лёгких прогонов — по одному, ≤ 1.5 ГБ, старт только при MemAvailable ≥ 2 ГБ (слово пользователя 2026-10-03)
cd "$(dirname "$0")"
while read -r envs args; do
  [ -z "$args" ] && continue
  until [ $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) -ge 2048 ]; do sleep 60; done
  echo "== $(date +%T) $envs $args"
  E=""; for kv in ${envs//,/ }; do E="$E -E $kv"; done
  systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 $E python3 $args 2>&1 | grep -v Running | cut -c1-330
done < "$1"
