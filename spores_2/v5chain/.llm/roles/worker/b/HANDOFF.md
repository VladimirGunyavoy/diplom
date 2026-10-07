# Session Handoff — линия B
[РУТИНА НЕ ВЫПОЛНЕНА]
Звено: hub-v5chain-worker-b7 (acc3, sonnet, 15:33–21:10), причина смены: ctx ~27% (софт 30%)
Сделано: п.49 [B] v8 (stepper.py, v8/src/algo/growN.py SYS=di с паузами, frames.py + PNG, тесты, смоук LiveStepper w25; reject-пауза для плохих затравок). п.47: проба m3d MAXC 200 (1.95M узлов, build 519 с) + экстраполяция в results.md; MAXC 500 завис на передаче слоёв Pool (RSS слоя 3–7 ГБ) — слои 6D строить по одному, на диск. п.48: ДИ NF2 в grow_cells2d точнее базы (мед 1.003–1.011 против 1.04–1.05, ×3 клеток, RMIN .01 оставить), разрост 247 не воспроизведён; nD NF — `nf_rows` + `NORMFRONT=1` в v7 growN (по умолч. 0).
Стоп на / следующий шаг: nD NF на di4 L1600 ХУЖЕ базы (reach 75% против 95%, мед 1.19 против 1.036, max 8.9; узлов −17%, строки короче). Разобрать: печать nf+nb NF1 против NF0; `has.all()` в nf_rows обрывает строки при любом клоне без пересечения — замораживать клонов, NFW 4–6, NFSUB 8; повторить di4 (скрипт `~/spore_v5/wb5/run_nfdi4.sh`, aida: build STGPU=0 DUMPL, solve LOADL под flock), потом manip. Если не лечится — оставить NF выкл., закрыть п.48 nD. Параллельно открыто: п.47 6D ждёт посев от research (слои по одному).
Грабли: хаб 4 ядра/7.9 ГБ RAM — не больше 2–3 процессов (диспетчер); тяжёлое — на aida через launch_bg+ssh (grep --line-buffered, sed -u); GPU только под flock /tmp/gpu.lock; dd w21-постановка хрупка: старт solve держится на ~4 клетках с узлами в цели (reach 0 ≠ баг NF); git: пуш настроен fixer-5 (git pull --no-rebase && git push), идентичность `-c user.name=gun.vladimir26 -c user.email=gun.vladimir26@gmail.com` (в репо не задана); Pool зависает при больших слоях.
Решения цепочки: нет (NF nD по умолчанию ВЫКЛ; RMIN .01 для NF2 ДИ).
Коммиты: 4a9ac12 п.48 di4 NF1 хуже; 6c2ce26; 7ef632e nf_rows; f404f54 ДИ NF2; a7930c9/afe6346 п.49
Токены: T_START 5ч 3% / ctx 5% $0.06 (15:33) / T_BEFORE_END 5ч 29% / ctx 26% $9.3
NEXT_LINK: hub-v5chain-worker-b8   NEXT_MODEL: sonnet
