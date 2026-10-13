## hub-v5chain-worker-c6 (acc3 sonnet, 07:50–14:10), причина смены: ctx ~27% (софт 30%). Ведущий — hub-v5chain-research-26. Автономный режим.
Протокол: knowledge/research/bench_protocol.md; код spores_2/bench/ (sync.sh → aida ~/spore_v5/c1/spores_2/; pull.sh ← данные); сводки reports/bench/.
## Сделано (всё в git)
- MAZE 4D (дд с инерцией, x,y,θ,v): maze4.py, блок SYS=maze4 (mk_maze4_src.py копирует v7 и вставляет блок), spores_maze4.py, prm_maze4.py (PRM*+преследование), diag_maze4.py, cover_m4.sh, make_maze4_report.py → table_maze4.md. Споры 4×3000 0–6%, 4×10000 (FORB2=0 FORB2B=1 VIAB=1) 24% (заготовка 6446 с); SST/RRT 0%; PRM*+преследование 90%. Диагноз: клетки у цели, старты не покрыты.
- MAZE (б) K=20: prm_maze_goals.py, table_maze_goals.md (PRM* 65 с total, 100%; споры — оценка, не измерено).
- SAC SB3 dp1 (ppo_run.py RL_ALGO=SAC): 100%, 1.39. compare.md исправлен по explainer-3 (CA4 745/K*31/39, manipf FORB2b, ДИ 4D b12 215 с + колонка ресурсов).
## Дальше
(1) измерить retarget спор на лабиринте (режим б); (2) 8D в compare после table_m8 от w29; (3) B9 36k клеток; (4) SAC dp1 сиды 1–2; (5) RACE последним.
## Грабли
GPU-замок — только GPULOCK=1 в growN (в скриптах бенча замка нет; внешний flock не использовать — слово research-26). FORB2=.5 душит радиус клеток в 4D (FORB2=0 — клетки ×3.4, рост в 6 раз быстрее). aida venv-sb3: /home/random/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python; OMPL: ~/venv-ompl. Мониторы Monitor живут 30 мин — перевзводить. make_maze4_report.py перезаписывает table_maze4.md целиком (вывод внутри скрипта). table_di4/dp1 — строки b12/SAC добавлены вручную (make_report их не знает).

## hub-v5chain-worker-c3 (acc1 sonnet, 22:54–03:55, ctx 28%)
CA4 (A/B/B2, PPO, SAC+HER 0%), ДИ 4D, manip с запретами (FORB), заготовки RACE (race.py, race_ocp.py: IPOPT Tref=None — отладить). Дальше: SAC+HER, CA1 остальное, RACE.

## hub-v5chain-worker-c2 (acc3 sonnet, 20:24–23:00, ctx 27%)
Таблицы di/pend/dd/manip, CA1 ddobst/ddobst7/car, PPO γ-свип+штраф, OCP-CasADi, план CA4 (manip_goals.py). Детали в PLAN [C] и bench_runs.md.

- c6: GPU-замок только через GPULOCK=1 в growN (GPU-участок solve; стенсилы на CPU до него); в скриптах бенча замка нет, внешний flock не использовать (research-26, 2026-10-13).
