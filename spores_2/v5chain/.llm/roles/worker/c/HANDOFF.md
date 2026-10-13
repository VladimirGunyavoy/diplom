# Session Handoff (линия C — bench)
Звено: hub-v5chain-worker-c5 (acc3 sonnet, 03:39–07:55), причина смены: ctx ~26% (софт 30%). Ведущий — hub-v5chain-research-26. Автономный режим (пользователя не спрашивать).
Протокол: knowledge/research/bench_protocol.md; код — spores_2/bench/ (sync.sh → aida ~/spore_v5/c1/spores_2/; pull.sh ← данные); сводки reports/bench/ (table_dp1, table_pend15, table_maze, table_manipf).
## Сделано (всё в git)
- dp1: дефект проверяльщика (эталон r23 на границе цели с допуском IPOPT) исправлен в common.DP1 (+2e-3 ящик, |w|≤3.01, точные старты); пересчёт, table_dp1.md; ocp_casadi dp1.
- pend15 (u .15): среда Pend15, spores_pend.py ENVN=pend15, table_pend15.md. manipf: строка спор FORB2b (A_f2b b11), старая — «до FORB2b».
- MAZE (дд 3D, лабиринт 6×6 DFS, 50 стартов): bench/maze.py (SDF, MazeDD, T_ref = A*+натяжение, верхняя оценка), prm_maze.py (PRM/PRM*), spores_maze.py (growN dd + подмена G.GF; LOADL/TAG/BUILDONLY), diag_maze.py (разбор провалов), make_maze_report.py → table_maze.md.
## Дальше
(1) MAZE 4D с инерцией (x,y,θ,v; блок race4 b10): maze.py + Env, SST/RRT/PPO/PRM-вариант (локальный переход — стрельба), споры growN. (2) MAZE режим (б): K=20 целей, retarget спор = solve. (3) OCP с инициализацией путём A* — отдельной строкой MAZE. (4) SAC+HER dp1. (5) RACE последним.
## Грабли
Старые growN на aida (manip_src, v7 10-07) без free/GF — брать ~/spore_v5/c1/maze_src (rsync v7/src/cells7). Рост слоёв — STGPU=0 без замка; solve — под замком (внешний flock + замок в скрипте = самоблокировка!). CPU-solve на 2M узлов не движется — SOLVEGPU=1 (venv-sb3). tqdm TQDM_MI=1000 → лог молчит. Старые Tref в aida-json MAZE неверны — таблица пересчитывает по per_start. Кросс-проверка эталона через проверяльщик — ПЕРВЫМ делом на новой среде. Убивать только свои PID.
Идёт на aida: ничего моего (B9 убит; maze_layers_{a,b,b6}.pkl на aida, V для b — maze_V_b.pkl).
NEXT_LINK: hub-v5chain-worker-c6   NEXT_MODEL: sonnet
