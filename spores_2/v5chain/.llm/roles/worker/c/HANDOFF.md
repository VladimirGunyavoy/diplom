# Session Handoff (линия C — bench)
**ПАУЗА по слову пользователя (14:52, research-27): новых прогонов не запускать; идущие досчитать, результаты записать.** c7: сделаны (1) ДИ 4D b12 53/53 1.015, (2) total_m8.png+HYB; идут на aida: лабиринт цели (3,1)/(5,3) — solve+агент (PID 2679796, 2686051, лог ~/spore_v5/c1/mazeb/solve_*.log, JSON maze_spores_g*.json → записать в table_maze_goals.md «замер 2 целей, ×20 — оценка») и SAC dp1 сиды 1,2 (~/spore_v5/c1/spores_2/bench/sac_dp1_s*.log → table_dp1).
**Лабиринт: solve_3_1 упал AttributeError finish_gen.shoot_grid (SRC_DIR=b12/cells7 несёт старый finish_gen.py), V потерян (DUMP не было); (5,3) упадёт так же. НЕ перезапускать до слова; слои L_3_1.pkl/L_5_3.pkl целы (~/spore_v5/c1/mazeb), перед solve синхронизировать finish_gen.py из v7/src; в bench/spores_maze.py локально (не синхр.) добавлены подмена finish_gen из manip_src и DUMPV.**
Звено: hub-v5chain-worker-c6 (acc3 sonnet, 07:50–14:10), причина смены: ctx ~27% (софт 30%). Ведущий — hub-v5chain-research-26. Автономный режим.
Протокол: knowledge/research/bench_protocol.md; код spores_2/bench/ (sync.sh → aida ~/spore_v5/c1/spores_2/; pull.sh ← данные); сводки reports/bench/.
## Сделано (всё в git)
- MAZE 4D (дд с инерцией, x,y,θ,v): maze4.py, блок SYS=maze4 (mk_maze4_src.py копирует v7 и вставляет блок), spores_maze4.py, prm_maze4.py (PRM*+преследование), diag_maze4.py, cover_m4.sh, make_maze4_report.py → table_maze4.md. Споры 4×3000 0–6%, 4×10000 (FORB2=0 FORB2B=1 VIAB=1) 24% (заготовка 6446 с); SST/RRT 0%; PRM*+преследование 90%. Диагноз: клетки у цели, старты не покрыты.
- MAZE (б) K=20: prm_maze_goals.py, table_maze_goals.md (PRM* 65 с total, 100%; споры — оценка, не измерено).
- SAC SB3 dp1 (ppo_run.py RL_ALGO=SAC): 100%, 1.39. compare.md исправлен по explainer-3 (CA4 745/K*31/39, manipf FORB2b, ДИ 4D b12 215 с + колонка ресурсов).
## Дальше
(1) измерить retarget спор на лабиринте (режим б); (2) 8D в compare после table_m8 от w29; (3) B9 36k клеток; (4) SAC dp1 сиды 1–2; (5) RACE последним.
## Грабли
GPU-замок — только GPULOCK=1 в growN (в скриптах бенча замка нет; внешний flock не использовать — слово research-26). FORB2=.5 душит радиус клеток в 4D (FORB2=0 — клетки ×3.4, рост в 6 раз быстрее). aida venv-sb3: /home/random/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python; OMPL: ~/venv-ompl. Мониторы Monitor живут 30 мин — перевзводить. make_maze4_report.py перезаписывает table_maze4.md целиком (вывод внутри скрипта). table_di4/dp1 — строки b12/SAC добавлены вручную (make_report их не знает).
Идёт на aida: ничего моего.
NEXT_LINK: hub-v5chain-worker-c7   NEXT_MODEL: sonnet
