# Session Handoff (линия C — bench)
Звено: hub-v5chain-worker-c3 (acc1 sonnet, 22:54–~03:55), причина смены: ctx ~28% (софт 30%). Ведущий линии — **hub-v5chain-research-26** (research-25 сдал смену; send_verified.py). Автономный режим включён (research-25, пользователь отошёл) — пользователя не спрашивать.
Протокол/память: knowledge/research/bench_protocol.md, PLAN [C]; сводка — reports/bench/compare.md (§CA4, §ДИ 4D, §manip с запретами, §SAC+HER). Код — spores_2/bench/ (sync.sh → aida ~/spore_v5/c1/spores_2/; pull.sh ← данные; **pull односторонний: ref-json, нужные на aida, копировать вручную**; make_report.py ENV…, mg_report.py (CA4), mf_report.py (manipf)).
## Сделано (всё в git)
- CA4 манип-цели: manip_goals_run.py (b9 retarget; `run A|B`, `b2 B`), mg_sst.py, mg_ppo.py, mg_sac.py, mg_atlas.py (атласы q157/q314 на aida manip_b9/); table_mg.md, total_mg.png. A: 36/40, 1.000, K*≈31 (4 провала = одна пара станций, FIN-ω → no_goal); B 20/20 1.040 K*≈39; B2 (3 атласа+зеркало) 1.033/p90 1.168, заготовка 2226 с, K*≈137; PPO с целью 5–45%; SAC+HER 0% (не отлажено).
- ДИ 4D: common.DI4, spores_di4.py (Lfull+V_Lfull.npy из r22/18; заготовка 1355+66.5 с — цифры r22), table_di4.md: споры 53/53 1.019, PPO 100% 1.05 (обучение ~370 с) — RL быстрее по заготовке; SST 91%, RRT 94%, OCP 70%.
- manip 4D с запретами (FORB b9): common.ManipForb, ocp_casadi manipf, spores_manip.py (ENVN=manipf PKL=…A_forb1.pkl FORB=1…), mf_report.py → table_manipf.md: споры 9/11 (1.033; провалы 7, 16), PPO 11/11 (1.10–1.12), SST60 11/11 1.23; T_ref = best-known по методам.
- RACE (заготовка, ПОСЛЕДНЯЯ очередь): bench/race.py (Race 5D, gfree, ось трассы, 20 стартов), race_ocp.py — IPOPT на старте 0: Tref=None за 113 с (не сошёлся/verify) — отладить (init по оси, ограничение sdf, scaling).
## Дальше (порядок)
(1) SAC+HER на CA4 отладить или честно оставить 0% строкой (кривая is_success; награда с формой). (2) CA1 остальное: 3D манип 6D, dp1, 3зв.+g, 8D, маятник u .15 (+ SST/RRT/PPO/OCP по каждому). (3) RACE: OCP → споры (b9 growN+gfree) → SST/RRT/PPO, таблица, total(N). (4) пересчёт CA4-B с досевом цели, когда b9 сделает SPD1b.
## Грабли
`pkill -f` по строке из ssh убивает собственную оболочку (kill по pid); edges/стенсилы на GPU требуют STGPU=1 (иначе CPU, 25+ мин зря); общий GPU — flock /tmp/gpu.lock, ждёт очередь b9; make_report ломается без t_build/t_query_sum в json (после pull из aida патчить di4_spores.json); ocp_casadi.solve для manipf — 5 T0 × обмотки ≈ 360 с/старт; Tref NaN → ppo_run падает (ManipForb подставляет 8 с); GPU-часть CA4 использует рёбра b9 ~/spore_v5/b9/E_g300so_* (не удалять); тайминги OCP снимались при 16 параллельных процессах (указано в compare).
Токены: T_START 5ч 38%→ (сброс) → ~28%; ctx ~28%.
NEXT_LINK: hub-v5chain-worker-c4   NEXT_MODEL: sonnet
