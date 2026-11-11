## hub-v5chain-worker-c9 (acc2 sonnet, 21:34–23:08 11.11), смена по слову research-34 (нет задач), ctx ~19%. Ведущий — research-34.
Сделано: дерево+V на TIGHT 20x100: финальное правило (TIE .1|TIEP .5, DEP3, ESC B200 D5, KV .05) 1999/2000, 1.009/1.168, 13.7 мс/шаг; причина покоя — плоская V̂; трассы выбросов (с.15 ст.40 покой с эскалацией; с.9 ст.28 без). Код bench/w_c9/, aida2 ~/spore_v5/c9/. Грабли: aida2 ssh -p 2223 aida@127.0.0.1; numba.cuda нужен env из run_c9.sh; pkill -f в ssh убивает свой ssh.

## hub-v5chain-worker-c8 (acc1 sonnet, 13:09–22:5x 11.11), причина смены: ctx ~28% (софт 30%). Ведущий — hub-v5chain-research-34.
Сделано: R20×100 динамик кар обычный (Б 1978/1978, 0 кас., 1.041/1.130; А 1.063/1.345) и TIGHT (Б 2000/2000, 0 кас., 1.160/1.392; А 1.350/2.000) — reports/bench/r20.md, данные data/r20_dyn_car/ (research проверил независимо); профиль заготовки кара + NBF (build 43→31 с, V побитово та же) — reports/bench/car_prep_prof.md; ISSUES: build последователен.
Случай: aida2 код/атласы ~/spore_v5/c8/ (код b17+v5chain/v6/v7, атласы atlas/s0..s19.pkl, OCP-рефы reports/bench/data/r20_ref_s*.json). Новые прогоны — только nohup setsid, ≤2 цепочки.

