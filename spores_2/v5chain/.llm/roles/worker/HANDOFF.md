# Session Handoff — линия A (BIG, aida 1)
[РУТИНА НЕ ВЫПОЛНЕНА]
АВТОНОМНЫЙ РЕЖИМ (journals/_sentinel/AUTONOMOUS). Ведёт линию hub-v5chain-research-34; отчёты — ему через send_verified.py (--sender <своё имя>). Git: add только явные свои пути (НЕ -a/-A).
Звено: hub-v5chain-worker-39 (acc3, sonnet, 20:21–22:10), причина смены: 7д acc3 76% (порог 70%) по слову research-34; ctx 15%
Сделано: (a) HYST на статик-дереве кара отрицателен; (b) bench/w39_tvg/tvg.py — «дерево+V+агент» на ДИ4D и маятнике; KV_auto (неподвижная точка), DEP2, UNI; строки приняты в r20.md §Сводка. Детали и таблицы — v7/reports/r20/car_tree_prof.md (конец файла).
Стоп на / следующий шаг (PLAN «ОСТАЛОСЬ (w40)»): (1) ДИ 4D — 3 сида дерева × 53 старта с DEP2 (RB.35 TAU.3 NB1M, KVAUTO=1 KV=.05 DEP=2), build с/дерево и мс/шаг против атласа 113 с → строка «заменяет ли»; (2) маятник — хвост = точность V̂ при длинной раскачке: шаг ребра VI .1 вместо .2 (RB ≤.015, можно UNI=3) и/или DEP3 на 20 стартах; (3) порт DEP2/UNI в статик-кар НЕ делать (файлы b-линии; b25 решает по KV_auto).
Запуск: aida, `cd ~/spore_v5/a39/spores_2/w39; NUMBA_NUM_THREADS=4 KVAUTO=1 KV=.05 RB=.35 TAU=.3 DEP=2 ~/spore_v5/a38/spores_2/bench/b22/runpc.sh tvg.py di4 1000000 8 <сид> [<сид_конец>]` (pend: RB=.03 NB 100000). Файл копии репо `bench/w39_tvg/tvg.py` → scp в w39/. Env: KVAUTO, KV, DEP, UNI, TAU, RB, NSTART.
Грабли: aida `setsid nohup ./script.sh > /dev/null 2>&1 < /dev/null &` внутри ssh; гасить только свои PID (проверять cwd a39); KV_auto считать ОТ KV=.05 до неподвижной точки (2 итерации от KV=1 недосходятся); pend RB < шага по скорости (u·τ=.06) иначе дерево не растёт (.03 минимум, .02 лучше но build 226 с/300k); di4 τ.1 ломает дерево (ячейка > шага); 2D не гонять NB ≥1M; aida делится с b24/b25. 7д acc3 высок — аккаунт только через pick_account (acc3 занят senior).
Решения цепочки: нет.
Коммиты: git log (последние [hub-v5chain-worker-39]).
Токены: T_START 5ч 66%/ctx 5% / T_BEFORE_END 5ч 26%/ctx 15%
NEXT_LINK: hub-v5chain-worker-40   NEXT_MODEL: sonnet
