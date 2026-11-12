# Session Handoff — линия D
Звено: hub-v5chain-worker-d3 (acc2, sonnet, 05:59–~07:0x 12.11), причина: ПАУЗА по слову пользователя (все мощности на журнал). Преемника нет. АВТОНОМНЫЙ РЕЖИМ.
Сделано: код линии в `reports/research/d3/` (tvg_vt/lp/dl с BOX=full и путями d3; g6d.py+lpcore.py — граф-схема 6D/8D для m3d/m3gd/m3gu/m8, НЕ запускался и не отлаживался). manip 4D 100k з.0–4 на aida1: 20/20, мед 1.069–1.104, p90 1.18–1.29, max 1.23–1.44, VI-LP 21 с, пик RSS 2.3 ГБ, у потолка ~9% (атлас prelim 18/20, 1.05, 1.156) — `reports/research/d3/table_inertia.md`, строка отправлена research-39.
Стоп / следующий шаг (когда вернут машины): dp1 100k з.0–4 (ожидаемо 17–19/20), manip/dp1 300k з.0–2, затем `g6d.py` (сначала смоук NB=5000 NSTART=3 на m3d, отладить; потом 100k/300k × 3 зерна для m3d, m3gd, m3gu, m8; для m8 нужен свой KLP и проверка RAM). Колонки: нарушения (verify why=='box'), RSS пик, хранимое МБ.
Запуск manip/dp1: `~/spore_v5/d3/run1.sh ENV N SEED TAG` (aida1; своё дерево `~/spore_v5/d3/spores_2` = копия a39 + refs r23 из c1; результаты `~/spore_v5/d3/out/lp_*.json`, логи log_*.log/time_*.txt). Для 6D: `ENVN=m3d NB=.. SEED=.. ~/spore_v5/d3/run6.sh`. Условия: nice -n 10, GPU не трогать, free -g/uptime перед запуском, бюджет research-39.
Грабли: (1) алиас `aida` = aida1 (порт 2222); aida2 — порт 2223, ключ с хаба НЕ принят (Permission denied) — позвал fixer-1 (доставлено, ответа не было). (2) контроль з.0 manip: мед 1.086 против c12 1.073 (p90/max совпали) — не разобрано (TAU=.4 RB=.35 мои). (3) common из a39 не содержит refs r23 — взяты из c1. (4) моих процессов на машинах нет (manip-прогоны кончились ~06:02).
Решения цепочки: нет.
Коммиты: reports/research/d3 (код, отчёт).
NEXT_LINK: нет (пауза) NEXT_MODEL: sonnet
