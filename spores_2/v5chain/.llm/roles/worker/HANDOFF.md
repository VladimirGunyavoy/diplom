# Session Handoff — линия A
[РУТИНА НЕ ВЫПОЛНЕНА]
АВТОНОМНЫЙ РЕЖИМ.
Звено: hub-v5chain-worker-42 (acc1, sonnet, 01:54–05:05), причина смены: передача большой задачи [1] (ctx 23%, задача m3gdg требует чистого контекста)
Сделано: адаптивные узлы ДИ4D (ADAPT, growNa.py; −22/−41% узлов, потери стартов у границы; бит 8 лечит ценой экономии), атлас ДИ4D 5 зёрен на 300 свежих r37 (253/253, мед 1.037, p90 1.092, max 2.075) — отчёт v7/reports/big/10_adapt/adapt_di4.md, json в data/. Строки отправлены research-38.
Стоп на / следующий шаг: PLAN [1] (блок «★ Решение senior-2 04:5x»): строка m3gdg — граф-схема 100k (RNDN KLP=20 VCAP=40 TAUS×3) против атласа и RRT*/SST по протоколу c7; build / VI на новую цель / запрос / память пик-хранимое; называть «кинодинамическая дорожная карта». Разведка: tvg_vt.py берёт среду из common.make(ENV) — m3gdg там НЕТ; сцена и цепочка целей — bench/m3gdg_goals.py (K=19 переходов, GSEED 11, диски OB, manip3g_dyn as G3, envs_6d); данные c7 — bench/data/m3gdg_*.json, прежние строки — presentation_prelim.md §3.13. Нужен класс среды m3gdg (6D, 2 диска, g) под tvg_vt/tvg_lp + цикл по целям (на каждую цель — только VI, граф один раз). Код граф-схемы: reports/research/r37/tvg_*.py (на aida1 ~/spore_v5/r37 — не трогать, своя копия, напр. reports/research/w43/). RAM 100k в 6D пик ~14 ГБ — по одному, free -g, запас 10 ГБ.
Грабли: (1) growN/growNj/growNd __main__ НЕ читают SEED (ISSUES) — для зёрен нужен Atlas(seed=..) (в growNa исправлено). (2) aida1 = `ssh aida` (порт 2222), aida2 = `ssh -p 2223 aida@127.0.0.1`; на aida1 python = ~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3/bin/python, на aida2 ~/spore_v5/venv. (3) pkill -f убивает собственный ssh — использовать pgrep -f "[x]". (4) ≥3 одновременных solve на GPU 16 ГБ — CUDA OOM (aida2); aida2 туннель 2223 раз падал и вернулся сам. (5) Слово пользователя (04:15): грузить железо до ~90% ядер пачкой, RAM-запас ≥10 ГБ; OOM-killer на aida1 задел research-38.
Решения цепочки: нет.
Идёт на aida2: lost.py (a42/out/lost2.log → lost_s0.json) — конечные точки застревания 4 вариантов (результат дописать в adapt_di4.md); a42/watch_solve.py (можно убить: pgrep -f "[w]atch_solve").
Коммиты: git log --oneline -3
Токены: T_START 5ч 40%/ctx 5% 01:54 / T_BEFORE_END 5ч 72%/ctx 23% 05:00
NEXT_LINK: hub-v5chain-worker-43   NEXT_MODEL: sonnet
