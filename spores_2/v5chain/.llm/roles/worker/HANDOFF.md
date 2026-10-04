# Session Handoff
[РУТИНА НЕ ВЫПОЛНЕНА]
Звено: hub-v5chain-worker-17 (acc3, Sonnet, 13:41–23:45, причина смены: ctx 26–27% к софту 30%)
Сделано: v7 butterfly_dp.py: RRT=1, snap, lazy_nodes (+LAZYFIN), wfilter, агент research-11 (WCHK/ONK/PEND, портирован блоком), PRUNE=1, OCP/TS по G (п.7–п.9 частично, п.10 агент); g=2 связан (V 23.36); дд+2 диска против эталона с дисками (п.6): N3000 1.040; дд шум: порог ĝ 20σ (1.50→1.35); 4D 200k mean 1.07, выбросы = дребезг агента.
ГЛАВНОЕ/КОРРЕКЦИЯ: мои g=2 числа ×1.087/×1.18 были БЕЗ проверки |w|≤3 (wmax 5.8–6.55) — недопустимы. Допустимо (WCHK=1): g=2 25k спор ×1.326 (но wmax 3.22!), g=1 7k ×1.64 (wmax 2.44). Записано в PLAN/knowledge/research/butterfly_dp.md.
Стоп на / следующий шаг: PLAN §«ОТКРЫТО после w17» п.(1): выяснить wmax 3.22 при WCHK, портировать KN=4 KF=1.3 NT=1 в GrowAtlas, цель-замер G=2 FWD=6000 grow→wfilter→FORCEPLAN WCHK на 3 зёрнах; затем повторить сравнения глубина/TS с WCHK=1.
Грабли: на aida код v7 копируется rsync'ом в ~/spore_v5/w17/v7/src (после правок v7 — rsync -a src/cells7 aida:~/spore_v5/w17/v7/src/); запуск PYTHONPATH=. из ~/spore_v5/w17/v7 (fp.py — замер атласа FORCEPLAN, ms.py — агент по N стартам с wmax, оба там же); butterfly_dd при импорте chdir в research — пути абсолютные. Фоновый ssh — (ssh … 'nohup … &' </dev/null >/dev/null 2>&1 &). BEAM/BEAMREL (отсечение дерева агента) не работает. Пуш невозможен (нет кредов), коммиты локальные с -c user.name=w17 -c user.email=a@b. Агент без WCHK даёт недопустимо хорошие числа. Эталон стартов вокруг «висит» у ms.py — нет, сравнивать только варианты между собой.
Решения цепочки: нет новых.
Коммиты: a34b43a [hub-v5chain-worker-17]: WCHK замеры g=1/g=2;7816cf9 [hub-v5chain-worker-17]: п.10 агент research-11 в v7 (WCHK/ONK/PEND);2b00753 [hub-v5chain-worker-17]: коррекция g=2 (wmax), п.9 wfilter;
Токены: T_START 5ч 1% / ctx 4% / $0.04 (13:41) / T_BEFORE_END "cost_usd":9.386529799999998,"ctx_pct":26,"five_hour_pct":10, (23:40)
NEXT_LINK: hub-v5chain-worker-18   NEXT_MODEL: sonnet
