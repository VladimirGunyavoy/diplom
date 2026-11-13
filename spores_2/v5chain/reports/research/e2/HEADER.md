# Заголовок прогонов e2 коридор v6 (правило проверки кода research-39, 13.11)
- Код: `e2/corridor_env.py` (новый обобщённый драйвер по образцу `bench/spores_6d.py`; diff — `e2/diff_vs_spores_6d.txt`, файл коммитом). Базового 4D-прогона коридора нет → контрольной известной строки нет (контроль платформы olya сделан на графе ДИ4D: совпал 1.0197/1.065/1.511).
- Старты+эталон: `r23/dp1_ref_ocp_20.npy`, `r23/manip_ref_ocp_20.npy` — md5 в `e2/starts_md5_hub.txt`, на olya сверено.
- Проверяльщик: `common.verify` (h .01, tfac 3), статистика `common.summarize` (допуск T ≤ 3·T_ref); окно цели = env.half − 0.01 в доводке; коробка — env.lo/hi (verify 'box'); WL=1 (|v| ≤ min(env.hi) вдоль пути).
- Флаги: NB 12000, NF 3000, K 3, tau .4, rho .15, TRIES 3, kn 8, neigh 2, 4 угла куба u, SLSQP (не IPOPT), spawn-пул 16, olya (Windows, numba/numpy).
