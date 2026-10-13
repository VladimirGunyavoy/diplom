# Session Handoff — линия A (открыта hub-v5chain-research-26, 2026-10-13 ~03:05, по слову пользователя «разрешаю ещё одного воркера»)
АВТОНОМНЫЙ РЕЖИМ (journals/_sentinel/AUTONOMOUS — первоисточник; пользователь спит, решения принимать самому с записью «решение <роль> (автономный режим)»).
Звено: hub-v5chain-worker-27 (первое в новой линии A; прошлый хендофф worker-26 — `.llm/chain/history/HANDOFF_worker-26_20261012.md`, тема другая, не продолжать).
Ведущий линии: hub-v5chain-research-26 (вопросы — ему, `send_verified.py`). Соседи: worker-c4 (бенч, CA1 лёгкие, PPO/SAC), worker-b10 (ускорение, владелец `v7/src/cells7/growN.py`).

## Глобальная цель (слово пользователя 2026-10-13 ~03:05)
«Сделать spore максимально быстрым и показать, что он быстрее других алгоритмов». Твоя часть — **ПОКАЗАТЬ** на 6D-средах, где конкуренты
деградируют с размерностью. Протокол бенча — `knowledge/research/bench_protocol.md` (§2: заготовка, вес, total(N), безубыточность N*), честность строже выигрыша.

## Задачи линии A (PLAN, тег [A]; по порядку)
1. **[A] CA1-6D-a — 3D двузвенный манипулятор 6D без g** (`knowledge/research/manip3d_2link.md`, `reports/all_systems.md` №11: споры коридор v6 NB12000 NF3000, 16/16, 1.014 к OCP CasADi).
   Среда — `spores_2/bench/envs_6d.py` (наследник `common.Env`; файл твой), 20 стартов, эталон — OCP CasADi (как в all_systems, переиспользовать скрипт research-21),
   методы: споры (версия из all_systems, заготовка/запрос раздельно), OCP IPOPT на запрос, SST/RRT (свои `sst_own.py` или OMPL, бюджеты .3/1/3/10/60 с), PPO 5M (`ppo_run.py`).
   Таблица `~/claude-work/projects/spore/reports/bench/table_m3d.md` + total(N) PNG тем же `make_report.py` (если ломается — своя копия `make_report_a.py`).
2. **[A] CA1-6D-b — 3 звена 6D + g** (all_systems №8: коридор v6, вниз/вверх 32/32) — так же, `table_m3g.md`; затем №8d с 2 дисками (6 — неизбежное столкновение: исключать для всех одинаково).
3. **[A] CA1-8D — 4 звена 8D** (эталона нет — доля дошедших и время по всем методам).
## Владение файлами
Твои: `bench/envs_6d.py`, `bench/*_6d*.py`, `bench/run_*6d*.sh`, таблицы `table_m3d/m3g/m8.md`. **Не править:** `bench/common.py`, `envs_heavy.py`, `make_report.py` (c-линия — просьба строкой в PLAN),
`v7/src/cells7/growN.py` (B). Код v6 коридора — читать/импортировать; правка — копией в bench/.
## Где считать
aida (`ssh -p 2222 random@127.0.0.1`, `~/spore_v5`, 32 ядра, 60 ГБ, GPU 16 ГБ через flock /tmp/gpu.lock): ноут на паузе — aida наша, но делим с c4/b10: RAM ≤ ~50 ГБ на всех.
Долгое — nohup + tqdm, ждать не циклом. Результаты (таблицы < 1 МБ) — коммит и пуш сразу (`git push git@github-paper:VladimirGunyavoy/diplom.git master` из `~/claude-work/projects/spore`; автор VladimirGunyavoy / gun.vladimir26@gmail.com).
