# Session Handoff — линия D (открыта hub-v5chain-research-26, 2026-10-13 ~13:15, слово пользователя: «пусти ещё одного воркера, на акк1, проверить на системе моего робота, legwheeled robot где-то на хабе, уже есть натренированный PPO»)
АВТОНОМНЫЙ РЕЖИМ (journals/_sentinel/AUTONOMOUS — первоисточник; решения — самому, с записью «решение <роль> (автономный режим)»).
Звено: hub-v5chain-worker-d1 (первое в линии D). Ведущий — hub-v5chain-research-26 (→27). Соседи: worker-30 (A, 6D/8D), worker-b12 (B, ускорение, владелец v7 growN), worker-c6 (C, бенч).

## Система пользователя — колёсно-ножной робот (Ascento-type), ЧУЖОЙ ПРОЕКТ — только читать
- Код: `~/claude-work/projects/colab-mhlb/legwheel_robot/` (копия `~/colab/legwheel_robot`): `PHYSICS.md` (уравнения, пределы — читать первым), `config.py` (главнее PHYSICS), `env.py`, `training/gym_env.py` (наблюдение 6D, награда, спавн, успех), `training/ARCHITECTURE.md`, `training/eval.py`.
- q = (x, ψ, α), действие (τ_wheel, τ_hip); состояние 6D (x, ψ, α, ẋ, ψ̇, α̇); α в наблюдении — отклонение от ALPHA_STAR.
- **Обученный PPO:** `~/claude-work/projects/colab-mhlb/runs/legwheel_robot/2026-09-14/003_165713/weights/0073/model.zip` (`last_eval/SOURCE.txt`: seq 0073, 9.57M шагов; спавн ψ ±0.1745, ψ̇ ±0.2094). `last_eval/metrics.json`: успех 1681/1681, время до цели ср 0.446 с, eval 435 эп/с.
- **Ничего в colab-mhlb не править** (чужой проект пользователя). Импортировать/копировать в `spores_2/bench/legwheel/` (файлы линии D).

## Задача D1 (PLAN [D] LW1)
1. Среда для бенча `bench/legwheel/env_lw.py` (наследник `common.Env`): та же динамика (импорт из colab-mhlb, не переписывать), те же спавн/критерий успеха/dt, что в eval PPO. Тест: траектории PPO через твой проверяльщик дают то же время и успех, что `metrics.json` (иначе — дефект постановки, как было с dp1).
2. PPO: инференс чекпоинта 0073 на ТЕХ ЖЕ стартах (N ≥ 50 из спавна eval), время до цели, успех, мс/запрос. Заготовка PPO — время обучения из логов прогона (честно, со своим железом).
3. Споры: лучшая 6D-версия — коридор v6 + refine_ca (CasADi/IPOPT), как в линии A на манипуляторах 6D (`bench/spores_6d.py`, `knowledge/research/bench_6d8d.md`); при нужде HYB (споры + OCP fast). Эталон — OCP CasADi multi (свободное T) на тех же стартах.
4. Таблица `~/claude-work/projects/spore/reports/bench/table_lw.md` по протоколу (`knowledge/research/bench_protocol.md` §2): дошли, T/T_ref, заготовка, запрос, total(N), N*.
Цель — честно: где споры быстрее/точнее PPO на системе пользователя, где нет.

## Где считать
aida (`ssh -p 2222 random@127.0.0.1`, `~/spore_v5`, venv `~/Calf_Barrier_Safe_MHLB_Code/.venv-sb3`): RAM ≤ ~50 ГБ на всех; GPU — только под flock /tmp/gpu.lock (одним способом: либо внешний flock, либо замок в скрипте). Логи с tqdm, первые минуты прогона проверять самому. git add — только свои файлы по именам.

## ПАУЗА LW1 (14:30, worker-d1 → VIZ по слову пользователя; ведущий research-27)
Сделано (aida ~/spore_v5/d1, код bench/legwheel/, sync_lw.sh): динамика CasADi (ош. 2e-5), env_lw+verify_real (тот же rk4_step), PPO 0073 сверен (1681/1681, T ср 0.4463 vs 0.44555), PPO на 50 стартах: T мед 0.478 с, инференс 0.047 мс; непрерывный OCP 40/50 T мед .0147 с; дискретный OCP (сетка .002) d50c считается (~/spore_v5/d1/ocp_d50c.log → data/lw_ocp_d.json, lw_ref.json); споры: spores_lw.py (LWTU=.01 обязательно; проба NB20000 NF3000 на 8: 3/8). Дальше: после d50c — spores_lw на 50 стартов, PPO-колонка, table_lw.md. Не забыть: вариант споры с квантованием переключений на сетку .002 (честность против PPO), alive=0 в награде PPO, успех PPO по награде не за скорость.
