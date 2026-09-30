# RESEARCH_HANDOFF — hub-research-3 → (нет следующего звена: очередь research пуста, idle по RESEARCH_ROUTINE)
Звено: hub-research-3 (acc3, Opus, 2026-09-30 15:40 → 16:30), причина смены: solved (вопрос «почему запросы медленные» закрыт), ctx 16%.
Сделано: профиль corridor_query по этапам на хабе и aida; таблица «система → время → узкое место» (`reports/research/query_speed.md`),
заметка `knowledge/research/query_speed.md`, рекомендации в PLAN.md (раздел research), 2 записи в ISSUES (хаб медленный — fixer; SLSQP хаотичен).
worker-6 получил сообщение (send_verified УСПЕХ).
Грабли: `pgrep -f`/`pkill -f` в `ssh ... 'bash -c ...'` находит/убивает саму ssh-команду (шаблон в её строке) — ждать по PID или файлам;
`task_log.py` пишет в `sessions_task_log/` ТЕКУЩЕЙ папки — не делать `cd` в той же команде; git без identity — `-c user.name=... -c user.email=...`;
скилла `anthropic-skills:deep-research` в списке скиллов нет — веб-часть делал WebSearch.
На aida оставлено: `~/spore_v5/r3prof/` (копия v6/src + rep/ со скриптами) и venv `~/spore_v5/r3prof/venv` (numpy 2.3.5, scipy 1.18.1).
Коммиты: d6e6a7a результат; 6fad4e2 старт.
Токены: T_START 5ч 0% / ctx 4% / $0.13; T_BEFORE_END 5ч 10% / ctx 16% / $3.8.
NEXT_LINK: hub-research-4 (поднимать по вопросу worker/dispatcher/пользователя)   NEXT_MODEL: opus
## Главное за звено (для пользователя)
1. **Где время:** ~98% запроса — SLSQP в `corridor_nd.refine`. Он зовёт Python-rk4 ОДНОЙ точкой: накладные numpy на векторе из 4 чисел
   (cProfile: accel/stack/ones_like). Дерево, KD-дерево, кандидаты — ≤ 3%.
2. **Лечение (прототипы, T те же на 8/8):** поток на `math` для точки (×13 хаб / ×22 aida на вызов), maxiter 300 → 80, точки зазора
   последовательно. Манипулятор на aida 49–60 с → 1.24 с; с препятствиями 87–111 → 5.5 с; маятник 4.9 → 0.47 с. Хаб → aida: ещё ×10.
3. **Хаб ≈ 10× медленнее aida на ядре** (чистый Python/C, не BLAS, не нагрузка) — тяжёлое только на aida, пул из 16 процессов (×13).
## Идеи для следующего research (не начаты)
- Якобиан ограничений зазора аналитически (остаток на препятствиях: dim 4+16·nseg конечными разностями); ns 16 → 8 с проверкой столкновений.
- Почему ~40% SLSQP по топологиям-промахам проваливаются (status 8) — отбор кандидатов до SLSQP (дешёвый тест допустимости).
