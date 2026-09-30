# Почему запросы corridor_query медленные (research hub-research-3, 2026-09-30)
**Вопрос** (PLAN, слово пользователя 2026-09-30): где время запроса (манипулятор 4D ~1–3.5 мин), разница хаб/aida, среда.
**Ответ одной фразой:** ~98% — SLSQP в `refine`, который зовёт Python-rk4 ОДНОЙ точкой (накладные numpy на векторе из 4 чисел ×13–22)
и изредка крутит maxiter=300; хаб сам по себе ≈10× медленнее aida на ядре (VM), а BLAS/потоки ни при чём.

## Схема (таблица — `reports/research/query_speed.md`)
| система | хаб, исх. | aida, исх. (scipy 1.11) | aida, рекоменд. | узкое место |
|---|---|---|---|---|
| дд 3D | 1.5 с | 0.13 с | 0.15 с | refine 90%, не критично |
| маятник | — | 5–6 с | 0.47 с | rk4 точкой; status 8 — 67% |
| манипулятор 4D | 124 с | 49–60 с | 1.24 с | numpy в accel/f4 85%; maxiter-хвост |
| манипулятор + препятствия | — | 87–111 с | 5.5 с | зазор: 16 точек, каждая с начала сегмента |
Рекомендуемое = поток на `math` для точки + maxiter 80 + точки зазора последовательно; T то же на 8/8 (препятствия 7/8, 1/8 +0.5%).

## Проверка (входы → результат)
- Профиль: `reports/research/profile_query.py SYS NQ` (из v6), сырые JSON — `reports/research/query_speed_runs/`; cProfile манипулятора:
  accel 47.8 с + stack 13.9 + ones_like 11.3 из 109 с — это накладные вызовов, не арифметика.
- `flow4_scalar.py`: 200 точек × 4 слоя, max|разница| 1.8e-15; мкс/вызов numpy 1403 (хаб) / 245 (aida) → math 110 / 11.
- maxiter 300/80/40 на 8 запросах: 80 — T тот же; 40 — манипулятор q2 2.319 → 2.677 ✗. Успешные nit: p50 8–9, p99 20–56.
- `EARLY` (бросить топологию после 1-й недопустимой попытки) ✗: теряет 1–2 запроса из 8 (None / +15%).
- Батч-якобиан (∂x_end/∂dt_k = Φ(после k)·f_{s_k}(x_k), столбцы одним батчем) верен (1e-7), но со скалярным потоком медленнее — не нужен.
- Среда: хаб Xeon Gold 6140 KVM 4 vCPU, numpy 2.5.3 / scipy 1.18.1; aida Ryzen 9 9950X 16C/32T, numpy 1.26.4 / scipy 1.11.4.
  Чистый Python 0.40 vs 0.038 с, sum(range(3e7)) 2.0 vs 0.19 с — не зависит от load (0.9 и 2.8), ядра, cgroup (не троттлит), steal ~1%.
  OMP/OPENBLAS/MKL_NUM_THREADS не заданы — не важно: матриц в запросе нет. aida: 16 процессов ×12.8 пропускной, 32 — не лучше.
- scipy 1.18 vs 1.11 (venv на aida): манипулятор 22 тыс. вызовов потока/запрос вместо 72 тыс. — SLSQP переписан на C в 1.16.
- SLSQP хаотичен: разница потока 1e-15 меняла T трудного запроса (2.319 ↔ 2.853) — сравнивать версии сериями запросов.

## Ссылки
SciPy 1.16.0 release notes (SLSQP Fortran → C): https://docs.scipy.org/doc/scipy/release/1.16.0-notes.html ;
Egerstedt, Wardi, Axelsson, «Transition-time optimization for switched-mode dynamical systems», IEEE TAC 51(1):110–115, 2006
(градиент по моментам переключения через сопряжённые/касательные — основа батч-якобиана); Stellato et al., arXiv:1608.08597 (2-й порядок).

## Что нужно worker'у
1. Ветку `x.ndim == 1` на `math` в `flow4` (и в pendulum/rk4, dd flow) — прототип `reports/research/flow4_scalar.py`; тест ≤ 1e-12.
2. `corridor_nd.refine`: maxiter 80; точки зазора последовательно (`refine_seq` в `reports/research/refine_fast.py`).
3. Серии запросов — на aida, `multiprocessing.Pool(16)`, build_back до пула. Хаб — лёгкие проверки.
4. scipy ≥ 1.16 на aida — fixer/пользователь (venv `~/spore_v5/r3prof/venv` проверен).
