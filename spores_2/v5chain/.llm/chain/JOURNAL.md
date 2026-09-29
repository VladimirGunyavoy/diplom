# JOURNAL — события цепочки (одна строка на событие, только `>>`)
hub-worker-1 | started | acc1 | sonnet | 04:12:04
hub-worker-1 | done-step | PLAN 1-2: ядро src/atlas + тест, цифры §6 совпали | 04:13:40
hub-worker-1 | done-step | PLAN 3: картинки + черновик отчёта | 04:23:42
hub-worker-1 | NEEDS-HUMAN | дифдрайв: (v, ω) — управления (кинематика, 3D) или часть состояния (5D)? по умолчанию делаю 3D-кинематику (схема A) | 04:35:58
hub-worker-1 | done-step | атлас DI SOLVED: отчёт reports/atlas_double_integrator.md; заготовка дизайна дифдрайва | 04:35:58
hub-worker-1 | done-step | дифдрайв A1: режимы Straight/Rotate/Arc + тест (замкнутая форма = численное интегрирование) | 05:00:46
hub-worker-1 | done-step | дифдрайв A2: state lattice 16 курсов + Дейкстра, тест | 05:10:46
hub-worker-1 | done-step | дифдрайв A2b: дуги (arc_sweeps) + тест | 06:31:06
hub-worker-1 | done-step | дифдрайв A4: rollout управлений из произвольной точки + тест | 06:41:06
hub-worker-1 | done-step | дифдрайв: тонкий финиш, сходимость дуг (не сошлась), A5-lite, отчёт до/после | 07:35:22
hub-worker-1 | done | ctx 22% + простой | атлас DI SOLVED, дифдрайв A1–A4+A5-lite, сдача по замечанию пользователя | next=hub-worker-2 | 14:46:06
hub-worker-2 | started | acc1 | sonnet | 14:46:32
hub-worker-2 | done-step | дифдрайв A2c: дуги в узле (arc_edges), сходимость не убывает | 14:47:39
hub-worker-2 | done-step | схема B 5D: five_d.py, тест, сравнение с A; интерполяция завышает T (2.87 vs 2.0) | 14:58:07
hub-worker-2 | done-step | индикатор адаптации A: ошибка равномерная, дробление по r не нужно | 14:58:38
hub-worker-2 | done-step | B 5D rollout5 depth1-3: конец в 0.05-0.1 от цели, покой; T завышена интерполяцией | 15:14:41
hub-worker-2 | done-step | B: сходимость по nth, повороты -2..-6% при nth 16→32 | 15:18:33
hub-worker-2 | done-step | B: выравнивание θ-решётки под шаг поворота снижает T на 15-19% | 15:25:53
hub-worker-2 | done-step | B rollout на согласованном поле: 3/4 в цель, время 0.92T | 15:35:25
hub-worker-2 | done-step | B: зависание 4-го старта — локальный минимум интерполированного T; следующее — политика по рёбрам Беллмана | 15:43:17
hub-worker-2 | done-step | B: depth 4 не устраняет остаток 0.24 | 15:53:41
hub-worker-2 | step | по слову пользователя расчёты B на aida: diffdrive_5d_big.py n=16,24 (~/spore_v5, big16/24.log) | 15:54:43
hub-worker-2 | done-step | B на aida: dt 1→.5 T -0..12%; большое поле 3/5; общий ложный минимум (-¼,0,θ=0) | 15:58:17
hub-worker-2 | done-step | B: ложный минимум = боковой сдвиг+интерполяция; snap 5/6 в цель | 15:58:56
hub-worker-2 | done-step | B n=24 на aida: цена внутри поля не меняется, у границы падает | 16:10:15
hub-worker-2 | done | передача по слову пользователя | A2c, схема B 5D, aida-расчёты | next=hub-worker-3 | 16:19:26
hub-worker-2 | resumed | смена НЕ передана (слово пользователя: продолжать, hub-worker-3 ждёт конца работы) | 16:24:46
hub-worker-2 | done-step | B финиш 6/6: причина зависания — ничья ждать/ехать при depth>=2; цель-поглотитель + snap | 16:26:03
hub-worker-2 | done-step | B массовый тест 39/40; 50% узлов покоя недостижимы по чётности подрешётки | 16:41:18
hub-worker-2 | done-step | B nearest_reachable: 40/40 случайных стартов | 16:50:25
hub-worker-2 | done-step | B: старты с v,ω≠0: 38/40 | 17:00:29
hub-worker-2 | done-step | B rollout5_best (3+4): 40/40 старты с v,ω | 17:14:07
hub-worker-2 | done-step | B граничные старты 64/64 | 17:21:14
hub-worker-2 | done-step | A vs B время rollout: B/A 1.7-3.3 | 17:30:56
hub-worker-2 | done-step | атлас DI: запросы у линии vmax замерены | 17:40:48
hub-worker-2 | done-step | создана роль research: RESEARCH_ROUTINE.md, ROLES, ARCH, pulse.py (30 мин, Opus) | 17:42:19
hub-research-1 | started | acc3 | opus | 17:44:32
hub-worker-2 | done | слово пользователя: передать hub-worker-3 | схема B 5D, роль research, хвост DI | next=hub-worker-3 | 17:45:11
hub-worker-3 | started | acc3 | sonnet | 17:46:06
hub-worker-3 | done-step | B nearest (без интерполяции): в 10× быстрее, T(1,0)=2.0; вероятно оптимистичен, проверить rollout'ом | 18:28:29
hub-worker-3 | done-step | B nearest: rollout snap=0.5 точен, честный snap=0 — не доезжает у 3/5 → nearest занижает T, нужны рёбра точно в узел | 18:30:31
hub-worker-3 | step | B закрыт итогом, перехожу на v6 по слову пользователя (через hub-research-1) | 19:20:43
