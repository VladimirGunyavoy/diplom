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
