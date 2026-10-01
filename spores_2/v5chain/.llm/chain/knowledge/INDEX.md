# knowledge/INDEX — вопрос → файл → раздел
| тема | где |
|---|---|
| решения пользователя по проекту | `user_decisions.md` |
| решения пользователя по цепочке (общие) | `~/claude-work/system/llm/user_decisions.md` |
| машины, VM, SSH, tmux, Colab, грабли | `~/claude-work/system/infra-rules/infra.md` |
| код: архитектура, менеджеры, ввод | `../../context/architecture.md`, `../../context/input_manager_guide.md` |
| атлас спор (двойной интегратор → дифдрайв) | `../../../docs/spore_atlas_double_integrator.md` |
- [atlas_di_tails.md](atlas_di_tails.md) — хвосты атласа DI: запросы у линии vmax
- [research/README.md](research/README.md) — заметки роли research (рисерч, математика)
- [research/di_gradV_switch.md](research/di_gradV_switch.md) — агент по ∇V из 5 точек клетки (DI): мелкие клетки 20/20, крупным нужна клетка цели
- [research/di_hessian_density.md](research/di_hessian_density.md) — точки сечения по гессиану V: ошибка −15…−54% при том же N (DI)
- [research/v6_value_tau.md](research/v6_value_tau.md) — V по графу клеток v6: занижение от τ=h/2, лечение τ≈√h (обратный CFL)
- [research/pendulum_plan.md](research/pendulum_plan.md) — маятник для v6: u_max 0.3/0.5, цилиндр, растяжение у седла, эталон мелкой сеткой
- [research/diffdrive_v6_plan.md](research/diffdrive_v6_plan.md) — дифдрайв v6: клетки слоя = копии шаблона (SE(2)), U ромб + эталон Balkcom–Mason
- [research/manipulator_plan.md](research/manipulator_plan.md) — манипулятор v6: 5 ступеней (кинематика T² с эталоном → динамика 6D), препятствия без C-space
- [research/pendulum_lqr_goal.md](research/pendulum_lqr_goal.md) — LQR-клетка цели наверху маятника: R=10, эллипс 1.5·c, |φ| до 0.66/1.11 рад при u=.3/.5
- [research/ndim_plan.md](research/ndim_plan.md) — n-мерные: тройной интегратор (3D) и плоский DI |a|≤1 с K направлениями (потеря ≤1/√cos(π/K))
- [research/dd_rhombus_ref.md](research/dd_rhombus_ref.md) — эталон дифдрайва (ромб-U): min(TGT, TGTGT), TGT завышает боковые сдвиги до 37%
- [research/dd_atlas_refine.md](research/dd_atlas_refine.md) — атлас дифдрайва: мельчить (x,y), не θ; окно цели — физическое; эталон «до окна»
- [research/adaptive_vs_grid.md](research/adaptive_vs_grid.md) — v6 ушёл в равномерную сетку: эксперимент «адаптивный атлас vs сетка» (H1), цепочки спор (H2)
- [research/docking_nd.md](research/docking_nd.md) — стыковка в nD: листы из семейств обратных цепочек от границы окна цели + проверка проигрышем
- [research/multiquery_corridor.md](research/multiquery_corridor.md) — многозапросность + коридор на дереве: NB 1200 + NF 10 → 30/30, T/эталон 1.004 (дд 3D); топологии с промахом, top-5, SLSQP
- [research/manip_dyn_corridor.md](research/manip_dyn_corridor.md) — манипулятор 4D: коридор 5/8 → 8/8 при NB 2400/NF 400, T = эталону перебором топологий
- [research/query_speed.md](research/query_speed.md) — почему запросы медленные: SLSQP зовёт numpy-rk4 точкой (×13–22 на math), maxiter 80, зазор последовательно; хаб ≈10× медленнее aida (VM); манипулятор 124 с → 1.2 с
- [research/tree_directed_6d.md](research/tree_directed_6d.md) — прямое дерево с A* по обратному (g+3h внутри уровня переключений): 6D+g NF800 решает q1 (как база NF3000), NB6000 решает q2
- [research/ref_6d.md](research/ref_6d.md) — эталон T 6D: OCP CasADi/IPOPT мультистарт (ref6d_ocp.py, ref6d_T.json, 32 запроса вниз/вверх); 32 запроса: T_corr/T_ref вниз мед. 1.018, вверх мед. 1.036 (макс 1.36, ~1/3 — не та топология); refine не держит |w|≤WM
- [research/pend_energy_vs_opt.md](research/pend_energy_vs_opt.md) — маятник из низа: энергонакачка+LQR 13.8/9.66 с против оптимума 12.32/7.44 (u=.3/.5): +12%/+30%, не эталон
- [research/explanation_log.md](research/explanation_log.md) — ЖУРНАЛ РАССКАЗА пользователю: шаги 1–12 сделаны, следующий — 13 манипуляторы
- [research/cell_model_size.md](research/cell_model_size.md) — размер клетки v7 с локальной моделью: квадратичная + tol 1e-2 → τ=1 почти везде; манипуляторы при tol 1e-3 — r=0.1
- [research/lit_control_spectrum.md](research/lit_control_spectrum.md) — литобзор «спектр управлений»: Chen–Fliess/зонотопы, воронки, примитивы, сингулярные дуги, sum-up rounding (searcher-1)
- [research/control_spectrum.md](research/control_spectrum.md) — спектр управлений: образ U за τ по 3 слоям/канал (+диагональ при G(x)); у цели спектр+LQR держит (остаток ×100–1000 меньше вершин), в быстродействии выигрыша нет
