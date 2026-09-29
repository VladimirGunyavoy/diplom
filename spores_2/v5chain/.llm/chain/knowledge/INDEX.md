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
