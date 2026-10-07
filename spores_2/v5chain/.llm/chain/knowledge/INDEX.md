# knowledge/INDEX — вопрос → файл → раздел
| тема | где |
|---|---|
| решения пользователя по проекту | `user_decisions.md` |
| решения пользователя по цепочке (общие) | `~/claude-work/system/llm/user_decisions.md` |
| как менять документацию и протокол (сверять каждую правку) | `~/claude-work/system/llm/DOC_CHANGES.md` |
| общие грабли (tmux, crontab, pick_account, воскрешение сессии) | `~/claude-work/system/infra-rules/infra.md` |
| машины: hub, aida, yoga (железо, доступ, грабли) | `~/claude-work/system/infra/machines/<машина>/README.md` |
| пакеты и headless-запуск v6 на хабе (ursina, scipy) | `../../context/hub_env.md` |
| память ролей (хендоффы, последние смены) | `../../roles/README.md` |
| код: архитектура, менеджеры, ввод | `../../context/architecture.md`, `../../context/input_manager_guide.md` |
| атлас спор (двойной интегратор → дифдрайв) | `../../../docs/spore_atlas_double_integrator.md` |
- [concepts.md](concepts.md) — **цель проекта и концепции/ограничения пользователя одним списком (правило · источник); читают на старте все роли, сверяют результаты**
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
- [research/branch2pi.md](research/branch2pi.md) — ветвь 2π в 6D вверх: только 2/5 худших (q14, q9, +22–26%); 3/5 — топология внутри ветви (+22–31%)
- [research/blind_di_heuristic.md](research/blind_di_heuristic.md) — слепая DI-эвристика V: допустима, h/V медиана 0.41/0.59 (маятник u .3/.5); A* с весом 1.5–3
- [research/v7_faces.md](research/v7_faces.md) — v7 п.3 без бокового перехода: общая сеть сечений (клетка = грань × слой, V на гранях); DI T/T* 1.137→1.011 (h .4→.05), O(h); нужна адаптивность граней
- [research/lit_bangbang_cost.md](research/lit_bangbang_cost.md) — литобзор: bang-bang дифдрайва (Balkcom–Mason 2002: все дуги на вершинах), L1/L2-цена (bang-off-bang / насыщ.-непрерывное) (searcher-1)
- [research/pend_faces_front.md](research/pend_faces_front.md) — маятник на гранях: 60% «недостижимых» — застой фронта строгой интерполяции, не физика; BIG-конечное → 99–100%
- [research/v7_spore_cells.md](research/v7_spore_cells.md) — v7 клетки из точных траекторий + гало 10% + свободное переключение: DI 100%, T/T* 1.02–1.04; оценка только после шага Δt (без 0-переходов)
- [research/v7_spectrum_cost.md](research/v7_spectrum_cost.md) — спектр на клетках v7: при T убирает дребезг (3→1 переключ.), при T+ρ∫u² bang-bang хуже на 12–46%, спектр 1.025 от ПМП-оптимума
- [research/exact_switch.md](research/exact_switch.md) — точный момент переключения: событие нуля σ = p·Δf по сопряжённой p + финиш из ≤2 дуг; маятник 1 переключение вместо 40, T −6%; при ошибке модели нужен пересчёт; базис 2-го порядка (идея пользователя) Ошибка модели u·.95: оценка коэффициента на ходу + приём ±2% — .955, max 1.03 (research-9).
- [research/pend_characteristics.md](research/pend_characteristics.md) — характеристики ПМП маятника: кривые переключения без V; V «фронтом» за 2–7 с, вилка эталона 2.7%, не строгий эталон (барьеры: потери ветвей / занижение до ×1.8)
- [research/butterfly_spores.md](research/butterfly_spores.md) — споры-«бабочки» (идея пользователя): нормальный сегмент (ядро Dᵀ, D = [∂Φ/∂t, ∂Φ/∂u]) × t × все u; ДИ 1.05 (мед. 1.03), 1–1.5 перекл., в 7× меньше узлов, чем клетки v7; 4D плоский ДИ — прототип Маятник (research-9): 5000 спор — 100%, T/лучшее известное мед. 1.03, 1.2 перекл.; дугу ограничивать окном, не временем.
- [research/butterfly_dd.md](research/butterfly_dd.md) — бабочки на дифдрайве: спора = точка × круг курсов, дуги формулой; 4000 спор — 100%, T/эталон 1.064 (мед. 1.055)
- [research/butterfly_dp.md](research/butterfly_dp.md) — бабочки на двойном маятнике: споры на трубках + прямое/обратное дерево + эллипс + агент мини-дерево (g = 1 ×1.054 OCP); g = 2 связан Вороным + «ленивыми узлами» (§research-11, T 20.9 = ×2.74)
- [research/grow_cells.md](research/grow_cells.md) — растущие споры-клетки, атлас на каждое управление, посев встык + обязательные пересечения: ДИ 258 спор T/T* 1.023, маятник 282 споры T/эталон 1.053 (мед. 1.018), адаптация размера по форме среза
- [research/cell_metric.md](research/cell_metric.md) — размер клетки по метрике грамиана («что управление исправит за время клетки»), STEPS, REFINE; хвост маятника = разрыв V внутри клетки, JUMP-заплатки неустойчивы
- [research/dp_energy_gap.md](research/dp_energy_gap.md) — g = 2, новый старт не связывается: деревья разделены по ЭНЕРГИИ (0 годных пар из 36–41 тыс.), у фронта 56% дуг веера нарушают |ω| ≤ 3; цели Вороного в полосе разрыва (EGAP) связали q8 за +2400 спор (база — нет и за +6000)
- [research/dp_corridor_refine.md](research/dp_corridor_refine.md) — g = 2: путь OCP чистый bang-bang (нет сингулярной дуги), потолок K дуг, запас |u| .9 стоит 16%; путь хода по рёбрам + доводка коридора IPOPT → T 7.6485 = ×1.0017 OCP (+ поиск топологий: лучший на атлас ×1.002 ×2, ×1.066 ×2); g = 1 ×1.01, g = .3 ×1.000 (research-12)
- [research/dp_ocp_ref.md](research/dp_ocp_ref.md) — эталон OCP двойного маятника висит→вверх: g=1 5.098 (v6 1.036), g=2 при |ω|≤3 7.636 — решаем
- research/cut_walls.md — скопление спор у цели = гребёнка стен CUT; CUTR (бисекция по траекториям), OWN, слабый мотор u .15 и эталон энергия+стрельба (research-15)
- [research/grow3_dd.md](research/grow3_dd.md) — grow3 дд: отказы = дребезг у цели → финиш стрельбой ≤3 дуг (60/60); строки RS 3 + петля на себя в solve (×2.6 узлов, ×2.8 быстрее); эталон в коробку dd_refbox_60 (точка дороже на ~4%) (research-17)
- [research/growN_4d.md](research/growN_4d.md) — growN 4D: covered() верен; число клеток = объём поля / объём клетки (~800/слой DI 4D), FRAC 1 ×10 меньше, GM в 4D ×3.8 объёма (research-17)
- [research/ellipse_scaling.md](research/ellipse_scaling.md) — выход из экспоненты по n: атлас под запрос — эллипс T_s+V ≤ (1+ε)C (доля поля 7e-3 в 4D, 3e-4 в 6D); двунапр. рост накрывает его без эвристики; грубая оценка не годится (√d у концов) (research-18)
- [research/lit_separatrix_detection.md](research/lit_separatrix_detection.md) — литобзор: автопоиск разрывов V/сепаратрис (ПМП-экстремали, эпиграф, полиномиальная аннигиляция; nD-метода с гарантиями нет; PDF не прочитаны) (searcher-1)
- [research/nf_curved_ends.md](research/nf_curved_ends.md) — кривые торцы клеток (NORMFRONT=2: ломаная по нормалям клонов) + изгиб DELTA: 1.0085 без стен; мёртвые узлы NF, кольца (SELFOV), адаптивные клоны
- [research/solve_bucket.md](research/solve_bucket.md) — solve по вёдрам V (коррекция меток в порядке V) вместо Якоби: та же V, рёбра считаются 2–3 раза вместо ~Vmax/DTN; маятник проход ×40, solve ×7 (research-21)
- [research/manip3d_2link.md](research/manip3d_2link.md) — 3D двузвенный манипулятор (основание+плечо+локоть, 6D, без g): модель точечных масс, сводится к плоскому 2-зв. v6; эталон OCP r21/ref_m3d.py; коридор v6 (research-21)
- [research/static_car.md](research/static_car.md) — «статик кар» (машинка-велосипед из journal-2026: (x,y,θ,v), a и ω, 3 диска, цель-круг): модель, границы, эталон (research-21)
- [research/gpu_solve.md](research/gpu_solve.md) — solve атласа на GPU aida: тот же Якоби di4 301 с → 6.0 / 2.4 с (f64/f32), V та же; почему вёдра и итерация по политике в 4D не работают (92% лучших стенсилов непричинны); неединственность V при PESS — гипотеза (research-22)
