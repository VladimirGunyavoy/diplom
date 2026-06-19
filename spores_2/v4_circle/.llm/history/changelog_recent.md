# Changelog - Последние сессии

**Last updated:** 2026-06-20 (v4_circle сессия 2 / общая сессия 21)

> Хранит последние 5 сессий. Если сессий стало > 5 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

---

## 2026-06-17 (v4_circle сессия 1 / сессия 20) - CirclePattern + ScalableTipArrow

**Что сделано:**
- 🆕 `src/spores/circle_pattern.py` — класс `CirclePattern`: n_circle спор по кругу радиуса radius вокруг look_point. 3 группы управлений (u=-a_max, 0, +a_max) через `_ControlGroup`, каждая с derivative-стрелками (ScalableTipArrow) и траекториями интегрирования (n_tau сегментов). `toggle_group(idx)` переключает видимость группы
- 🆕 `src/core/scalable_tip_arrow.py` — класс `ScalableTipArrow` (наследник ScalableLine): треугольник-наконечник на конце (p2), а не в середине как ScalableArrow. `__setattr__` синхронизирует color/enabled
- 🔄 `src/core/line_manager.py` — метод `create_tip_arrow()`
- 🔄 `src/math/pendulum.py`, `src/math/double_integrator.py` — добавлен `derivative(x, v, u)` — вычисляет (dx/dt, dv/dt) напрямую, без step() с dt→0
- 🔄 `main.py` — v4_circle: CirclePattern, параметры radius/n_circle/a_max/tau/n_tau (клавиши 1-6), ctrl+1/2/3 toggle групп управления
- 🔄 `config/colors.json` — цвета `circle/node`, `circle/edge`, `circle/arrow_neg/zero/pos`

**Технические детали:**
- Длина стрелки: `r/3 * tanh(3/r * ||f||)` — мягкое насыщение, не выходит за r/3
- Ursina: `held_keys['control']` всегда 0 — использовать `'left control'`/`'right control'`
- `_rebuild()` вызывается автоматически при изменении n_circle или n_tau

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-06-12 (v3 сессия 5 / сессия 19) - Toggle клавиш 1-8 → control+1-8

**Что сделано:**
- 🔄 `main.py` — toggle видимости веток перенесён с клавиш `1`-`8` на `control+1`-`control+8`, чтобы не конфликтовать с параметрами 1-4 (resize/tau/a_max/n_tau)
- 🔄 `src/core/input_manager.py` — обработка ctrl-комбо через `held_keys['left control'] or held_keys['right control']` (в Ursina `held_keys['control']` всегда 0)

**Технические детали:**
- В Ursina события ctrl приходят как `'left control'`/`'right control'`, а не `'control'` — `held_keys['control']` всегда 0
- Рабочие файлы были не закоммичены на момент handoff

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-06-12 (v3 сессия 4 / сессия 18) - Toggle видимости веток (клавиши 1-8)

**Что сделано:**
- 🔄 `src/spores/boundary_ray_family.py` — `_BranchRay.set_visible()`; `BranchFamily.set_visible()/toggle()` управляет видимостью спор, рёбер-стрелок, root-линий, envelope и fan_surface ветки целиком
- 🆕 `src/core/line_manager.py` — `set_visible(name, visible)`
- 🆕 `src/core/surface_manager.py` — `toggle(name)`, `set_visible(name, visible)`
- 🔄 `main.py` — все 8 веток (`b_ppp` ... `b_mmm`) присвоены переменным при создании; клавиши `1`-`8` (mode='press') toggle'ят видимость соответствующей ветки

**Технические детали:**
- `BranchFamily._visible` сохраняется как состояние и переустанавливается в конце `_build()` (`set_visible(self._visible)`) — видимость ветки переживает rebuild при смене `n_tau`
- Toggle на press не конфликтует с существующими `1`-`4` на mode='scroll' (resize/tau/a_max/n_tau при удержании+скролле)

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-06-12 (v3 сессия 3 / сессия 17) - Миграция .llm протокола v1 → v2 (Decision #20)

**Что сделано:**
- 🆕 `AGENT_START_ROUTINE.md` — новый файл: обработка handoff → обновление state/ → запись в changelog/decisions → git commit → токены. Выполняется в начале сессии, пока контекст почти пуст
- 🔄 `AGENT_END_ROUTINE.md` — упрощён с 209 строк до handoff-only: спросить токены → записать `session_handoff.md` с пометкой `[РУТИНА НЕ ВЫПОЛНЕНА]` → закоммитить только его → спросить токены
- 🔄 `AGENT_START.md` — указывает на новую стартовую рутину; убрана дублирующая секция про токены; добавлен блок «СТОП — спроси токены первым делом»: T_START спрашивается ДО любых tool calls (даже если файлы уже прочитаны в этом батче)
- 🔄 `AGENT_START_ROUTINE.md` (шаг 0) — уточнён аналогично: спросить токены сразу после AGENT_START.md, до handoff, не параллельно с другими tool calls (иначе вопрос теряется среди вызовов)
- 🆕 `state/session_handoff.md` — новый файл, в конце сессии помечается `[РУТИНА НЕ ВЫПОЛНЕНА]`, обрабатывается стартовой рутиной следующей сессии
- 🔄 `state/token_stats.md` — новая схема на 4 точки (T_START/T_AFTER/T_BEFORE_END/T_END), сессии 9-16 переразмечены (T_AFTER = T_START, т.к. в v1 рутина была только в конце)
- 🔄 `history/decisions.md` — добавлен Decision #20 (разделение рутины)
- 🔄 `history/changelog_recent.md` — уточнено правило ротации: хранит 5 сессий

**Технические детали:**
- Причина: в v1 финишная рутина (changelog, decisions, state/, git commit) выполнялась поверх заполненного контекста сессии — стоила несоразмерно дорого (сессия 16: Работа=8%, Рутина↓=5%)
- v2: тяжёлая часть переехала в начало следующей сессии (контекст почти пуст), финишная рутина = только запись handoff (1 файл, 1 коммит)
- Задач разработки v3 (код player_zoom) в этой сессии не было — чисто протокольная миграция

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-06-12 (v3 сессия 2 / сессия 16) - Pendulum + мягкая радуга + arrow_scale

**Что сделано:**
- 🆕 `src/math/pendulum.py` — класс `Pendulum`: RK4 (n_sub=10) для `θ̈ = -sin(θ) + u`, `θ=0` — нижнее устойчивое положение, нормировка g/l=1, m·l²=1; векторизован (np.sin поэлементно), интерфейс `step(x0,v0,u,dt) -> (θ,ω)`
- 🔄 `src/math/double_integrator.py`, `src/math/__init__.py` — `DoubleIntegrator.step()` теперь возвращает `(x,v)` (был `np.array`), убран неиспользуемый `import numpy` — единый интерфейс с `Pendulum`
- 🔄 `src/spores/boundary_ray_family.py` — `_recompute()` зовёт `self._ctx.model.step(x, v, u, dt)` вместо инлайн-формулы double integrator; убран прямой импорт `_step`
- 🔄 `main.py` — новая секция `DYNAMICS MODEL`: `model = Pendulum(shared_context)`, `shared_context.bind('model', ...)` — переключение динамики = одна строка
- 🔄 `config/colors.json` — новая палитра "мягкая радуга": `node/edge/surface_{ppp..mmm}` — 8 цветов равномерно по кругу оттенков (шаг 45°, S≈0.55, V≈0.82), единый RGB на ветку с alpha 1.0/0.6/0.4; 4 цвета фазовых рёбер (`edge_fwd/bwd_pos/neg`) приведены к той же тональности
- 🆕 `src/core/scalable_arrow.py` — `size_factor` (default 1.0): `size = length/4 * size_factor`
- 🔄 `src/core/line_manager.py` — `arrow_scale` + `increase/decrease_arrow_size()`, применяется в `create_arrow()` и ко всем существующим стрелкам
- 🔄 `main.py` — `_resize()`: клавиша `1` (scroll/Q-E) меняет `spore_manager` и `line_manager.arrow_scale` одновременно
- 🔄 `src/core/input_manager.py` — Q/E дублируют логику `scroll up/down` (held-keys 1-4 приоритетнее зума)

**Технические детали:**
- Маятник: `θ̈ = sin(θ) + u` соответствует θ=0=верх (неустойчиво); для θ=0=низ (устойчиво) знак гравитационного члена меняется на `-sin(θ)`
- Класс был изначально `InvertedPendulum`, переименован в `Pendulum` после смены конвенции (при θ=0=низ это уже не "перевёрнутый" маятник)
- `model.step()` — общий интерфейс для `DoubleIntegrator`/`Pendulum`, диспетчеризуется через `shared_context.model`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```
