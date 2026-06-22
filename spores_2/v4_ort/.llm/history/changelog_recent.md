# Changelog - Последние сессии

**Last updated:** 2026-06-23 (v4_ort сессия 2 / сессия 21)

> Хранит последние 5 сессий. Если сессий стало > 5 — самую старую перенести в конец [changelog_archive.md](changelog_archive.md)
> Полная история → [changelog_archive.md](changelog_archive.md)

---

## 2026-06-20 (v4_ort сессия 1 / сессия 20) - OrthoGrid: flow-orthogonal сетка

**Что сделано:**
- 🆕 `src/spores/ortho_grid.py` — класс `OrthoGrid`: flow-orthogonal сетка в фазовом пространстве
  - Три режима шага клонов: TIME (одинаковый dtau), ARC (match arc length центральной), ORTHO (optimize front ⊥ через scipy minimize_scalar)
  - `_ortho_step()` — шаг ⊥ фазовой скорости с smooth fallback вблизи equilibria (exp-blending)
  - `_integrate_by_arc()` — интеграция по arc length с sub-stepping
  - `_integrate_by_ortho()` — оптимизация ⊥ фронтов к вектору поля соседней траектории
  - Front lines на каждом time step, surfaces fwd/bwd (grid-триангуляция)
  - Alpha fade для крайних траекторий
  - Dirty flag: пересчёт при изменении look_point/a_max/tau/r_s
- 🔄 `src/math/double_integrator.py`, `src/math/pendulum.py` — добавлен `deriv(x,v,u) -> (fx,fv)` как общий интерфейс
- 🔄 `main.py` — два экземпляра OrthoGrid (grid_p +1, grid_m -1); _cycle_u() для переключения u_sign (+1/-1/±1); параметры n_s, r_s; клавиши 5-8
- 🔄 `src/core/input_manager.py` — поддержка новых биндингов

**Технические детали:**
- OrthoGrid заменяет BranchFamily как основная визуализация
- `deriv()` необходим для вычисления ⊥-направления (до этого модели имели только `step()`)
- r_s фиксирует разброс (ds = r_s / n_s); два экземпляра для ±u_sign
- scipy.optimize.minimize_scalar — зависимость для ORTHO режима

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
- 🆕 `AGENT_START_ROUTINE.md` — стартовая рутина: обработка handoff → state/ → changelog/decisions → git commit → токены
- 🔄 `AGENT_END_ROUTINE.md` — упрощён до handoff-only
- 🔄 `AGENT_START.md` — указывает на стартовую рутину; блок «СТОП — спроси токены»
- 🆕 `state/session_handoff.md`, обновлён `state/token_stats.md`

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-06-12 (v3 сессия 2 / сессия 16) - Pendulum + мягкая радуга + arrow_scale

**Что сделано:**
- 🆕 `src/math/pendulum.py` — класс `Pendulum`: RK4 (n_sub=10), `θ̈ = -sin(θ) + u`
- 🔄 `src/math/double_integrator.py` — единый интерфейс `step(x0,v0,u,dt) -> (x,v)`
- 🔄 `config/colors.json` — "мягкая радуга": 8 цветов node/edge/surface
- 🆕 `src/core/scalable_arrow.py` — `size_factor`
- 🔄 `src/core/line_manager.py` — `arrow_scale` + increase/decrease

**Участники:** Пользователь + Claude Sonnet 4.6

---

## 2026-05-27 (v3 сессия 1 / сессия 15) - ScalableArrow + семантические цвета + docs

**Что сделано:**
- 🆕 `src/core/scalable_arrow.py` — ScalableArrow: треугольник в середине линии, t_sign
- 🔄 `src/core/line_manager.py` — `create_arrow()`, alpha=1
- 🔄 `src/spores/boundary_ray_family.py` — per-segment цвет и t_sign, root lines → arrows, grid линии удалены
- 🔄 `config/colors.json` — 4 семантических цвета рёбер
- 🆕 `docs/reachability.md` — алгоритм + теория достижимости

**Участники:** Пользователь + Claude Sonnet 4.6

---

## Шаблон для новых записей

```markdown
## YYYY-MM-DD (сессия N) - Краткая тема

**Что сделано:**
- 🆕/🔄/🐛/🗑️ `файл` — что изменилось

**Участники:** Пользователь + Claude X
```
