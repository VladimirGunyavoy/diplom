# Окружение хаба для spores_2/v6 (пакеты, headless)
Перенесено из общего `~/claude-work/system/infra-rules/infra.md` (hub-v5chain-meta-1, 2026-10-01): это факты проекта, не общие правила.

## ursina на хабе gun-mid-us (hub-fixer-2, 2026-09-29, задача hub-worker-3 для spores_2/v6)
- Поставлено: `pip install --user --break-system-packages ursina` (PEP 668, numpy тоже в `~/.local`) → ursina 8.3.0, Panda3D 1.10.16. `import ursina` проходит (предупреждение OpenAL «No open device» безобидно).
- `Ursina()` НЕ создаётся: «No graphics pipe is available» — на хабе нет `libGL.so.1` (нужен `libpandagl.so`), нет Xvfb, `DISPLAY` пуст, `window_type='offscreen'` не помогает. Для проверки окна нужен `sudo apt install xvfb libgl1 libgl1-mesa-dri` (не ставилось — вне задачи «pip», sudo без пароля есть). Тогда: `xvfb-run -a python3 run.py`.

## scipy и прочие пакеты для spores_2/v6 на хабе (hub-fixer-2, 2026-09-29, задача hub-worker-3)
- Поставлено `pip install --user --break-system-packages scipy` → scipy 1.18.1 (`from scipy.optimize import minimize_scalar` ок). Прочих недостающих пакетов нет: все импорты `main.py` проходят. `IPython` не установлен, но нужен только внутри `try` в `src/utils/nb_logger.py` (не критично). `utils` = `src/utils` (`run.py` кладёт `src` в sys.path).
- `Ursina(window_type='none')` создаётся (параметр — аргумент конструктора; `Ursina()` из `main.py:52` вызывается без него, поэтому без дисплея падает). Сборка сцены `main.py` с подменой на 'none' доходит до `FirstPersonController` и падает: `mouse.locked = True` → `application.base.win` = None (нет окна). Это код v6, не пакет: для headless-проверки нужен флаг/обход в v6 (не создавать FirstPersonController при `application.window_type == 'none'`) либо xvfb (см. раздел про ursina выше). Скрипт проверки: `/tmp/claude-1000/v6chk/chk.py` (подменяет Ursina на window_type='none', `app.run()` пропускает).
