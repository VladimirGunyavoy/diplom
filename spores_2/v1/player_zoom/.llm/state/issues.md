# Known Issues - Известные проблемы

**Last updated:** 2026-03-19

---

## 🔴 Критические

*Нет критических проблем.*

---

## 🟡 Важные

### ~~Issue #1: MyObject.real_position не обновляется при анимации~~ ✅ ИСПРАВЛЕНО

**Исправлено:** 2026-03-19 (сессия 3)

**Решение:** Комбинация двух изменений:
1. `my_object.py`: `update_position()` теперь обновляет `self.real_position = np.array(self.position)` после вычисления новой позиции
2. `update_manager.py`: после `my_object.update_position(dt)` вызывается `my_object.apply_transform(a, b)` с текущими параметрами зума

Это обеспечивает корректное поведение MyObject при активном зуме: объект движется по кругу И масштабируется/сдвигается по текущей zoom-трансформации

---

### ~~Issue #2: Отсутствует config/colors.json~~ ✅ ИСПРАВЛЕНО (сессия 11)

**Описание:**
`ColorManager` пытается загрузить цвета из `config/json/colors.json`, но файл не существует.

**Код:**
```python
# src/color_manager.py
def load_colors_from_json(self, path='config/json/colors.json'):
    if os.path.exists(path):
        # ...
    else:
        print(f"[ColorManager] Colors file not found: {path}")
        # ...
```

**Текущее поведение:**
- Выводится warning
- Используются дефолтные цвета
- Проект работает нормально

**Последствия:**
- Warning в консоли при каждом запуске
- Невозможно кастомизировать цвета через JSON

**Решения:**

*Вариант 1:* Создать colors.json с дефолтными цветами
```bash
mkdir -p config/json
# создать файл с DEFAULT_COLORS
```

*Вариант 2:* Удалить попытку загрузки JSON (упростить)
```python
# Всегда использовать дефолты, убрать JSON логику
```

*Вариант 3:* Сделать путь опциональным параметром
```python
ColorManager(colors_path=None)  # None = использовать дефолты
```

**Приоритет:** 🟢 Низкий (не мешает работе)

**Назначено:** Не назначено

---

## 🟢 Незначительные

### Issue #3: Нет обработки ошибок при загрузке arrow.obj

**Описание:**
`Frame` пытается загрузить модель `arrow.obj` без проверки существования файла.

**Код:**
```python
# src/frame.py
self.x_axis = Scalable(
    parent=self,
    model='arrow.obj',  # ← Если файла нет - краш
    color=self.color_manager.get_color('frame', 'x_axis'),
    rotation=(0, 0, 90)
)
```

**Риск:**
Если `assets/arrow.obj` удален - программа упадет с ошибкой.

**Решение:**
Добавить fallback на примитивную модель:
```python
try:
    model = 'arrow.obj'
    self.x_axis = Scalable(parent=self, model=model, ...)
except:
    # Fallback на cube или cylinder
    model = 'cube'
    self.x_axis = Scalable(parent=self, model=model, ...)
```

**Приоритет:** 🟢 Низкий (файл существует)

**Назначено:** Не назначено

---

### Issue #4: Magic numbers в коде

**Описание:**
Много hardcoded значений без объяснения.

**Примеры:**
```python
# src/zoom_manager.py
self.zoom_fact = 1 + 1/8  # Почему 1/8?

# main.py
origin_scale=0.05  # Почему именно 0.05?
scale=40  # Размер пола - почему 40?
test_object_1 = Scalable(scale=1/10)  # Почему 1/10?
```

**Проблема:**
Непонятно почему выбраны эти значения. Сложно настраивать.

**Решение:**
Вынести в константы с комментариями:
```python
# Zoom configuration
ZOOM_STEP_FACTOR = 1.125  # 1 + 1/8, chosen for smooth zoom feel
```

**Приоритет:** 🟢 Низкий (качество кода, не баг)

**Назначено:** Не назначено

---

### Issue #5: Print вместо logging

**Описание:**
Везде используется `print()` для debug вывода.

**Проблема:**
- Невозможно отключить debug вывод
- Невозможно фильтровать по уровню (INFO/DEBUG/ERROR)
- Невозможно перенаправить в файл

**Решение:**
Заменить на logging:
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Zoom in")
```

**Контр-аргумент:**
Это песочница, не production код. Print проще и быстрее.

**Приоритет:** 🟢 Низкий (философский вопрос)

**Назначено:** Не назначено

---

### ~~Issue #6: Spore.position не применяется — спора всегда в (0,0,0)~~ ✅ ИСПРАВЛЕНО

**Исправлено:** 2026-04-19 (сессия 8)

**Решение:** Опечатка в `main.py` — передавалось `positions=(1,1)` вместо `position=(1,1)`. Из-за этого `Spore.__init__` получал `position=None` и использовал `(0, y_offset, 0)`. Исправлена одна буква.

---

## 🔵 Потенциальные проблемы (не подтверждены)

### Potential #1: Edge case в identify_invariant_point()

**Описание:**
```python
if abs(np.tan(phi)) < 0.001:
    return 0, 0
```

Если камера смотрит почти горизонтально - возвращает (0, 0). Это может быть неожиданным поведением.

**Альтернатива:**
Возвращать текущую позицию камеры проецированную на плоскость?

**Статус:** Требует исследования

---

### Potential #2: Ursina Entity удаление

**Описание:**
В `ZoomManager.update_transform()` есть try-except для invalid entities:

```python
try:
    if hasattr(obj, 'enabled') and obj.enabled:
        obj.apply_transform(...)
except (AssertionError, AttributeError, RuntimeError):
    continue
```

**Вопрос:** Когда и почему entities становятся invalid?

**Статус:** Требует исследования

---

## 📋 Шаблон для новых issues

```markdown
### Issue #N: Краткое название

**Описание:**
Детальное описание проблемы.

**Как воспроизвести:**
1. Шаг 1
2. Шаг 2

**Ожидаемое поведение:**
Что должно происходить.

**Фактическое поведение:**
Что происходит на самом деле.

**Код (если применимо):**
```python
# Проблемный код
```

**Возможные решения:**
- Вариант 1
- Вариант 2

**Приоритет:** 🔴/🟡/🟢 (Критический/Важный/Незначительный)

**Назначено:** Имя или "Не назначено"

**Связанные issues:** #N, #M
```

---

## 🏷️ Теги

- `bug` - баг в коде
- `enhancement` - улучшение существующего
- `question` - требует исследования
- `documentation` - проблема в документации
- `wontfix` - не будет исправлено (by design)

---

**Итого:** 5 известных issues (0 критических, 1 важный, 3 незначительных, 1 исправлен)

**См. также:** `llm/review_and_ideas.md` — дополнительные проблемы и идеи (мёртвый код в SceneSetup, identify_invariant_point вызывается впустую, import time magic)
