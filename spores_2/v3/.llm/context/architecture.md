# Architecture - Детальное описание архитектуры

**Last updated:** 2026-03-19

---

## 🏗️ Архитектурные паттерны

### 1. Manager Pattern

Все менеджеры являются **координаторами** для своей области ответственности:

| Manager | Ответственность | Зависимости |
|---------|----------------|-------------|
| `ColorManager` | Загрузка и предоставление цветов | Нет |
| `WindowManager` | Управление окном, мониторами, fullscreen | Ursina window |
| `InputManager` | Централизованная обработка ввода | Все компоненты (через DI) |
| `UpdateManager` | Координация update() всех компонентов | Все компоненты (через DI) |
| `ZoomManager` | Управление масштабированием сцены | SceneSetup, ColorManager |

**Важно:** менеджеры создаются **независимо**, компоненты регистрируются **после создания**.

---

### 2. Dependency Injection (DI)

**Проблема:** циркулярные импорты при сложных зависимостях.

```python
# ❌ Плохо (циркулярный импорт):
# input_manager.py
from .zoom_manager import ZoomManager  # импорт zoom_manager

# zoom_manager.py
from .input_manager import InputManager  # импорт input_manager
```

**Решение:** компоненты не импортируют друг друга, а получают через регистрацию:

```python
# ✅ Хорошо:
# input_manager.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .zoom_manager import ZoomManager  # только для type hints

class InputManager:
    def __init__(self):
        self.zoom_manager: Optional["ZoomManager"] = None

    def register_zoom_manager(self, zm: "ZoomManager"):
        self.zoom_manager = zm

# main.py
input_manager = InputManager()
zoom_manager = ZoomManager(...)
input_manager.register_zoom_manager(zoom_manager)  # ← DI
```

---

### 3. Scalable Pattern

**Идея:** все объекты сцены, которые должны масштабироваться при зуме, наследуются от `Scalable`.

```python
class Scalable(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем оригинальные значения
        self.real_position = np.array(self.position)
        self.real_scale = np.array(self.scale)

    def apply_transform(self, a: float, b: np.ndarray):
        """Аффинное преобразование: x' = a*x + b"""
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a
```

**Использование:**
1. Объект регистрируется в `ZoomManager`
2. При зуме `ZoomManager` вызывает `apply_transform()` у всех объектов
3. Объект пересчитывает свою позицию/масштаб от оригинальных значений

**Важно:** трансформация всегда применяется к **оригинальным** (`real_*`) значениям, а не к текущим.

---

## 🔄 Потоки данных

### Input Flow

```
┌─────────────────────────────────────────┐
│  User presses key (e.g., 'e' for zoom)  │
└──────────────────┬──────────────────────┘
                   ▼
          ┌────────────────────┐
          │   main.py: input() │ ◄─── Ursina global handler
          └────────┬───────────┘
                   ▼
      ┌────────────────────────────┐
      │ InputManager.handle_input() │
      └────┬───────────────────────┘
           │
           ├─→ key == 'e' ──→ zoom_manager.zoom_in()
           ├─→ key == 'alt' ──→ scene_setup.toggle_freeze()
           ├─→ key == 'f11' ──→ window_manager.toggle_fullscreen()
           ├─→ key == '1' ──→ my_object.decrease_speed()
           └─→ key == 'h' ──→ _print_debug_info()
```

---

### Update Flow

```
┌──────────────────────────────────┐
│  Ursina main loop (every frame)  │
└────────────┬─────────────────────┘
             ▼
    ┌────────────────────┐
    │  main.py: update() │
    └────────┬───────────┘
             ▼
    ┌─────────────────────────┐
    │ UpdateManager.update_all(dt) │
    └────┬────────────────────┘
         │
         ├─→ input_manager.update()
         │       └─→ (пока пустой, резерв для per-frame input logic)
         │
         ├─→ scene_setup.update(dt)
         │       └─→ player.y += (Space - Shift) * speed * dt
         │
         ├─→ my_object.update_position(dt)
         │       └─→ angle += speed * dt
         │       └─→ position = (r*cos(angle), 0, r*sin(angle))
         │
         └─→ zoom_manager.identify_invariant_point()
                 └─→ calculate look point (ray-plane intersection)
```

---

### Zoom Flow

```
User presses 'e' (zoom in)
    ▼
InputManager.handle_input('e')
    ▼
ZoomManager.zoom_in()
    ▼
ZoomManager.change_zoom(+1)
    │
    ├─→ 1. Calculate invariant point (look point)
    │       inv = identify_invariant_point()
    │       → ray-plane intersection (y=0)
    │
    ├─→ 2. Update transformation parameters
    │       zoom_multiplier = zoom_fact^sign = 1.125^1 = 1.125
    │       a *= zoom_multiplier
    │       b = zoom_multiplier * b + (1 - zoom_multiplier) * inv
    │
    └─→ 3. Apply to all registered objects
            for obj in objects:
                obj.apply_transform(a, b)
                    → position = real_position * a + b
                    → scale = real_scale * a
```

---

## 📦 Компоненты

### SceneSetup

**Ответственность:**
- Создание и настройка камеры (FirstPersonController)
- Освещение сцены (DirectionalLight, AmbientLight)
- Управление курсором (locked/unlocked)

**Ключевые методы:**
- `toggle_freeze()` - переключение захвата курсора
- `update(dt)` - обновление вертикального движения (Space/Shift)

**Особенности:**
- `input_frozen` флаг - если True, движение заблокировано (курсор свободен)
- `input_manager_mode` - если True, input обрабатывается InputManager'ом

---

### ZoomManager

**Ответственность:**
- Управление масштабированием сцены
- Вычисление инвариантной точки
- Регистрация и обновление Scalable объектов

**Состояние:**
```python
self.zoom_fact = 1 + 1/8 = 1.125  # Множитель зума за один шаг
self.a_transformation = 1.0        # Текущий коэффициент масштаба
self.b_translation = [0, 0, 0]     # Текущий вектор сдвига
self.objects = {}                  # Зарегистрированные объекты
```

**Ключевые методы:**
- `register_object(obj, name)` - регистрация Scalable объекта
- `identify_invariant_point()` - вычисление точки взгляда (ray-plane intersection)
- `change_zoom(sign)` - изменение зума (sign: +1 = in, -1 = out)
- `reset_all()` - сброс к начальному состоянию

**Математика инвариантной точки:**
```python
h = camera.y           # высота камеры
psi = camera.yaw       # азимут (rotation_y)
phi = camera.pitch     # наклон (camera_pivot.rotation_x)

d = h / tan(phi)       # расстояние до точки на плоскости y=0
x_0 = camera.x + d * sin(psi)
z_0 = camera.z + d * cos(psi)

return (x_0, z_0)
```

---

### Frame

**Ответственность:**
- Визуализация координатной системы
- 4 элемента: origin_cube (начало координат) + 3 оси (X, Y, Z)

**Особенности:**
- Все элементы Frame - это `Scalable` объекты
- Регистрируются в ZoomManager индивидуально
- Можно скрыть/показать через `toggle_visibility()`

---

### InputManager

**Ответственность:**
- Централизованная обработка всего пользовательского ввода
- Делегирование команд соответствующим компонентам

**Особенности:**
- Не обрабатывает ввод если `scene_setup.input_frozen == True`
- Есть debug режим (H) - показывает состояние всех компонентов

---

### UpdateManager

**Ответственность:**
- Вызов `update()` у всех зарегистрированных компонентов
- Правильный порядок обновления

**Порядок:**
1. InputManager (per-frame input logic)
2. SceneSetup (движение камеры)
3. MyObject (анимация)
4. ZoomManager (пересчет invariant point)

---

## 🔗 Граф зависимостей

```
main.py
   ├─→ ColorManager (независимый)
   ├─→ WindowManager (независимый)
   ├─→ InputManager (независимый, потом регистрируются компоненты)
   ├─→ UpdateManager (независимый, потом регистрируются компоненты)
   │
   ├─→ SceneSetup
   │      └─→ ColorManager
   │      └─→ InputManager (опционально)
   │      └─→ UpdateManager (опционально)
   │
   ├─→ Frame
   │      └─→ ColorManager
   │
   ├─→ ZoomManager
   │      └─→ SceneSetup
   │      └─→ ColorManager
   │
   ├─→ ScalableFloor
   └─→ MyObject

Регистрация:
   InputManager.register:
      ├─→ SceneSetup
      ├─→ ZoomManager
      ├─→ Frame
      ├─→ WindowManager
      └─→ MyObject

   UpdateManager.register:
      ├─→ InputManager
      ├─→ SceneSetup
      ├─→ ZoomManager
      └─→ MyObject

   ZoomManager.register:
      ├─→ Floor
      ├─→ Frame elements (4x)
      ├─→ Test objects (3x)
      └─→ MyObject
```

---

## 🎨 Особенности реализации

### TYPE_CHECKING import

Используется **везде** для избежания runtime циркулярных импортов:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .other_module import OtherClass

class MyClass:
    def __init__(self):
        self.other: Optional["OtherClass"] = None
```

`TYPE_CHECKING` = True только во время type checking (mypy, IDE), но False в runtime.

---

### Color Management

`ColorManager` пытается загрузить цвета из `config/json/colors.json`. Если файла нет - использует дефолтные:

```python
DEFAULT_COLORS = {
    'scene': {
        'floor': color.rgb(50, 50, 50),
        'directional_light': color.rgb(255, 244, 214),
        'ambient_light': color.rgb(50, 50, 100),
        'window_background': color.rgb(135, 206, 235)
    },
    'frame': {
        'origin': color.white,
        'x_axis': color.red,
        'y_axis': color.green,
        'z_axis': color.blue
    }
}
```

---

### Watcher (auto-reload)

`run.py` → `watcher.py` → перезапуск `main.py` при изменении `.py` файлов в `src/`.

**Логика:**
- Exit code 0 (нормальный выход) → автоматически перезапустить
- Exit code != 0 (ошибка) → остановиться, дать исправить
- Ctrl+C → остановить watcher

---

## 📐 Математика зума

См. подробное описание в [`../history/decisions.md`](../history/decisions.md) → "Zoom с инвариантной точкой".

**Краткая версия:**

Аффинное преобразование: `x' = a*x + b`

Условие сохранения точки `p`: `p' = p`

```
a*p + b = p
b = p - a*p = (1 - a)*p
```

При изменении зума на коэффициент `k`:
```
a_new = k * a_old
b_new = k * b_old + (1 - k) * p
```

где `p` - инвариантная точка (look point).

---

## 🔮 Будущие расширения

Возможные направления развития (не в приоритете сейчас):

1. **Picking** - выбор объектов мышью (было в v16_picker)
2. **UI система** - информационные панели, HUD
3. **Конфигурация** - загрузка параметров из JSON/YAML
4. **Logging** - замена print на proper logging
5. **Тесты** - юнит-тесты для математики зума

См. [`../state/plan.md`](../state/plan.md) для актуального плана.
