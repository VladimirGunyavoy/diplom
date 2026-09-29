# Dependencies - Граф зависимостей

**Last updated:** 2026-03-19

---

## 🌳 Дерево зависимостей

### Легенда:
- `A → B` - A зависит от B (требует B для работы)
- `A ⇢ B` - A опционально использует B (может быть None)
- `[DI]` - зависимость через Dependency Injection (регистрация)

---

## 📦 Main.py → Компоненты

```
main.py
├── ColorManager (независимый, создается первым)
├── WindowManager (независимый)
├── InputManager (независимый, компоненты регистрируются позже)
├── UpdateManager (независимый, компоненты регистрируются позже)
│
├── SceneSetup
│   ├── → ColorManager
│   ├── ⇢ InputManager (optional, для делегирования)
│   └── ⇢ UpdateManager (optional, для делегирования)
│
├── Frame
│   └── → ColorManager
│
├── ZoomManager
│   ├── → SceneSetup
│   └── → ColorManager
│
├── ScalableFloor
│   └── (extends Scalable)
│
└── MyObject
    └── (extends Scalable)
```

---

## 🔄 Регистрация компонентов (DI)

### InputManager регистрирует:

```
InputManager [DI]
├── SceneSetup           (для toggle_freeze, состояния input_frozen)
├── ZoomManager          (для zoom_in/out/reset)
├── Frame                (потенциально для toggle_visibility)
├── WindowManager        (для toggle_fullscreen)
└── MyObject             (для increase/decrease_speed)
```

### UpdateManager регистрирует:

```
UpdateManager [DI]
├── InputManager         (вызов update())
├── SceneSetup           (вызов update(dt) для движения камеры)
├── ZoomManager          (вызов identify_invariant_point())
└── MyObject             (вызов update_position(dt))
```

### ZoomManager регистрирует:

```
ZoomManager [DI]
├── ScalableFloor        (для масштабирования)
├── Frame.entities[0..3] (origin_cube, x_axis, y_axis, z_axis)
├── test_object_1        (demo sphere)
├── test_object_2        (demo sphere)
├── test_object_3        (demo sphere)
└── MyObject             (движущаяся сфера)
```

---

## 🔗 Детальные зависимости по файлам

### src/scalable.py
```
Scalable (Entity)
└── numpy (для np.array)
```

**Наследники:**
- ScalableFloor
- ScalableFrame (внутри Frame)
- MyObject

---

### src/color_manager.py
```
ColorManager
└── ursina.color (для дефолтных цветов)
```

**Кто использует:** все компоненты с цветами

---

### src/window_manager.py
```
WindowManager
└── ursina.window (для управления окном/мониторами)
```

**Кто использует:** InputManager (toggle fullscreen)

---

### src/scene_setup.py
```
SceneSetup
├── ursina (Entity, Light, FirstPersonController, mouse, window)
├── ColorManager (обязательный)
├── ⇢ InputManager (опциональный)
└── ⇢ UpdateManager (опциональный)
```

**Кто использует:**
- ZoomManager (для доступа к player.position/rotation)
- InputManager (для toggle_freeze)
- UpdateManager (для вызова update(dt))

---

### src/zoom_manager.py
```
ZoomManager
├── ursina (для базовых типов)
├── numpy (для математики)
├── → Scalable (TYPE_CHECKING import)
├── → ColorManager (обязательный)
└── → SceneSetup (обязательный, для доступа к камере)
```

**Кто использует:**
- InputManager (для zoom команд)
- UpdateManager (для identify_invariant_point)
- main.py (для регистрации объектов)

---

### src/frame.py
```
Frame (Entity)
├── ursina (Entity, scene)
├── → ColorManager (обязательный)
└── → Scalable (для дочерних элементов)
```

**Кто использует:**
- main.py (создание)
- InputManager (потенциально для toggle visibility)

---

### src/input_manager.py
```
InputManager
├── ursina (held_keys, mouse, application)
├── [DI] ⇢ SceneSetup
├── [DI] ⇢ ZoomManager
├── [DI] ⇢ Frame
├── [DI] ⇢ WindowManager
└── [DI] ⇢ MyObject
```

Все зависимости через TYPE_CHECKING + DI!

**Кто использует:**
- main.py (глобальный input handler)
- UpdateManager (вызов update())

---

### src/update_manager.py
```
UpdateManager
├── [DI] ⇢ SceneSetup
├── [DI] ⇢ ZoomManager
├── [DI] ⇢ InputManager
└── [DI] ⇢ MyObject
```

Все зависимости через TYPE_CHECKING + DI!

**Кто использует:**
- main.py (глобальный update handler)

---

### src/my_object.py
```
MyObject (Scalable)
├── math (для cos/sin)
├── ursina.color
└── → Scalable (наследование)
```

**Кто использует:**
- main.py (создание)
- UpdateManager (вызов update_position)
- InputManager (управление скоростью)

---

## 🎯 Порядок инициализации в main.py

**Критически важен!** Нарушение порядка → ошибки.

```
1. ColorManager()           # Независимый
2. WindowManager()          # Независимый
3. InputManager()           # Независимый (пустой)
4. UpdateManager()          # Независимый (пустой)

5. SceneSetup(              # Требует ColorManager
     color_manager,
     input_manager,
     update_manager
   )

6. Frame(                   # Требует ColorManager
     color_manager
   )

7. ZoomManager(             # Требует SceneSetup, ColorManager
     scene_setup,
     color_manager
   )

8. ScalableFloor()          # Независимый
9. test_objects (x3)        # Независимые
10. MyObject()              # Независимый

11. Регистрация в InputManager:
    input_manager.register_scene_setup(scene_setup)
    input_manager.register_zoom_manager(zoom_manager)
    input_manager.register_frame(frame)
    input_manager.register_window_manager(window_manager)
    input_manager.register_my_object(my_object)

12. Регистрация в UpdateManager:
    update_manager.register_input_manager(input_manager)
    update_manager.register_scene_setup(scene_setup)
    update_manager.register_zoom_manager(zoom_manager)
    update_manager.register_my_object(my_object)

13. Регистрация в ZoomManager:
    zoom_manager.register_object(floor, 'floor')
    zoom_manager.register_object(frame.entities[i], f'frame_child_{i}')
    zoom_manager.register_object(test_object_1, 'test_object_1')
    zoom_manager.register_object(test_object_2, 'test_object_2')
    zoom_manager.register_object(test_object_3, 'test_object_3')
    zoom_manager.register_object(my_object, 'my_object')
```

---

## 🔄 Runtime зависимости (потоки данных)

### Input Flow:
```
User → Ursina → main.input() → InputManager.handle_input()
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
             ZoomManager     SceneSetup      WindowManager
```

### Update Flow:
```
Ursina → main.update() → UpdateManager.update_all()
                              ↓
              ┌───────────────┼──────────────┐
              ↓               ↓              ↓
         InputManager   SceneSetup      MyObject
                              ↓
                         ZoomManager
```

### Zoom Flow:
```
ZoomManager.change_zoom()
    ↓
    ├─→ SceneSetup (read player.position/rotation)
    │       └─→ для identify_invariant_point()
    │
    └─→ registered Scalable objects
            └─→ apply_transform(a, b)
```

---

## 🚫 Предотвращенные циркулярные зависимости

### ❌ Потенциальные циклы (если бы не DI):

```
InputManager ←→ ZoomManager
    ↓                ↑
SceneSetup ←────────┘
```

- InputManager нужен ZoomManager для команд зума
- ZoomManager нужен SceneSetup для позиции камеры
- SceneSetup нужен InputManager для toggle_freeze
- → Цикл!

### ✅ Решение через DI:

```
main.py
  ├── создает InputManager (пустой)
  ├── создает SceneSetup
  ├── создает ZoomManager(scene_setup)
  └── регистрирует ZoomManager в InputManager
```

Зависимости разрешаются **во время исполнения**, а не **во время импорта**.

---

## 📚 TYPE_CHECKING imports

Используется **везде** для type hints без runtime импортов:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .other_module import OtherClass  # импорт только для mypy/IDE

class MyClass:
    def __init__(self):
        self.other: Optional["OtherClass"] = None  # строка!
```

**Почему строка?** Forward reference - класс еще не импортирован в runtime.

---

## 🔮 Будущие зависимости

Если будет добавлен:

- **UIManager** → зависит от ColorManager, InputManager
- **ConfigManager** → от него зависят все (для загрузки параметров)
- **LogManager** → заменит print statements
- **EventBus** → для decoupling компонентов (pub/sub pattern)

См. [`../state/plan.md`](../state/plan.md) для актуального плана.
