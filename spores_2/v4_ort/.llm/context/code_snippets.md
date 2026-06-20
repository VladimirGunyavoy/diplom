# Code Snippets - Важные куски кода

**Last updated:** 2026-03-19

> Здесь собраны ключевые куски кода с объяснениями. Не нужно читать весь проект - достаточно понять эти паттерны.

---

## 🎯 Scalable.apply_transform()

**Файл:** `src/scalable.py`

```python
class Scalable(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем ОРИГИНАЛЬНЫЕ значения
        self.real_position = np.array(self.position)
        self.real_scale = np.array(self.scale)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs):
        """
        Аффинное преобразование: x' = a*x + b

        ВАЖНО: применяется к real_*, а не к текущим значениям!
        Иначе будет накопление ошибок.
        """
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a
```

**Почему важно:**
- Трансформация всегда применяется к **оригинальным** значениям
- Если применять к текущим → накопление ошибок при повторном зуме
- Это ключевая идея всей системы зума

---

## 🔍 ZoomManager.identify_invariant_point()

**Файл:** `src/zoom_manager.py`

```python
def identify_invariant_point(self) -> Tuple[float, float]:
    """
    Вычислить точку взгляда (где луч камеры пересекает плоскость y=0).

    Геометрия:
    - Камера на высоте h, смотрит под углом phi вниз
    - Азимут (направление) = psi
    - Расстояние до точки на плоскости: d = h / tan(phi)
    """
    player = self.scene_setup.player
    psi = np.radians(player.rotation_y)           # азимут (yaw)
    phi = np.radians(player.camera_pivot.rotation_x)  # наклон (pitch)

    h = player.camera_pivot.world_position.y      # высота камеры

    # Edge case: камера смотрит прямо вперед (горизонт)
    if abs(np.tan(phi)) < 0.001:
        return 0, 0

    d = h / np.tan(phi)  # расстояние до точки

    # Проекция на плоскость XZ
    dx = d * np.sin(psi)
    dz = d * np.cos(psi)

    x_0 = player.camera_pivot.world_position.x + dx
    z_0 = player.camera_pivot.world_position.z + dz

    return x_0, z_0
```

**Почему важно:**
- Это математическое ядро системы зума
- Определяет точку, которая должна остаться неподвижной при зуме
- Edge case для горизонтального взгляда (tan(phi) → 0)

---

## 🔄 ZoomManager.change_zoom()

**Файл:** `src/zoom_manager.py`

```python
def change_zoom(self, sign: int) -> None:
    """
    Изменить зум с сохранением инвариантной точки.

    sign: +1 (zoom in), -1 (zoom out)
    """
    # 1. Получить инвариантную точку (2D на плоскости y=0)
    inv = np.array(self.identify_invariant_point())
    inv_3d = np.array([inv[0], 0, inv[1]])  # (x, y=0, z)

    # 2. Вычислить новые параметры трансформации
    zoom_multiplier = self.zoom_fact ** sign  # 1.125^(±1)

    self.a_transformation *= zoom_multiplier
    self.b_translation = zoom_multiplier * self.b_translation + \
                         (1 - zoom_multiplier) * inv_3d

    # 3. Применить ко всем объектам
    self.update_transform()
```

**Почему важно:**
- Формула `b_new = k*b_old + (1-k)*p` - ключевая для сохранения точки
- `update_transform()` вызывает `apply_transform()` у всех объектов
- `zoom_multiplier` в степени sign - элегантное решение для ±1

---

## 🎮 InputManager.handle_input()

**Файл:** `src/input_manager.py`

```python
def handle_input(self, key: str) -> None:
    """Централизованная обработка ввода."""

    # Escape - всегда работает
    if key == 'escape':
        application.quit()
        return

    # Fullscreen - всегда работает
    if key == 'f11' and self.window_manager:
        self.window_manager.toggle_fullscreen()
        return

    # Toggle cursor - всегда работает
    if key == 'alt' and self.scene_setup:
        self.scene_setup.toggle_freeze()
        return

    # Если курсор освобожден - остальные команды не работают
    if self.scene_setup and self.scene_setup.input_frozen:
        return

    # === ZOOM ===
    if self.zoom_manager:
        if key == 'e':
            self.zoom_manager.zoom_in()
        elif key == 'q':
            self.zoom_manager.zoom_out()
        elif key == 'r':
            self.zoom_manager.reset_zoom()

    # === MY OBJECT ===
    if self.my_object:
        if key == '1':
            self.my_object.decrease_speed()
        elif key == '2':
            self.my_object.increase_speed()

    # === DEBUG ===
    if key == 'h':
        self._print_debug_info()
```

**Почему важно:**
- Порядок проверок: escape/f11/alt → проверка frozen → остальные
- Если `input_frozen` → игнорируем все кроме системных клавиш
- Паттерн делегирования: InputManager не знает КАК работает зум, только КОГДА вызвать

---

## 🔗 Dependency Injection паттерн

**Файл:** `main.py`

```python
# 1. Создаем менеджеры (независимо)
color_manager = ColorManager()
input_manager = InputManager()
update_manager = UpdateManager()

# 2. Создаем компоненты (с зависимостями через конструктор)
scene_setup = SceneSetup(
    color_manager=color_manager,
    input_manager=input_manager,
    update_manager=update_manager
)

zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)

# 3. Регистрируем компоненты в менеджерах (DI!)
input_manager.register_scene_setup(scene_setup)
input_manager.register_zoom_manager(zoom_manager)

update_manager.register_input_manager(input_manager)
update_manager.register_scene_setup(scene_setup)
update_manager.register_zoom_manager(zoom_manager)
```

**Почему важно:**
- Разрывает циркулярные зависимости
- Менеджеры не импортируют компоненты → только TYPE_CHECKING
- Компоненты регистрируются после создания всех объектов

---

## 🎨 TYPE_CHECKING паттерн

**Файл:** `src/input_manager.py`

```python
from typing import Optional, TYPE_CHECKING

# Импорты только для type hints (не в runtime!)
if TYPE_CHECKING:
    from .scene_setup import SceneSetup
    from .zoom_manager import ZoomManager

class InputManager:
    def __init__(self):
        # Type hints используют строки (forward references)
        self.scene_setup: Optional["SceneSetup"] = None
        self.zoom_manager: Optional["ZoomManager"] = None

    def register_scene_setup(self, scene_setup: "SceneSetup") -> None:
        self.scene_setup = scene_setup
```

**Почему важно:**
- `TYPE_CHECKING` = True только при type checking (mypy, IDE)
- В runtime эти импорты не выполняются → нет циркулярных зависимостей
- Forward references через строки: `"SceneSetup"` вместо `SceneSetup`

---

## 🔄 UpdateManager.update_all()

**Файл:** `src/update_manager.py`

```python
def update_all(self, dt: float) -> None:
    """
    Вызывается каждый кадр из main.py: update()

    Порядок важен!
    """
    # 1. Per-frame input logic (пока пустой)
    if self.input_manager:
        self.input_manager.update()

    # 2. Движение камеры (Space/Shift для вертикали)
    if self.scene_setup:
        self.scene_setup.update(dt)

    # 3. Анимация объектов
    if self.my_object:
        self.my_object.update_position(dt)

    # 4. Пересчет invariant point (для debug info)
    if self.zoom_manager:
        self.zoom_manager.identify_invariant_point()
```

**Почему важно:**
- Единая точка входа для всех update()
- Гарантирует правильный порядок обновлений
- Легко добавить новые компоненты

---

## 🎭 SceneSetup.toggle_freeze()

**Файл:** `src/scene_setup.py`

```python
def toggle_freeze(self) -> None:
    """Переключить режим захвата курсора."""
    self.input_frozen = not self.input_frozen

    # Обновить состояние курсора
    mouse.locked = not self.input_frozen
    mouse.visible = self.input_frozen

    # Заблокировать/разблокировать плеера
    self.player.enabled = not self.input_frozen

    status = "unlocked" if self.input_frozen else "locked"
    print(f"[SceneSetup] Cursor {status}")
```

**Почему важно:**
- `input_frozen` управляет и курсором и player.enabled
- `mouse.locked` и `mouse.visible` - зеркально противоположны
- `player.enabled = False` → WASD не работает (встроенный механизм Ursina)

---

## 🌍 Frame регистрация

**Файл:** `main.py`

```python
# Создаем Frame (содержит 4 Scalable объекта)
frame = Frame(color_manager=color_manager, origin_scale=0.05)

# Регистрируем каждый элемент Frame отдельно!
for i, entity in enumerate(frame.entities):
    zoom_manager.register_object(entity, name=f'frame_child_{i}')
```

**Почему важно:**
- Frame сам по себе - не Scalable (обычный Entity)
- Но его дети (origin_cube, x_axis, y_axis, z_axis) - Scalable
- Регистрируем детей, а не родителя
- Это позволяет масштабировать оси координат при зуме

---

## 🎯 MyObject движение по кругу

**Файл:** `src/my_object.py`

```python
def update_position(self, dt: float) -> None:
    """Обновить позицию (движение по кругу)."""
    # Обновить угол
    self.angle += self.speed * dt

    # Нормализовать угол в [0, 2π]
    if self.angle >= 2 * math.pi:
        self.angle -= 2 * math.pi

    # Параметрическое уравнение окружности
    x = self.radius * math.cos(self.angle)
    z = self.radius * math.sin(self.angle)
    self.position = (x, 0, z)  # y=0 (на плоскости)
```

**Почему важно:**
- Демонстрирует анимацию в координатах до трансформации
- После этого `apply_transform()` применит зум к позиции
- `self.position` устанавливается напрямую → `real_position` не меняется автоматически!

**⚠️ Потенциальная проблема:**
`real_position` не обновляется при анимации → может быть баг. См. `state/issues.md`.

---

## 📊 Полный цикл зума

```python
# 1. User presses 'e'
main.py: input('e')

# 2. Делегирование в InputManager
InputManager.handle_input('e')
    → zoom_manager.zoom_in()

# 3. ZoomManager вычисляет трансформацию
ZoomManager.zoom_in()
    → ZoomManager.change_zoom(+1)
        → inv_point = identify_invariant_point()
        → a *= 1.125
        → b = 1.125*b + (1-1.125)*inv_point
        → update_transform()

# 4. Применение ко всем объектам
ZoomManager.update_transform()
    for obj in objects.values():
        → obj.apply_transform(a, b)

# 5. Каждый Scalable пересчитывает позицию
Scalable.apply_transform(a, b)
    → position = real_position * a + b
    → scale = real_scale * a

# Результат: сцена масштабировалась, точка взгляда на месте
```

---

## 🔧 Полезные паттерны

### Optional с проверкой

```python
if self.zoom_manager:  # проверка что зарегистрирован
    self.zoom_manager.zoom_in()
```

### Try-except для invalid entities

```python
# В ZoomManager.update_transform()
for obj in self.objects.values():
    try:
        if hasattr(obj, 'enabled') and obj.enabled and hasattr(obj, 'position'):
            obj.apply_transform(self.a_transformation, self.b_translation)
    except (AssertionError, AttributeError, RuntimeError):
        continue  # Объект удален/невалиден - пропускаем
```

**Почему:** Ursina может удалить Entity в любой момент → graceful degradation.

---

Эти снипеты покрывают 90% логики проекта. Остальное - boilerplate.
