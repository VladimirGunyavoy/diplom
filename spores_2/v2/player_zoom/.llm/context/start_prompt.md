# Player Zoom - Quick Start

**Last updated:** 2026-04-19

---

## 🎯 TL;DR

**Player Zoom** - исследовательская песочница на базе **Ursina** (Python 3D движок) для экспериментов с камерой и зумом. Экстрагирован из большого проекта v16_picker для независимой разработки.

**Ключевая фича:** математически корректный зум с сохранением **инвариантной точки** (точка пересечения луча взгляда камеры с плоскостью y=0).

**Архитектура:** 5 менеджеров (Color, Window, Input, Update, Zoom) + иерархия Scalable объектов. Dependency Injection для избежания циркулярных импортов.

**Стек:** Python 3.12, Ursina, NumPy, Watchdog (для auto-reload).

---

## 📐 Ключевые концепции

### 1. Manager Pattern

Все менеджеры независимы и создаются первыми:

```python
# main.py
color_manager = ColorManager()
window_manager = WindowManager()
input_manager = InputManager()
update_manager = UpdateManager()
zoom_manager = ZoomManager(scene_setup, color_manager)
```

Затем компоненты **регистрируются** в менеджерах (DI):

```python
input_manager.register_scene_setup(scene_setup)
input_manager.register_zoom_manager(zoom_manager)
update_manager.register_input_manager(input_manager)
```

**Зачем?** Избежание циркулярных импортов при сложных зависимостях.

---

### 2. Scalable Hierarchy

Базовый класс для всех объектов, которые должны масштабироваться при зуме:

```python
class Scalable(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_position = np.array(self.position)  # Оригинальная позиция
        self.real_scale = np.array(self.scale)        # Оригинальный масштаб

    def apply_transform(self, a: float, b: np.ndarray):
        """Применить аффинное преобразование: x' = a*x + b"""
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a
```

**Наследники:**
- `ScalableFloor` - пол с текстурой
- `ScalableFrame` - элементы координатной системы (в Frame)
- `MyObject` - демонстрационный объект (движется по кругу)

---

### 3. Zoom Manager - математика

**Задача:** масштабировать сцену так, чтобы точка, на которую смотрит камера, оставалась на месте.

**Решение:** аффинное преобразование `x' = a*x + b`, где:
- `a` - коэффициент масштабирования
- `b` - вектор сдвига

**Инвариантная точка** (look point):
```python
def identify_invariant_point(self):
    """Точка пересечения луча взгляда с плоскостью y=0"""
    h = camera.world_position.y
    psi = camera.rotation_y  # азимут
    phi = camera.rotation_x  # наклон

    d = h / tan(phi)  # расстояние до точки
    x_0 = camera.x + d * sin(psi)
    z_0 = camera.z + d * cos(psi)

    return (x_0, z_0)
```

При зуме:
```python
zoom_multiplier = zoom_fact ** sign  # zoom_fact = 1.125 (=1+1/8)
a *= zoom_multiplier
b = zoom_multiplier * b + (1 - zoom_multiplier) * invariant_point
```

Все объекты в `zoom_manager.objects` получают `apply_transform(a, b)`.

---

### 4. Input Flow

```
User input (key press)
    ↓
main.py: input(key)  ← глобальный хендлер Ursina
    ↓
InputManager.handle_input(key)
    ↓
    ├─→ zoom_manager.zoom_in/out()
    ├─→ scene_setup.toggle_freeze()
    ├─→ window_manager.toggle_fullscreen()
    └─→ my_object.increase_speed()
```

---

### 5. Update Flow

```
main.py: update()  ← вызывается каждый кадр Ursina
    ↓
UpdateManager.update_all(dt)
    ↓
    ├─→ input_manager.update()
    ├─→ scene_setup.update(dt)  ← движение камеры (Space/Shift)
    ├─→ my_object.update_position(dt)  ← движение по кругу
    └─→ zoom_manager.identify_invariant_point()  ← пересчет точки взгляда
```

---

## 🗂️ Структура проекта

```
player_zoom/
├── main.py                 # Точка входа, демонстрация
├── run.py                  # Запуск с watcher (auto-reload)
├── README.md               # Документация
├── assets/
│   └── arrow.obj           # 3D модель для осей координат
├── src/
│   ├── scalable.py         # Базовые классы: Scalable, ScalableFloor
│   ├── frame.py            # Координатная система (X,Y,Z оси)
│   ├── scene_setup.py      # Камера + освещение + управление курсором
│   ├── zoom_manager.py     # Управление зумом
│   ├── color_manager.py    # Управление цветами
│   ├── window_manager.py   # Управление окном/мониторами
│   ├── input_manager.py    # Обработка ввода
│   ├── update_manager.py   # Координация update() всех компонентов
│   ├── my_object.py        # Демонстрационный объект
│   ├── watcher.py          # Автоперезапуск при изменении файлов
│   └── math/
│       └── double_integrator.py  # 2D double integrator (x_ddot=u)
└── llm/                    # Контекст для AI агентов
```

---

## 🎮 Управление

| Действие | Клавиши |
|----------|---------|
| Движение | WASD |
| Взгляд | Mouse |
| Вверх/вниз | Space / Shift |
| Zoom in | E |
| Zoom out | Q |
| Reset zoom | R |
| Захват/освобождение курсора | Alt |
| Полноэкранный режим | F11 |
| Debug info | H |
| Скорость MyObject | 1 (медленнее), 2 (быстрее) |
| Выход | Escape |

---

## 🚀 Запуск

```bash
# Обычный запуск
python main.py

# С автоперезапуском (для разработки)
python run.py
```

**Зависимости:**
```bash
pip install ursina numpy watchdog
```

---

## 📍 Текущая работа

- Актуальные задачи → [`state/plan.md`](../state/plan.md)
- Текущее состояние → [`state/current.md`](../state/current.md)
- Известные проблемы → [`state/issues.md`](../state/issues.md)

---

## 🔗 Дальнейшее чтение

- **Детали архитектуры** → [`architecture.md`](architecture.md)
- **Граф зависимостей** → [`dependencies.md`](dependencies.md)
- **Важные куски кода** → [`code_snippets.md`](code_snippets.md)
- **История решений** → [`../history/decisions.md`](../history/decisions.md)
