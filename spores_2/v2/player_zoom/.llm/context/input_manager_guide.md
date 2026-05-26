# InputManager - Руководство по работе с вводом

**Last updated:** 2026-03-19

---

## 📋 Назначение

`InputManager` - централизованный обработчик всего пользовательского ввода в player_zoom.

**Файл:** [`src/input_manager.py`](../../src/input_manager.py)

---

## 🎯 Принципы работы

### 1. Dependency Injection Pattern

InputManager **не импортирует** компоненты напрямую. Вместо этого:
1. Создаётся **независимо** в `main.py`
2. Компоненты **регистрируются** через `register_*()` методы
3. Использует `TYPE_CHECKING` для type hints

**Почему?** Избежание циркулярных импортов (см. [`architecture.md`](architecture.md))

```python
# main.py
input_manager = InputManager()
zoom_manager = ZoomManager(...)
input_manager.register_zoom_manager(zoom_manager)  # ← DI
```

---

### 2. Централизованная обработка

Все клавиши обрабатываются в **одном месте** - метод `handle_input(key)`:

```
User presses 'e'
    ↓
main.py: input(key)  ← Ursina global handler
    ↓
InputManager.handle_input('e')
    ↓
Делегирует → zoom_manager.zoom_in()
```

---

### 3. Input Freeze механизм

InputManager проверяет флаг `scene_setup.input_frozen`:
- **True** → большинство команд **игнорируются** (курсор свободен)
- **False** → нормальная обработка

**Исключения** (всегда работают):
- `Escape` - выход
- `F11` - fullscreen
- `Alt` - toggle freeze

```python
if self.scene_setup and self.scene_setup.input_frozen:
    return  # ← блокируем остальные команды
```

---

## ⌨️ Текущие привязки клавиш

### Системные команды (всегда активны)

| Клавиша | Действие | Код |
|---------|----------|-----|
| `Escape` | Выход из приложения | `application.quit()` |
| `F11` | Toggle fullscreen | `window_manager.toggle_fullscreen()` |
| `Alt` | Toggle cursor lock/freeze | `scene_setup.toggle_freeze()` |

### Zoom команды

| Клавиша | Действие | Компонент |
|---------|----------|-----------|
| `E` | Zoom in | `zoom_manager.zoom_in()` |
| `Q` | Zoom out | `zoom_manager.zoom_out()` |
| `R` | Reset zoom | `zoom_manager.reset_zoom()` |

### Frame команды

| Клавиша | Действие | Компонент |
|---------|----------|-----------|
| `U` | Toggle frame visibility | `frame.toggle_visibility()` |

### Object control

| Клавиша | Действие | Компонент |
|---------|----------|-----------|
| `1` | Уменьшить скорость MyObject | `my_object.decrease_speed()` |
| `2` | Увеличить скорость MyObject | `my_object.increase_speed()` |

### Debug

| Клавиша | Действие | Описание |
|---------|----------|----------|
| `H` | Показать debug info | Выводит состояние камеры, зума, объектов |

---

## 🔧 Добавление новой клавиши

### Шаг 1: Зарегистрировать компонент (если новый)

```python
# В main.py
input_manager.register_frame(frame)
```

```python
# В input_manager.py
def register_frame(self, frame: "Frame") -> None:
    self.frame = frame
    print(f"   frame: registered")
```

### Шаг 2: Добавить обработку в handle_input()

```python
def handle_input(self, key: str) -> None:
    # ... существующие проверки ...

    # Проверка input_frozen (если команда должна блокироваться)
    if self.scene_setup and self.scene_setup.input_frozen:
        return

    # === НОВАЯ КОМАНДА ===
    if key == 'u' and self.frame:
        self.frame.toggle_visibility()
        print("   [Frame] Visibility toggled")
        return
```

### Шаг 3: Обновить документацию

Добавить клавишу в таблицу выше ☝️

---

## 📐 Структура метода handle_input()

```python
def handle_input(self, key: str) -> None:
    # 1. Системные команды (ВСЕГДА обрабатываются)
    if key == 'escape': ...
    if key == 'f11': ...
    if key == 'alt': ...

    # 2. Проверка input_frozen
    if self.scene_setup and self.scene_setup.input_frozen:
        return

    # 3. Основные команды (только если input НЕ frozen)
    if key == 'e': ...  # zoom
    if key == '1': ...  # object control
    if key == 'h': ...  # debug
```

**Порядок важен!** Системные команды должны быть **до** проверки `input_frozen`.

---

## 🐛 Debug режим (H)

При нажатии `H` выводится:
- Позиция и rotation камеры
- Состояние cursor_locked / input_frozen
- Параметры зума (a, b)
- Количество зарегистрированных объектов
- Look point (invariant point)
- Frame visibility

**Использование:** Быстрая проверка состояния при разработке.

---

## ⚠️ Что НЕ делать

### ❌ Плохо: Обработка ввода напрямую в компонентах

```python
# В zoom_manager.py - ПЛОХО!
def input(self, key):
    if key == 'e':
        self.zoom_in()
```

**Почему плохо:**
- Распределённая логика (сложно найти все привязки)
- Конфликты клавиш
- Сложнее контролировать input_frozen

### ✅ Хорошо: Всё через InputManager

```python
# В input_manager.py - ХОРОШО!
def handle_input(self, key):
    if key == 'e' and self.zoom_manager:
        self.zoom_manager.zoom_in()
```

---

## 🔮 Будущие улучшения (в backlog)

См. [`plan.md`](../state/plan.md) → Backlog → Input & Controls

Возможные направления:
- **Декларативная структура** - словарь с описанием привязок
- **Автогенерация help** - список всех клавиш
- **Комбинации клавиш** - Ctrl+E, Shift+Q
- **Конфликты** - автопроверка занятых клавиш
- **Rebinding** - пользовательская настройка клавиш

**Приоритет:** Низкий (текущий подход достаточен для ~10-20 команд)

---

## 📝 Примеры использования

### Пример 1: Добавить toggle frame на клавишу U

```python
# В handle_input(), после проверки input_frozen:
if key == 'u' and self.frame:
    self.frame.toggle_visibility()
    print(f"   [Frame] {'visible' if self.frame.is_visible() else 'hidden'}")
    return
```

### Пример 2: Добавить reset camera на клавишу C

```python
# 1. Добавить метод в SceneSetup
def reset_camera_position(self):
    self.player.position = (0, 5, -10)
    self.player.rotation_y = 0
    self.player.camera_pivot.rotation_x = 0

# 2. В InputManager.handle_input():
if key == 'c' and self.scene_setup:
    self.scene_setup.reset_camera_position()
    print("   [Camera] Reset to default position")
    return
```

### Пример 3: Условное действие (только при определённом состоянии)

```python
# Zoom только если Frame видим
if key == 'e' and self.zoom_manager and self.frame and self.frame.is_visible():
    self.zoom_manager.zoom_in()
    print("   [Zoom] Zoom in")
    return
```

---

## 🔗 Связанные файлы

- [`src/input_manager.py`](../../src/input_manager.py) - код
- [`context/architecture.md`](architecture.md) - Input Flow диаграмма
- [`state/plan.md`](../state/plan.md) - будущие улучшения в backlog
- [`main.py`](../../main.py) - как InputManager создаётся и регистрируется

---

**Принцип:** Весь ввод через InputManager. Один файл - одна ответственность. Просто и понятно.
