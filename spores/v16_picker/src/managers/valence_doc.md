# Документация: Система Валентности Спор

> **⚠️ ВАЖНО:** Эта документация описывает **исправленную версию v2.0** (2025-10-17).
> См. раздел [История исправлений](#история-исправлений) для деталей.

## Оглавление
1. [Краткая сводка исправлений](#краткая-сводка-исправлений) ⭐ **НАЧНИТЕ ЗДЕСЬ**
2. [Обзор](#обзор)
3. [Концепция валентности](#концепция-валентности)
4. [Структура слотов](#структура-слотов)
5. [Физический смысл](#физический-смысл)
6. [Правила анализа](#правила-анализа)
7. [Примеры](#примеры)
8. [Технические детали](#технические-детали)
9. [История исправлений](#история-исправлений)

---

## Краткая сводка исправлений

### 🎯 В чем была проблема?

#### Главная проблема: Неправильное понимание физики

**❌ Ошибочно думали:** При движении против ребра управление инвертируется (u → -u)

**✅ На самом деле:** Управление **НЕ меняется**, меняется только **знак времени** (dt → -dt)

#### Аналогия с поездом:

```
━━━━━━━━━━━━━━━► Рельсы (управление u=-2, голубые)

Поезд едет вперед:  dt=+0.049  → forward_min
Поезд едет назад:   dt=-0.049  → backward_min

Рельсы (управление) остались теми же!
Изменилось только направление движения (время)!
```

### Все исправленные проблемы:

1. **❌ Проблема:** Управление инвертировалось для входящих связей
   **✅ Исправлено:** Теперь НЕ инвертируется (остается как в ребре)

2. **❌ Проблема:** Только 4 маршрута внуков вместо 8
   **✅ Исправлено:** Теперь 8 маршрутов с обязательной альтернацией управления

3. **❌ Проблема:** Неправильный порядок слотов детей
   **✅ Исправлено:** forward_max, forward_min, backward_max, backward_min

4. **❌ Проблема:** dt_value был None после загрузки из JSON
   **✅ Исправлено:** Читаем данные из EdgeInfo напрямую

### Ключевой принцип (запомнить!)

```
┌─────────────────────────────────────────────────────┐
│  УПРАВЛЕНИЕ ОПРЕДЕЛЯЕТ ТРАЕКТОРИЮ                   │
│  ВРЕМЯ ОПРЕДЕЛЯЕТ НАПРАВЛЕНИЕ ДВИЖЕНИЯ ПО НЕЙ      │
│                                                     │
│  Управление НЕ МЕНЯЕТСЯ при изменении направления! │
└─────────────────────────────────────────────────────┘
```

### Было → Стало

| Аспект | До v2.0 ❌ | После v2.0 ✅ |
|--------|-----------|---------------|
| Управление для IN-связей | Инвертировалось | НЕ инвертируется |
| Маршруты внуков | 4 | 8 (с альтернацией) |
| Порядок детей | Неправильный | Правильный |
| Загрузка из JSON | dt=None | Работает |

---

## Обзор

**Валентность споры** — это система отслеживания всех возможных валентных (связевых) мест споры с её соседями первого и второго порядка.

Каждая спора имеет **12 валентных слотов**:
- **4 слота для детей** (соседи 1-го порядка)
- **8 слотов для внуков** (соседи 2-го порядка, только с альтернацией управления)

### Основные компоненты:

```
spores/v16_picker/src/
├── logic/
│   └── valence.py              # Классы ValenceSlot и SporeValence
└── managers/
    └── valence_manager.py      # ValenceManager для анализа графа
```

---

## Концепция валентности

### Что такое валентный слот?

Валентный слот — это возможное направление роста дерева от споры, определяемое двумя параметрами:
1. **Направление времени** (forward/backward)
2. **Тип управления** (max/min)

### Почему именно 12 слотов?

**4 слота для детей:**
```
forward_max       (dt > 0, control = +u_max)
forward_min       (dt > 0, control = -u_max)
backward_max      (dt < 0, control = +u_max)
backward_min      (dt < 0, control = -u_max)
```

**8 слотов для внуков** (только с чередованием управления):
```
forward_max_forward_min       (max → min, оба forward)
forward_max_backward_min      (max → min, forward → backward)
forward_min_forward_max       (min → max, оба forward)
forward_min_backward_max      (min → max, forward → backward)
backward_max_forward_min      (max → min, оба backward)
backward_max_backward_min     (max → min, backward → forward)
backward_min_forward_max      (min → max, оба backward)
backward_min_backward_max     (min → max, backward → forward)
```

**⚠️ ВАЖНО:** Для внуков учитываются **ТОЛЬКО** пути с **чередованием управления** (max→min или min→max).

Пути без чередования **исключены**:
```
❌ forward_max → forward_max   (max→max)
❌ forward_min → forward_min   (min→min)
❌ backward_max → backward_max (max→max)
❌ backward_min → backward_min (min→min)
```

---

## Структура слотов

### ValenceSlot

```python
@dataclass
class ValenceSlot:
    slot_type: str              # 'child' или 'grandchild'
    time_direction: str         # 'forward' или 'backward'
    control_type: str           # 'max' или 'min'
    second_time_direction: Optional[str] = None  # Только для внуков

    # Данные о занятости
    dt_value: Optional[float] = None       # Суммарное время
    dt_sequence: Optional[List[float]]     # Для внуков: [dt1, dt2]
    occupied: bool = False                 # Занят ли слот
    neighbor_id: Optional[int] = None      # ID соседа
    is_fixed: bool = False                 # Для оптимизатора
```

### SporeValence

```python
@dataclass
class SporeValence:
    spore_id: int
    total_slots: int = 12
    children_slots: List[ValenceSlot]      # 4 слота
    grandchildren_slots: List[ValenceSlot] # 8 слотов
```

---

## Физический смысл

### Ключевой принцип: Управление неизменно вдоль траектории

🔑 **Самое важное для понимания:**

**Управление стрелки НЕ МЕНЯЕТСЯ** при изменении направления движения!

#### Пример:

Есть голубая стрелка (управление u = -2) от споры 6 к споре 2:

```
     6 ←[голубая, u=-2]→ 2
```

**Из споры 6 в спору 2:**
- Управление: u = -2 (min)
- Время: dt = +0.049 (forward, положительное)
- Слот: **forward_min**

**Из споры 2 в спору 6:**
- Управление: u = -2 (min) ← **ТО ЖЕ САМОЕ!**
- Время: dt = -0.049 (backward, отрицательное)
- Слот: **backward_min**

### Почему управление не меняется?

Физически мы движемся **по той же траектории** в фазовом пространстве, только в **противоположном направлении по времени**. Управление определяет **геометрию траектории**, а не направление движения по ней.

### Визуализация

```
Спора 2 (корень дерева):

  ┌─────────────────────────────┐
  │         Спора 2             │
  │    4 детей + 4 внука       │
  └─────────────────────────────┘
        ↓ ↓     ↑ ↑

ДЕТИ (4 слота):
  ↓ forward_max → спора 3  (красная стрелка, dt=+0.050)
  ↓ forward_min → спора 5  (голубая стрелка, dt=+0.049)
  ↑ backward_max ← спора 4 (красная стрелка, dt=-0.050)
  ↑ backward_min ← спора 6 (голубая стрелка, dt=-0.049)

ВНУКИ (8 слотов, только альтернация):
  ✓ forward_max_forward_min → спора 7
  ✗ forward_max_backward_min (свободно)
  ✓ forward_min_forward_max → спора 7
  ✗ forward_min_backward_max (свободно)
  ✗ backward_max_forward_min (свободно)
  ✓ backward_max_backward_min → спора 10
  ✗ backward_min_forward_max (свободно)
  ✓ backward_min_backward_max → спора 10
```

---

## Правила анализа

### 1. Анализ исходящих связей (детей)

**Правило:** Для исходящих связей всегда `forward`

```python
# Из спора_id -> в child_id
time_direction = 'forward'
dt = abs(link.dt_value)      # Всегда положительное
control = link.control_value  # Берем как есть из ребра
```

**Пример:**
```
Ребро: Спора 2 → Спора 3
link.control_value = +2.0 (max)
link.dt_value = 0.05

Результат:
→ forward_max: dt=+0.05, к споре 3
```

### 2. Анализ входящих связей (родителей)

**Правило:** Для входящих связей всегда `backward`

```python
# Из parent_id -> в spore_id (ребро)
# Мы идем: из spore_id -> в parent_id (против ребра)

time_direction = 'backward'
dt = -abs(link.dt_value)     # Всегда отрицательное
control = link.control_value  # ⚠️ НЕ ИНВЕРТИРУЕМ! Берем как есть
```

**Пример:**
```
Ребро: Спора 6 → Спора 2
link.control_value = -2.0 (min)
link.dt_value = 0.049

Из споры 2 в спору 6 (против ребра):
→ backward_min: dt=-0.049, к споре 6
   control = -2.0 (остался min!)
```

### 3. Анализ внуков (2-й порядок)

**Обязательное условие:** Управление ДОЛЖНО чередоваться!

```python
if first_control_type == second_control_type:
    continue  # Пропускаем маршрут без альтернации
```

**Алгоритм:**

1. Получаем всех прямых соседей (детей) текущей споры
2. Для каждого промежуточного соседа получаем его соседей
3. Проверяем чередование управления:
   ```python
   if first_control_type != second_control_type:
       # Это валидный маршрут с альтернацией
   ```
4. Создаем слот внука

**Пример:**
```
Спора 2 → [forward_max] → Спора 3 → [forward_min] → Спора 7
         u=+2, dt=+0.05            u=-2, dt=+0.048

Проверка:
  first_control: max
  second_control: min
  max != min ✓ → Это валидный маршрут!

Слот: forward_max_forward_min
  dt_sequence = [+0.05, +0.048]
  total_dt = +0.098
  neighbor = спора 7
```

---

## Примеры

### Пример 1: Простая спора с 4 детьми

```json
{
  "spore_id": "2",
  "position": [-0.198, 1.682],
  "out_links": [
    {"to": "3", "control": 2.0,  "dt": 0.050},  // forward_max
    {"to": "5", "control": -2.0, "dt": 0.049}   // forward_min
  ],
  "in_links": [
    {"from": "4", "control": 2.0,  "dt": 0.050}, // backward_max
    {"from": "6", "control": -2.0, "dt": 0.049}  // backward_min
  ]
}
```

**Валентность:**
```
📊 ВАЛЕНТНОСТЬ СПОРЫ 2:
   Всего слотов: 12
   Занято: 8
   Свободно: 4

   👶 ДЕТИ (4 слотов, занято 4, свободно 0):
      🔒🔧 forward_max: dt=+0.050000 →3
      🔒🔧 forward_min: dt=+0.049000 →5
      🔒🔧 backward_max: dt=-0.050000 →4
      🔒🔧 backward_min: dt=-0.049000 →6

   👶👶 ВНУКИ (8 слотов, занято 4, свободно 4):
      🔒🔧 forward_max_forward_min: dt=+0.098000 →7
      🔓 forward_max_backward_min: dt=None
      🔒🔧 forward_min_forward_max: dt=+0.099000 →7
      🔓 forward_min_backward_max: dt=None
      🔓 backward_max_forward_min: dt=None
      🔒🔧 backward_max_backward_min: dt=-0.099000 →10
      🔓 backward_min_forward_max: dt=None
      🔒🔧 backward_min_backward_max: dt=-0.100000 →10
```

### Пример 2: Почему управление не инвертируется

**Ситуация:** Красная стрелка от споры 4 к споре 2

```
Ребро: 4 →[red, u=+2]→ 2
       link.control_value = +2.0
       link.dt_value = 0.05
```

**Анализ из споры 2:**

❌ **НЕПРАВИЛЬНО** (старая логика):
```python
# Входящая связь → инвертируем control
control = -link.control_value  # -2.0 (min)
→ backward_min  # НЕПРАВИЛЬНО!
```

✅ **ПРАВИЛЬНО** (текущая логика):
```python
# Входящая связь → НЕ инвертируем control!
control = link.control_value  # +2.0 (max)
dt = -abs(link.dt_value)      # -0.05 (backward)
→ backward_max  # ПРАВИЛЬНО!
```

**Объяснение:**

Красная стрелка означает управление u=+2 (max). Это управление **определяет траекторию** и **не зависит от направления движения** по ней:

- Из 4 в 2: u=+2, dt=+0.05 → движемся вперед по времени с управлением max
- Из 2 в 4: u=+2, dt=-0.05 → движемся назад по времени с **тем же** управлением max

---

## Технические детали

### Класс ValenceManager

**Основные методы:**

```python
class ValenceManager:
    def analyze_spore_valence(spore_id: str) -> SporeValence:
        """Анализирует валентность споры"""

    def _get_direct_neighbors(spore_id: str) -> List[Dict]:
        """Получает соседей 1-го порядка"""

    def _get_neighbors_at_distance_2(spore_id: str) -> List[Dict]:
        """Получает соседей 2-го порядка (с альтернацией)"""

    def print_valence_report(spore_id: str):
        """Выводит подробный отчет"""
```

### Формат данных соседа

```python
neighbor_info = {
    'target_spore': <Spore object>,
    'target_id': '3',
    'path': ['2', '3'],
    'time_direction': 'forward',  # или 'backward'
    'dt': 0.05,                   # Знак соответствует direction
    'dt_sequence': [0.05],        # Для внуков: [dt1, dt2]
    'control': 2.0,               # Положительное для max, отрицательное для min
    'raw_direction': 'outgoing',  # или 'incoming'
}
```

### Интеграция с оптимизатором

```python
valence = manager.analyze_spore_valence(spore_id)

# Получить зафиксированные dt (не трогать при оптимизации)
fixed_dt = valence.get_fixed_dt_values()
# {'forward_max': 0.05, 'backward_min': -0.049, ...}

# Получить свободные слоты (нужно заполнить)
free_slots = valence.get_free_slot_names()
# ['forward_max_backward_min', 'forward_min_backward_max', ...]
```

---

## Диаграмма классов

```
┌─────────────────────────────────────────────┐
│           ValenceManager                    │
├─────────────────────────────────────────────┤
│ + analyze_spore_valence(spore_id)          │
│ + print_valence_report(spore_id)           │
│ - _get_direct_neighbors(spore_id)          │
│ - _get_neighbors_at_distance_2(spore_id)   │
│ - _occupy_slot_from_neighbor(...)          │
└─────────────────────────────────────────────┘
                    │ creates
                    ▼
        ┌───────────────────────────┐
        │     SporeValence          │
        ├───────────────────────────┤
        │ + spore_id                │
        │ + children_slots (4)      │
        │ + grandchildren_slots (8) │
        │ + total_slots = 12        │
        ├───────────────────────────┤
        │ + get_free_slots()        │
        │ + get_occupied_slots()    │
        │ + find_slot_by_name()     │
        └───────────────────────────┘
                    │ contains
                    ▼
            ┌──────────────────────┐
            │   ValenceSlot        │
            ├──────────────────────┤
            │ + slot_type          │
            │ + time_direction     │
            │ + control_type       │
            │ + second_time_dir    │
            │ + dt_value           │
            │ + occupied           │
            │ + neighbor_id        │
            ├──────────────────────┤
            │ + get_slot_name()    │
            └──────────────────────┘
```

---

## Частые ошибки

### ❌ Ошибка 1: Инвертирование управления для входящих связей

```python
# НЕПРАВИЛЬНО
control_value = -raw_control  # Инвертируем управление
```

**Почему неправильно:** Управление определяется **траекторией**, а не направлением движения по ней.

### ❌ Ошибка 2: Забыть про альтернацию для внуков

```python
# НЕПРАВИЛЬНО - включаем все маршруты
for first in children:
    for second in grandchildren:
        add_slot(first, second)  # Будет 16 слотов вместо 8!
```

**Правильно:**

```python
# ПРАВИЛЬНО - только с альтернацией
if first_control_type != second_control_type:
    add_slot(first, second)  # Только 8 слотов
```

### ❌ Ошибка 3: Неправильный знак dt

```python
# НЕПРАВИЛЬНО для входящих
dt_value = abs(raw_dt)  # Положительное для backward!
```

**Правильно:**

```python
# Для outgoing: всегда положительное
dt_value = abs(raw_dt)

# Для incoming: всегда отрицательное
dt_value = -abs(raw_dt)
```

---

## Тестирование

### Простой тест структуры:

```python
from logic.valence import SporeValence

valence = SporeValence(spore_id="test")

# Проверка детей
assert len(valence.children_slots) == 4
assert valence.children_slots[0].get_slot_name() == 'forward_max'
assert valence.children_slots[1].get_slot_name() == 'forward_min'
assert valence.children_slots[2].get_slot_name() == 'backward_max'
assert valence.children_slots[3].get_slot_name() == 'backward_min'

# Проверка внуков
assert len(valence.grandchildren_slots) == 8
slot_names = [s.get_slot_name() for s in valence.grandchildren_slots]
assert 'forward_max_forward_min' in slot_names
assert 'forward_max_forward_max' not in slot_names  # Нет без альтернации!
```

---

## История исправлений

### v2.0 - Текущая версия (2025-10-17) ✅

**Исправленные файлы:**
- `valence.py` - структура слотов
- `valence_manager.py` - логика анализа

**Все исправления:**

1. **Управление для входящих связей**
   - ❌ Было: `control_value = -raw_control` (инвертировалось)
   - ✅ Стало: `control_value = raw_control` (НЕ инвертируется)
   - **Причина:** Управление определяет траекторию, не направление движения

2. **Порядок слотов детей**
   - ❌ Было: `forward_max, backward_min, forward_min, backward_max`
   - ✅ Стало: `forward_max, forward_min, backward_max, backward_min`

3. **Количество слотов внуков**
   - ❌ Было: 4 слота (все маршруты)
   - ✅ Стало: 8 слотов (только с альтернацией управления)
   - **Фильтр:** `if first_control_type == second_control_type: continue`

4. **Загрузка из JSON**
   - ❌ Было: `dt_value = None` (link_object не создавался)
   - ✅ Стало: Чтение из `EdgeInfo` напрямую
   - **Код:**
     ```python
     # Пробуем link_object
     if hasattr(edge_info, 'link_object') and edge_info.link_object:
         return edge_info.link_object.dt_value
     # Fallback: читаем из EdgeInfo напрямую
     if hasattr(edge_info, 'dt_value'):
         return edge_info.dt_value
     ```

5. **Удалены дубликаты**
   - Удалены дублированные функции `_convert_to_float`, `_determine_time_direction`, `_determine_control_type`

**Результаты тестирования:**

До исправления:
```
backward_max → 6  ❌ (должно быть 4)
backward_min → 4  ❌ (должно быть 6)
```

После исправления:
```
forward_max → 3   ✅
forward_min → 5   ✅
backward_max → 4  ✅
backward_min → 6  ✅
Внуков: 8         ✅
```

### v1.0 - Первая версия (ошибочная) ❌

**Проблемы:**
1. Управление инвертировалось для входящих связей
2. Неправильный порядок слотов детей
3. Только 4 слота внуков вместо 8
4. Не работала загрузка из JSON

**Причина ошибки:** Неправильное понимание физического смысла. Думали что управление меняется при изменении направления движения.

---

## Заключение

Система валентности позволяет:

1. **Отслеживать** занятые и свободные валентные места каждой споры
2. **Планировать** рост дерева (какие слоты свободны для новых спор)
3. **Оптимизировать** dt значения, сохраняя существующие связи
4. **Визуализировать** структуру соседства в понятном виде

**Ключевой принцип:** Управление определяет **траекторию**, время определяет **направление движения** по ней.

### Быстрая справка

```python
# Анализ валентности
valence = manager.analyze_spore_valence(spore_id)
valence.print_summary()

# Получить занятые/свободные слоты
occupied = valence.get_occupied_slots()  # Существующие связи
free = valence.get_free_slots()          # Куда можно расти

# Для оптимизатора
fixed_dt = valence.get_fixed_dt_values()      # Не трогать
free_slots = valence.get_free_slot_names()    # Нужно заполнить
```

---

*Документация создана: 2025-10-17*
*Версия: 2.0*
*Автор: ValenceManager Team*
*Статус: ✅ ВСЕ ПРОБЛЕМЫ ИСПРАВЛЕНЫ*
