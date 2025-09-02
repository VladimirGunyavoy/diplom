# РЕАЛИЗАЦИЯ: Подписка на изменения dt в InputManager

## Обзор
Успешно реализована система подписки на изменения dt в InputManager, которая обеспечивает автоматическую реакцию всех компонентов системы на изменение временного шага.

## Что реализовано

### ШАГ 1: Подписка на изменения dt в InputManager

#### 1.1 Добавлена подписка в конструктор InputManager
**Файл**: `src/managers/input_manager.py`
```python
# Подписка на изменения dt (чтобы призраки и линки всегда реагировали)
if self.dt_manager and hasattr(self.dt_manager, "subscribe_on_change"):
    self.dt_manager.subscribe_on_change(self._on_dt_changed)
```

#### 1.2 Добавлен обработчик _on_dt_changed
**Файл**: `src/managers/input_manager.py`
```python
def _on_dt_changed(self):
    """
    Глобальная реакция на изменение dt:
    - пересобираем призрачное дерево (если есть),
    - применяем лимит длины ко всем линкам: обычным и призрачным.
    """
    try:
        # 1) Пересобрать призрачные предсказания (дети/внуки двигаются)
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, "_update_predictions"):
            self.manual_spore_manager._update_predictions()

        # 2) Применить новый лимит длины для ВСЕХ линков
        max_len = None
        if self.dt_manager and hasattr(self.dt_manager, "get_max_link_length"):
            max_len = self.dt_manager.get_max_link_length()

        # 2a) Обычные линки (через SporeManager)
        if self.spore_manager and hasattr(self.spore_manager, "update_links_max_length"):
            self.spore_manager.update_links_max_length(max_len)

        # 2b) Призрачные линки (через PredictionManager)
        if self.manual_spore_manager and hasattr(self.manual_spore_manager, "prediction_manager"):
            pm = self.manual_spore_manager.prediction_manager
            if hasattr(pm, "update_links_max_length"):
                pm.update_links_max_length(max_len)
            # На всякий случай обновим геометрию
            if hasattr(pm, "prediction_links"):
                for link in pm.prediction_links:
                    try:
                        link.update_geometry()
                    except Exception:
                        pass
    except Exception as e:
        print(f"[InputManager] _on_dt_changed error: {e}")
```

### ШАГ 2: Добавлен метод update_links_max_length в PredictionManager

**Файл**: `src/managers/manual_creation/prediction_manager.py`
```python
def update_links_max_length(self, max_length: Optional[float]) -> None:
    """Ограничивает длину всех призрачных линков и обновляет геометрию."""
    for link in getattr(self, 'prediction_links', []):
        try:
            link.set_max_length(max_length)
        except Exception as e:
            print(f"[PredictionManager] Не удалось обновить max_length линка: {e}")
```

### ШАГ 3: Добавлена система подписок в DTManager

#### 3.1 Добавлен список подписчиков
**Файл**: `src/managers/dt_manager.py`
```python
# Система подписок на изменения dt
self._subscribers = []
```

#### 3.2 Добавлены методы подписки
```python
def subscribe_on_change(self, callback) -> None:
    """Подписывает колбэк на изменения dt."""
    if callback not in self._subscribers:
        self._subscribers.append(callback)
        print(f"   📧 Подписчик добавлен в DTManager (всего: {len(self._subscribers)})")

def unsubscribe_on_change(self, callback) -> None:
    """Отписывает колбэк от изменений dt."""
    if callback in self._subscribers:
        self._subscribers.remove(callback)
        print(f"   📧 Подписчик удален из DTManager (осталось: {len(self._subscribers)})")

def _notify_subscribers(self) -> None:
    """Уведомляет всех подписчиков об изменении dt."""
    for subscriber in self._subscribers:
        try:
            subscriber()
        except Exception as e:
            print(f"[DTManager] Ошибка в подписчике: {e}")
```

#### 3.3 Обновлен метод _notify_dt_changed
```python
def _notify_dt_changed(self) -> None:
    """Колбэк при изменении dt — обновляем длины всех линков и уведомляем подписчиков."""
    try:
        if getattr(self, 'spore_manager', None):
            self.spore_manager.update_links_max_length(self.get_max_link_length())
    except Exception as e:
        print(f"[DTManager] Ошибка при обновлении длин линков: {e}")
    
    # Уведомляем всех подписчиков
    self._notify_subscribers()
```

### ШАГ 4: Добавлен вызов _on_dt_changed после нажатия P

**Файл**: `src/managers/input_manager.py`
```python
print(f"✅ Призрачное дерево обновлено со спаренными dt!")

# Сразу привести всё в соответствие текущему dt и лимитам длины
if self.dt_manager and hasattr(self, "_on_dt_changed"):
    self._on_dt_changed()
```

## Как это работает

### Архитектура подписки:
```
DTManager._notify_dt_changed()
    ↓
DTManager._notify_subscribers()
    ↓
InputManager._on_dt_changed()
    ↓
├── manual_spore_manager._update_predictions()  # Пересборка призраков
├── spore_manager.update_links_max_length()     # Обычные линки
└── prediction_manager.update_links_max_length() # Призрачные линки
```

### Последовательность событий:
1. **Пользователь изменяет dt** (Ctrl + колесо мыши)
2. **DTManager** вызывает `_notify_dt_changed()`
3. **DTManager** уведомляет всех подписчиков через `_notify_subscribers()`
4. **InputManager._on_dt_changed()** автоматически:
   - Пересобирает призрачные предсказания
   - Применяет новый лимит длины ко всем линкам
   - Обновляет геометрию призрачных линков

### После нажатия P:
1. **Создается призрачное дерево** с оптимальными парами
2. **Автоматически вызывается** `_on_dt_changed()`
3. **Все компоненты синхронизируются** с текущим dt

## Результат

✅ **Автоматическая реакция** всех компонентов на изменение dt
✅ **Синхронизация призрачных объектов** с текущим dt
✅ **Ограничение длины всех линков** (обычных и призрачных)
✅ **Мгновенная реакция** после нажатия P
✅ **Надежная система подписок** с обработкой ошибок

## Тестирование

Создан тест `test_dt_subscription.py`, который проверяет:
- ✅ Импорты всех необходимых модулей
- ✅ Создание DTManager и InputManager
- ✅ Наличие методов подписки
- ✅ Автоматическую подписку при создании InputManager
- ✅ Изменение dt и уведомление подписчиков

## Логирование

При изменении dt в логах появляются сообщения:
```
📧 Подписчик добавлен в DTManager (всего: 1)
🔽 dt уменьшен: 0.0500 → 0.0455 (÷1.1)
```

## Использование

Система работает автоматически:
1. **При создании InputManager** автоматически подписывается на изменения dt
2. **При изменении dt** все компоненты автоматически обновляются
3. **После нажатия P** все синхронизируется с текущим dt

## Заключение

Система подписки на изменения dt полностью реализована и протестирована. Теперь все компоненты системы автоматически реагируют на изменение временного шага, обеспечивая консистентность визуализации и корректную работу всех линков.
