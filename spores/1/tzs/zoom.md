# Техническое задание: Система независимого 3D зума

## 🎯 Цель
Реализовать систему независимого масштабирования по осям X, Y, Z для 3D сцены с маятником в Ursina, где размеры объектов и дистанции масштабируются, но размеры осей координат остаются фиксированными.

## 📋 Основные требования

### **Функциональные требования:**
1. **Независимое масштабирование** по трем осям (X, Y, Z)
2. **Линейное изменение зума** (не экспоненциальное)
3. **Плавная анимация** изменения зума
4. **Адаптивная скорость камеры** (обратно пропорциональна зуму)
5. **Отображение логических координат** камеры (θ, ω)
6. **Сохранение позиции камеры** при зуме

### **Технические требования:**
1. **Диапазон зума:** 0.001x - 1000x (теоретически безграничный)
2. **Скорость зума:** Настраиваемая (по умолчанию 10% в секунду)
3. **Управление:** Стрелки клавиатуры
4. **Координатная система:** θ (X), ω (Z), высота (Y)

### **Объекты для масштабирования:**
- ✅ Точки состояния (StatePoint)
- ✅ Связи между точками
- ✅ Границы области (theta_lims, omega_lims)
- ✅ Все игровые объекты
- ❌ Оси координат (Frame)
- ❌ UI элементы и текст

## 🏗️ Архитектура системы

### **1. Основной класс ZoomManager**

```python
class ZoomManager:
    """Центральный менеджер системы зума"""
    
    def __init__(self):
        # Текущие значения зума по осям
        self.zoom_x = 1.0
        self.zoom_y = 1.0
        self.zoom_z = 1.0
        
        # Целевые значения для плавной анимации
        self.target_zoom_x = 1.0
        self.target_zoom_y = 1.0
        self.target_zoom_z = 1.0
        
        # Настройки
        self.zoom_speed = 1.0  # Скорость изменения зума (1.0 = 100% в секунду)
        self.zoom_limits = (0.001, 1000.0)  # Минимальный и максимальный зум
        
        # Список объектов для масштабирования
        self.scalable_objects = []
        
    def set_target_zoom(self, axis, zoom_factor):
        """Устанавливает целевой зум для указанной оси"""
        zoom_factor = max(self.zoom_limits[0], min(self.zoom_limits[1], zoom_factor))
        
        if axis == 'x':
            self.target_zoom_x = zoom_factor
        elif axis == 'y':
            self.target_zoom_y = zoom_factor
        elif axis == 'z':
            self.target_zoom_z = zoom_factor
            
    def update(self, dt):
        """Обновляет зум с плавной анимацией"""
        # Плавное приближение к целевым значениям
        self.zoom_x = lerp(self.zoom_x, self.target_zoom_x, self.zoom_speed * dt)
        self.zoom_y = lerp(self.zoom_y, self.target_zoom_y, self.zoom_speed * dt)
        self.zoom_z = lerp(self.zoom_z, self.target_zoom_z, self.zoom_speed * dt)
        
        # Обновляем все зарегистрированные объекты
        self._update_all_objects()
    
    def register_scalable(self, scalable_object):
        """Регистрирует объект для масштабирования"""
        self.scalable_objects.append(scalable_object)
        
    def unregister_scalable(self, scalable_object):
        """Удаляет объект из списка масштабируемых"""
        if scalable_object in self.scalable_objects:
            self.scalable_objects.remove(scalable_object)
    
    def _update_all_objects(self):
        """Обновляет трансформации всех зарегистрированных объектов"""
        for obj in self.scalable_objects:
            obj.update_visual_transform(self.zoom_x, self.zoom_y, self.zoom_z)
    
    def get_camera_speed_multiplier(self):
        """Возвращает множитель скорости камеры"""
        # Используем среднее значение зума для расчета скорости
        avg_zoom = (self.zoom_x + self.zoom_z) / 2.0
        return 1.0 / max(avg_zoom, 0.001)  # Избегаем деления на ноль
    
    def get_logical_position(self, camera_position):
        """Преобразует визуальную позицию камеры в логические координаты"""
        return Vec3(
            camera_position.x / self.zoom_x,  # θ
            camera_position.y / self.zoom_y,  # высота
            camera_position.z / self.zoom_z   # ω
        )
```

### **2. Базовый класс ScalableObject**

```python
class ScalableObject:
    """Базовый класс для всех масштабируемых объектов"""
    
    def __init__(self, logical_position, entity):
        # Логические координаты (исходные, неизменные)
        self.logical_position = Vec3(logical_position)
        
        # Ursina Entity для визуального отображения
        self.entity = entity
        
        # Флаги масштабирования (по умолчанию все оси)
        self.scale_x = True
        self.scale_y = True
        self.scale_z = True
        
        # Исходный размер объекта
        self.original_scale = entity.scale if hasattr(entity, 'scale') else Vec3(1, 1, 1)
        
    def update_visual_transform(self, zoom_x, zoom_y, zoom_z):
        """Обновляет визуальную позицию и масштаб объекта"""
        # Рассчитываем новую визуальную позицию
        new_position = Vec3(
            self.logical_position.x * zoom_x if self.scale_x else self.logical_position.x,
            self.logical_position.y * zoom_y if self.scale_y else self.logical_position.y,
            self.logical_position.z * zoom_z if self.scale_z else self.logical_position.z
        )
        
        # Обновляем позицию entity
        self.entity.position = new_position
        
        # Опционально: масштабируем размер объекта (для точек, связей)
        if hasattr(self.entity, 'scale'):
            scale_factor = Vec3(
                zoom_x if self.scale_x else 1.0,
                zoom_y if self.scale_y else 1.0,
                zoom_z if self.scale_z else 1.0
            )
            # Применяем масштаб, но с ограничениями для читаемости
            final_scale = Vec3(
                self.original_scale.x * min(scale_factor.x, 10.0),
                self.original_scale.y * min(scale_factor.y, 10.0), 
                self.original_scale.z * min(scale_factor.z, 10.0)
            )
            self.entity.scale = final_scale
    
    def set_logical_position(self, new_logical_position):
        """Обновляет логическую позицию объекта"""
        self.logical_position = Vec3(new_logical_position)
```

### **3. Модификация StatePoint**

```python
class StatePoint(ScalableObject):
    """Масштабируемая точка состояния системы"""
    
    def __init__(self, logical_position, is_start=False, is_goal=False, zoom_manager=None):
        # Создаем визуальный объект
        entity = Entity(
            model='sphere',
            color=color.green if is_start else (color.red if is_goal else color.blue),
            scale=0.1,
            position=logical_position  # Временная позиция
        )
        
        # Инициализируем ScalableObject
        super().__init__(logical_position, entity)
        
        # Дополнительные свойства
        self.is_start = is_start
        self.is_goal = is_goal
        self.connections = []
        
        # Создаем подпись
        self.label = Text(
            text='Старт' if is_start else 'Цель',
            parent=self.entity,
            position=(0, 0.3, 0),
            scale=50,
            color=color.white,
            billboard=True,
            background=True
        )
        
        # Регистрируем в ZoomManager
        if zoom_manager:
            zoom_manager.register_scalable(self)
```

### **4. Система управления зумом**

```python
class ZoomControls:
    """Обработчик управления зумом"""
    
    def __init__(self, zoom_manager):
        self.zoom_manager = zoom_manager
        self.zoom_step = 0.1  # 10% изменение при нажатии
        
    def handle_input(self, dt):
        """Обрабатывает ввод для управления зумом"""
        
        # Зум по оси X (θ)
        if held_keys['left arrow']:
            current = self.zoom_manager.target_zoom_x
            self.zoom_manager.set_target_zoom('x', current * (1 - self.zoom_step * dt))
            
        if held_keys['right arrow']:
            current = self.zoom_manager.target_zoom_x
            self.zoom_manager.set_target_zoom('x', current * (1 + self.zoom_step * dt))
        
        # Зум по оси Z (ω)
        if held_keys['down arrow']:
            current = self.zoom_manager.target_zoom_z
            self.zoom_manager.set_target_zoom('z', current * (1 - self.zoom_step * dt))
            
        if held_keys['up arrow']:
            current = self.zoom_manager.target_zoom_z
            self.zoom_manager.set_target_zoom('z', current * (1 + self.zoom_step * dt))
        
        # Зум по оси Y (высота) - PageUp/PageDown
        if held_keys['page up']:
            current = self.zoom_manager.target_zoom_y
            self.zoom_manager.set_target_zoom('y', current * (1 + self.zoom_step * dt))
            
        if held_keys['page down']:
            current = self.zoom_manager.target_zoom_y
            self.zoom_manager.set_target_zoom('y', current * (1 - self.zoom_step * dt))
        
        # Сброс зума - Home
        if held_keys['home']:
            self.zoom_manager.set_target_zoom('x', 1.0)
            self.zoom_manager.set_target_zoom('y', 1.0)
            self.zoom_manager.set_target_zoom('z', 1.0)
```

### **5. UI информации о зуме**

```python
class ZoomDisplay:
    """Отображение информации о зуме и координатах"""
    
    def __init__(self, zoom_manager, camera):
        self.zoom_manager = zoom_manager
        self.camera = camera
        
        # Текст с информацией о зуме
        self.zoom_info = Text(
            text="Zoom: X=1.0 Y=1.0 Z=1.0",
            position=(-0.9, 0.45),
            scale=0.7,
            color=color.white,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7)
        )
        
        # Текст с логическими координатами
        self.coords_info = Text(
            text="Position: θ=0.0 h=0.0 ω=0.0",
            position=(-0.9, 0.4),
            scale=0.7,
            color=color.yellow,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7)
        )
        
        # Инструкции по управлению
        self.controls_info = Text(
            text="Zoom: ←→ (θ), ↑↓ (ω), PgUp/PgDn (h), Home (reset)",
            position=(-0.9, 0.35),
            scale=0.6,
            color=color.gray,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7)
        )
    
    def update(self):
        """Обновляет отображаемую информацию"""
        # Информация о зуме
        self.zoom_info.text = f"Zoom: X={self.zoom_manager.zoom_x:.2f} Y={self.zoom_manager.zoom_y:.2f} Z={self.zoom_manager.zoom_z:.2f}"
        
        # Логические координаты камеры
        logical_pos = self.zoom_manager.get_logical_position(self.camera.position)
        self.coords_info.text = f"Position: θ={logical_pos.x:.2f} h={logical_pos.y:.2f} ω={logical_pos.z:.2f}"
```

## 🔄 Интеграция с существующей системой

### **Модификация SceneSetup:**

```python
class SceneSetup:
    def __init__(self):
        # Существующий код...
        
        # Добавляем систему зума
        self.zoom_manager = ZoomManager()
        self.zoom_controls = ZoomControls(self.zoom_manager)
        self.zoom_display = ZoomDisplay(self.zoom_manager, self.player)
        
    def update(self, dt):
        # Существующий код...
        
        # Обновляем систему зума
        self.zoom_controls.handle_input(dt)
        self.zoom_manager.update(dt)
        self.zoom_display.update()
        
        # Адаптируем скорость камеры
        base_speed = 5.0
        self.player.speed = base_speed * self.zoom_manager.get_camera_speed_multiplier()
```

### **Модификация PendulumScene:**

```python
class PendulumScene:
    def __init__(self):
        self.scene = SceneSetup()
        
        # Получаем доступ к zoom_manager
        self.zoom_manager = self.scene.zoom_manager
        
        # Создаем объекты с поддержкой зума
        self.start_point = StatePoint(
            logical_position=(0, 0, 0),
            is_start=True,
            zoom_manager=self.zoom_manager
        )
        
        self.goal_point = StatePoint(
            logical_position=(np.pi, 0, 0),
            is_goal=True,
            zoom_manager=self.zoom_manager
        )
```

## 📋 План реализации с отладкой

### **Этап 1: Базовая архитектура**
**Реализация:**
- [ ] Создать классы `ZoomManager` и `ScalableObject`
- [ ] Добавить базовую логику трансформации координат
- [ ] Создать тестовый объект (простой куб)

**Отладка и тестирование:**
- [ ] **Дебаг 1.1:** Проверить создание `ZoomManager` - вывести в консоль значения зума по умолчанию
- [ ] **Дебаг 1.2:** Создать тестовый куб на позиции (1, 0, 1), зарегистрировать в `ZoomManager`
- [ ] **Дебаг 1.3:** Вручную изменить `zoom_x = 2.0`, проверить что куб переместился в позицию (2, 0, 1)
- [ ] **Дебаг 1.4:** Проверить `get_logical_position()` - должен вернуть исходные координаты (1, 0, 1)
- [ ] **Дебаг 1.5:** Проверить все три оси независимо: установить разные зумы, убедиться что куб масштабируется корректно

### **Этап 2: Интеграция с StatePoint**
**Реализация:**
- [ ] Модифицировать `StatePoint` для наследования от `ScalableObject`
- [ ] Обновить создание точек в `PendulumScene`
- [ ] Добавить регистрацию в `ZoomManager`

**Отладка и тестирование:**
- [ ] **Дебаг 2.1:** Создать 3 тестовые точки на координатах (0,0,0), (π,0,0), (0,0,5)
- [ ] **Дебаг 2.2:** Вывести в консоль logical_position каждой точки при создании
- [ ] **Дебаг 2.3:** Установить zoom_x = 0.5, проверить что точка (π,0,0) переместилась в (π/2,0,0)
- [ ] **Дебаг 2.4:** Установить zoom_z = 3.0, проверить что точка (0,0,5) переместилась в (0,0,15)
- [ ] **Дебаг 2.5:** Проверить, что подписи точек остались в правильных позициях
- [ ] **Дебаг 2.6:** Убедиться что Frame (оси координат) НЕ масштабируются

### **Этап 3: Система управления**
**Реализация:**
- [ ] Реализовать `ZoomControls` с управлением стрелками
- [ ] Добавить плавную анимацию зума
- [ ] Интегрировать в `SceneSetup.update()`

**Отладка и тестирование:**
- [ ] **Дебаг 3.1:** Добавить отладочный вывод при нажатии каждой стрелки: `print(f"Key pressed: {key}, target_zoom: {value}")`
- [ ] **Дебаг 3.2:** Проверить что `target_zoom` изменяется при удержании стрелок
- [ ] **Дебаг 3.3:** Проверить плавность анимации - добавить `print(f"Current zoom: {zoom_x:.3f}, Target: {target_zoom_x:.3f}")` в update
- [ ] **Дебаг 3.4:** Тестировать каждую ось отдельно - зажать одну стрелку, убедиться что меняется только соответствующая ось
- [ ] **Дебаг 3.5:** Проверить ограничения зума - попытаться выйти за пределы (0.001, 1000)
- [ ] **Дебаг 3.6:** Тестировать сброс (Home) - убедиться что все оси возвращаются к 1.0

### **Этап 4: Адаптивная скорость**
**Реализация:**
- [ ] Реализовать `get_camera_speed_multiplier()`
- [ ] Интегрировать адаптивную скорость в `FirstPersonController`
- [ ] Настроить комфортную скорость движения

**Отладка и тестирование:**
- [ ] **Дебаг 4.1:** Добавить отладочный вывод скорости: `print(f"Zoom: {avg_zoom:.2f}, Speed multiplier: {multiplier:.3f}, Final speed: {final_speed:.2f}")`
- [ ] **Дебаг 4.2:** Протестировать экстремальные зумы:
  - При зуме 0.1x - скорость должна увеличиться в 10 раз
  - При зуме 10x - скорость должна уменьшиться в 10 раз
- [ ] **Дебаг 4.3:** Проверить что движение ощущается одинаково при разных зумах:
  - Время перемещения между двумя точками должно быть примерно одинаковое
- [ ] **Дебаг 4.4:** Убедиться что вертикальное движение (Space/Shift) тоже адаптируется
- [ ] **Дебаг 4.5:** Проверить обработку деления на ноль при очень маленьких зумах

### **Этап 5: UI и отображение**
**Реализация:**
- [ ] Создать `ZoomDisplay` для информации о зуме
- [ ] Добавить отображение логических координат
- [ ] Добавить инструкции по управлению

**Отладка и тестирование:**
- [ ] **Дебаг 5.1:** Проверить что текст зума обновляется в реальном времени при изменении зума
- [ ] **Дебаг 5.2:** Переместить камеру в известную позицию (3, 1, -8), установить зум (2, 1, 0.5), проверить правильность расчета логических координат
- [ ] **Дебаг 5.3:** Убедиться что UI элементы не перекрывают друг друга
- [ ] **Дебаг 5.4:** Проверить читаемость текста на разных фонах
- [ ] **Дебаг 5.5:** Тестировать обновление UI при быстром изменении зума
- [ ] **Дебаг 5.6:** Проверить корректность отображения при экстремальных зумах (очень большие/маленькие числа)

### **Этап 6: Расширенные объекты**
**Реализация:**
- [ ] Добавить поддержку зума для связей между точками
- [ ] Добавить поддержку границ области (theta_lims, omega_lims)
- [ ] Создать ScalableConnection класс

**Отладка и тестирование:**
- [ ] **Дебаг 6.1:** Создать связь между двумя точками, проверить что линия масштабируется вместе с точками
- [ ] **Дебаг 6.2:** Проверить что толщина линий адекватно изменяется при зуме (не становится слишком толстой/тонкой)
- [ ] **Дебаг 6.3:** Тестировать границы области - они должны масштабироваться но оставаться на правильных логических позициях
- [ ] **Дебаг 6.4:** Создать несколько точек с разными типами связей, убедиться что все корректно масштабируется
- [ ] **Дебаг 6.5:** Проверить производительность при большом количестве связей (50-100 объектов)

### **Этап 7: Оптимизация и стресс-тестирование**
**Реализация:**
- [ ] Оптимизировать производительность при большом количестве объектов
- [ ] Добавить обработку краевых случаев
- [ ] Добавить валидацию входных данных

**Отладка и тестирование:**
- [ ] **Дебаг 7.1:** Стресс-тест производительности:
  - Создать 1000 ScalableObject'ов
  - Измерить FPS при интенсивном изменении зума
  - Добавить профилирование времени выполнения `_update_all_objects()`
- [ ] **Дебаг 7.2:** Тестирование краевых случаев:
  - Зум 0.001x - проверить стабильность
  - Зум 1000x - проверить что не происходит переполнения
  - Очень быстрое изменение зума - проверить отсутствие рывков
- [ ] **Дебаг 7.3:** Тестирование математической точности:
  - Установить зум 2.0, затем 0.5 - объект должен вернуться в исходную позицию
  - Проверить накопление ошибок округления при длительной работе
- [ ] **Дебаг 7.4:** Тестирование памяти:
  - Создать и удалить множество объектов
  - Проверить корректность работы `unregister_scalable()`
  - Убедиться в отсутствии утечек памяти
- [ ] **Дебаг 7.5:** Интеграционное тестирование:
  - Запустить всю систему на 10 минут с активным использованием
  - Проверить стабильность при переключении между экстремальными зумами
  - Убедиться что все системы работают корректно вместе

### **🔧 Отладочные утилиты**

**Добавить в ZoomManager для отладки:**
```python
def debug_print_state(self):
    """Выводит текущее состояние системы зума"""
    print(f"=== ZOOM DEBUG ===")
    print(f"Current zoom: X={self.zoom_x:.3f} Y={self.zoom_y:.3f} Z={self.zoom_z:.3f}")
    print(f"Target zoom:  X={self.target_zoom_x:.3f} Y={self.target_zoom_y:.3f} Z={self.target_zoom_z:.3f}")
    print(f"Registered objects: {len(self.scalable_objects)}")
    print(f"Speed multiplier: {self.get_camera_speed_multiplier():.3f}")
    
def debug_validate_objects(self):
    """Проверяет корректность всех зарегистрированных объектов"""
    for i, obj in enumerate(self.scalable_objects):
        if not hasattr(obj, 'logical_position'):
            print(f"ERROR: Object {i} missing logical_position")
        if not hasattr(obj, 'entity'):
            print(f"ERROR: Object {i} missing entity")
```

**Горячие клавиши для отладки:**
```python
# Добавить в input_handler:
if key == 'f1':  # Вывести состояние зума
    self.zoom_manager.debug_print_state()
if key == 'f2':  # Проверить объекты
    self.zoom_manager.debug_validate_objects()
if key == 'f3':  # Сброс к дефолтным значениям
    self.zoom_manager.zoom_x = self.zoom_manager.target_zoom_x = 1.0
    self.zoom_manager.zoom_y = self.zoom_manager.target_zoom_y = 1.0  
    self.zoom_manager.zoom_z = self.zoom_manager.target_zoom_z = 1.0
```

## 🎮 Управление (итоговое)

```
← → (Left/Right Arrow)  - Зум по оси X (θ)
↑ ↓ (Up/Down Arrow)     - Зум по оси Z (ω)  
PgUp/PgDn               - Зум по оси Y (высота)
Home                    - Сброс всех зумов к 1.0x
WASD                    - Движение (адаптивная скорость)
Space/Shift             - Вверх/вниз (адаптивная скорость)
```

## 🔧 Технические детали

### **Система координат:**
- **X (θ):** Угол маятника, горизонтальная ось
- **Y (h):** Высота, вертикальная ось  
- **Z (ω):** Угловая скорость, глубина

### **Трансформация координат:**
```
visual_position = logical_position * zoom_vector
где zoom_vector = (zoom_x, zoom_y, zoom_z)
```

### **Адаптивная скорость:**
```
camera_speed = base_speed / average_zoom
где average_zoom = (zoom_x + zoom_z) / 2
```

Это ТЗ готово для передачи в Cursor или любой другой AI-ассистент для реализации!