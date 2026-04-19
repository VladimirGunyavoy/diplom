# Техническое задание: Рефакторинг архитектуры класса Spore

## Проблема
Текущий класс `Spore` объединяет в себе два типа функциональности:
1. **Математическую логику** - эволюция системы, расчеты, симуляция
2. **Визуализацию** - наследование от `Scalable` (который наследует `Entity` из ursina), управление цветом, трансформации

Это нарушает принцип единственной ответственности (SRP) и делает код менее гибким.

## Анализ текущей архитектуры

### Иерархия классов:
```
Entity (ursina) 
└── Scalable (real_position, real_scale, apply_transform)
    ├── Spore (логика + визуализация)
    ├── SpawnArea 
    ├── Cost
    ├── Link
    └── Frame элементы
```

### Взаимодействия Spore с другими компонентами:

1. **SporeManager**:
   - Управляет коллекцией спор
   - Создает ghost-споры для предпросмотра
   - Создает Link между спорами
   - Регистрирует споры в ZoomManager

2. **ZoomManager**:
   - Применяет трансформации ко всем Scalable объектам
   - Использует `real_position`/`real_scale` для хранения "истинных" координат
   - Применяет масштабирование через `apply_transform(a, b, spores_scale)`

3. **Link**:
   - Соединяет parent_spore и child_spore визуальными стрелками
   - Использует `real_position` спор для расчета геометрии
   - Наследует от Scalable

4. **PendulumSystem**:
   - Предоставляет математическую модель (discrete_step, linearize_continuous)
   - Не зависит от визуализации

5. **Cost**:
   - Использует `goal_position.real_position` для расчетов
   - Работает в системе координат XZ (2D проекция)

### Ключевые методы Spore:

**Математические (работают в 2D):**
- `evolve_2d(state, control)` - 2D эволюция через pendulum.discrete_step (основная логика)
- `evolve_3d(state, control)` - обертка для совместимости (конвертирует 2D→3D)
- `step(state, control)` - создает новую спору после эволюции
- `clone()` - клонирование споры
- `calc_2d_pos()` - возвращает текущую 2D позицию (x, z)
- `calc_3d_pos(pos_2d)` - конвертирует 2D в 3D для визуализации
- `sample_random_controls(N)`, `sample_mesh_controls(N)` - генерация управлений
- `simulate_controls(controls)` - симуляция множественных управлений

**Визуальные:**
- `apply_transform(a, b, spores_scale)` - применение зума и масштаба
- Управление цветом через ColorManager
- Хранение визуальных параметров для клонирования

### Система координат:
- **SporeLogic.position_2d** - 2D логическое состояние (x, z) - основа всех вычислений
- **SporeVisual.real_position** - 3D координаты для визуализации (x, y, z)
- **SporeVisual.position** - экранные координаты после apply_transform
- **Конвертация**: 2D логика ↔ 3D визуализация через y_coordinate (обычно 0)

## Целевая архитектура

```
Entity (ursina)
└── Scalable
    └── SporeVisual (только визуализация + координация)
        └── Spore (агрегация SporeLogic + наследование от SporeVisual)

SporeLogic (чистая математика, без наследования от Entity)
```

## Детальный план рефакторинга

### 1. Создание класса SporeLogic
**Файл:** `spore_logic.py`

**Ответственность:**
- Хранение математического состояния
- Работа с PendulumSystem
- Все математические операции
- Независимость от GUI

```python
class SporeLogic:
    def __init__(self, pendulum: PendulumSystem, dt: float, goal_position: np.array, 
                 initial_position_2d: np.array):
        self.pendulum = pendulum
        self.dt = dt
        self.goal_position_2d = np.array(goal_position)  # Только 2D координаты (x, z)
        self.position_2d = np.array(initial_position_2d)  # Только 2D логическое состояние
        
        # Функция стоимости работает в 2D
        self.cost_function = lambda x=self.position_2d: np.sum((x - self.goal_position_2d) ** 2)
        self.cost = self.cost_function()
    
    def get_position_2d(self) -> np.array:
        """Возвращает текущую 2D позицию (x, z)"""
        return self.position_2d.copy()
    
    def set_position_2d(self, position_2d: np.array):
        """Устанавливает новую 2D позицию"""
        self.position_2d = np.array(position_2d)
        self.cost = self.cost_function()
    
    def evolve(self, state: np.array = None, control: float = 0) -> np.array:
        """Эволюция в 2D пространстве состояний"""
        if state is None or len(state) != 2:
            state = self.position_2d
        next_state = self.pendulum.discrete_step(state, control)
        return next_state
    
    def step(self, state: np.array = None, control: float = 0) -> 'SporeLogic':
        """Создает новую SporeLogic после эволюции"""
        if state is None:
            state = self.position_2d
        new_position_2d = self.evolve(state=state, control=control)
        
        new_logic = SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position=self.goal_position_2d,
            initial_position_2d=new_position_2d
        )
        return new_logic
    
    def clone(self) -> 'SporeLogic':
        """Клонирование логического состояния"""
        return SporeLogic(
            pendulum=self.pendulum,
            dt=self.dt,
            goal_position=self.goal_position_2d,
            initial_position_2d=self.position_2d
        )
    
    def sample_random_controls(self, N: int) -> np.array:
        a, b = self.pendulum.get_control_bounds()
        return np.random.uniform(a, b, N)
    
    def sample_mesh_controls(self, N: int) -> np.array:
        a, b = self.pendulum.get_control_bounds()
        return np.linspace(a, b, N)
    
    def simulate_controls(self, controls: np.array) -> np.array:
        """Симулирует множественные управления от текущего состояния"""
        states = np.zeros((len(controls), 2))
        for i, control in enumerate(controls):
            states[i] = self.evolve(state=self.position_2d, control=control)
        return states
```

### 2. Создание класса SporeVisual
**Файл:** `spore_visual.py`

**Ответственность:**
- Визуальное представление
- Интеграция с системой трансформаций
- Синхронизация с логическим состоянием

```python
class SporeVisual(Scalable):
    def __init__(self, model='sphere', color_manager=None, is_goal=False, 
                 y_coordinate=0.0, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        
        # Управление цветом
        if color_manager is None:
            color_manager = ColorManager()
        self.color_manager = color_manager
        
        color_type = 'goal' if is_goal else 'default'
        self.color = self.color_manager.get_color('spore', color_type)
        self.is_goal = is_goal
        
        # Y координата для 3D визуализации (обычно 0)
        self.y_coordinate = y_coordinate
    
    def apply_transform(self, a, b, **kwargs):
        """Переопределяем для поддержки spores_scale"""
        spores_scale = kwargs.get('spores_scale', 1.0)
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a * spores_scale
    
    def sync_with_logic(self, spore_logic: SporeLogic):
        """Синхронизирует визуальное состояние с логическим (конвертирует 2D в 3D)"""
        pos_2d = spore_logic.get_position_2d()
        # Преобразуем 2D логическую позицию в 3D визуальную: (x, y, z)
        self.real_position = np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def get_logic_position_2d(self) -> np.array:
        """Извлекает 2D позицию из 3D визуальной позиции для логики"""
        return np.array([self.real_position[0], self.real_position[2]])
    
    def set_color_type(self, color_type: str):
        """Изменяет тип цвета (default, ghost, goal)"""
        self.color = self.color_manager.get_color('spore', color_type)
```

### 3. Рефакторинг класса Spore
**Файл:** `spore.py`

```python
class Spore(SporeVisual):
    def __init__(self, dt, pendulum: PendulumSystem, goal_position, 
                 model='sphere', color_manager=None, is_goal=False, 
                 position=(0,0,0), *args, **kwargs):
        
        # Инициализация визуальной части
        super().__init__(model=model, color_manager=color_manager, 
                        is_goal=is_goal, position=position, *args, **kwargs)
        
        # Извлекаем 2D позицию для логики и 2D целевую позицию
        initial_pos_2d = self.get_logic_position_2d()
        goal_pos_2d = np.array([goal_position[0], goal_position[2]]) if len(goal_position) == 3 else np.array(goal_position)
        
        # Создание логической части (работает только в 2D)
        self.logic = SporeLogic(
            pendulum=pendulum, 
            dt=dt, 
            goal_position=goal_pos_2d,
            initial_position_2d=initial_pos_2d
        )
        
        # Сохранение параметров для клонирования
        self._init_params = {
            'dt': dt,
            'pendulum': pendulum,
            'goal_position': goal_position,
            'model': model,
            'color_manager': color_manager,
            'is_goal': is_goal,
            'args': args,
            'kwargs': kwargs
        }
        
        # Синхронизация (обновляет real_position из логики)
        self.sync_with_logic(self.logic)
    
    # Методы-проксы к логике (для обратной совместимости)
    def evolve_2d(self, state=None, control=0):
        """2D эволюция - основной метод логики"""
        return self.logic.evolve(state, control)
    
    def evolve_3d(self, state=None, control=0):
        """3D эволюция - конвертирует результат 2D эволюции в 3D координаты"""
        if state is None:
            state_2d = self.logic.get_position_2d()
        else:
            # Извлекаем 2D из 3D состояния если передали 3D
            state_2d = np.array([state[0], state[2]]) if len(state) == 3 else state
        
        next_state_2d = self.logic.evolve(state_2d, control)
        # Конвертируем обратно в 3D для совместимости
        return np.array([next_state_2d[0], self.y_coordinate, next_state_2d[1]])
    
    def step(self, state=None, control=0):
        """Создает новую спору после эволюции"""
        if state is not None and len(state) == 3:
            # Конвертируем 3D в 2D для логики
            state_2d = np.array([state[0], state[2]])
        else:
            state_2d = state
            
        new_logic = self.logic.step(state_2d, control)
        return self._create_from_logic(new_logic)
    
    def clone(self):
        """Клонирование споры"""
        new_logic = self.logic.clone()
        return self._create_from_logic(new_logic)
    
    def _create_from_logic(self, spore_logic: SporeLogic) -> 'Spore':
        """Создает новую Spore из SporeLogic"""
        params = self._init_params.copy()
        
        # Конвертируем 2D позицию логики в 3D для визуализации
        pos_2d = spore_logic.get_position_2d()
        pos_3d = (pos_2d[0], self.y_coordinate, pos_2d[1])
        params['kwargs']['position'] = pos_3d
        
        new_spore = Spore(
            dt=params['dt'],
            pendulum=params['pendulum'],
            goal_position=params['goal_position'],
            model=params['model'],
            color_manager=params['color_manager'],
            is_goal=params['is_goal'],
            *params['args'],
            **params['kwargs']
        )
        new_spore.logic = spore_logic
        new_spore.sync_with_logic(spore_logic)
        return new_spore
    
    # Методы для обратной совместимости с текущим API
    def calc_2d_pos(self):
        """Возвращает текущую 2D позицию"""
        return self.logic.get_position_2d()
    
    def calc_3d_pos(self, pos_2d):
        """Конвертирует 2D позицию в 3D для визуализации"""
        return np.array([pos_2d[0], self.y_coordinate, pos_2d[1]])
    
    def sample_random_controls(self, N):
        return self.logic.sample_random_controls(N)
    
    def sample_mesh_controls(self, N):
        return self.logic.sample_mesh_controls(N)
    
    def simulate_controls(self, controls):
        return self.logic.simulate_controls(controls)
    
    @property
    def cost(self):
        return self.logic.cost
    
    @property
    def cost_function(self):
        return self.logic.cost_function
```

### 4. Обновление зависимых классов

#### SporeManager
- Минимальные изменения, так как публичный API Spore сохранен
- Возможно потребуется обновление методов работы с ghost-спорами

#### Link
- Должен продолжать работать, так как использует `real_position`
- `real_position` остается доступным через SporeVisual

#### Cost
- Должен продолжать работать без изменений

### 5. Миграционная стратегия

**Этап 1: Создание новых классов (1-2 дня)**
1. Создать `SporeLogic` с тестами
2. Создать `SporeVisual` 
3. Протестировать независимо от существующего кода

**Этап 2: Интеграция (1-2 дня)**
1. Обновить `Spore` для использования новой архитектуры
2. Запустить существующие скрипты для проверки

**Этап 3: Тестирование и отладка (1 день)**
1. Проверить все функции приложения
2. Исправить найденные проблемы

**Этап 4: Очистка (0.5 дня)**
1. Удалить устаревший код
2. Обновить документацию

## Преимущества нового подхода

1. **Разделение ответственности** - логика и визуализация независимы
2. **Тестируемость** - можно тестировать математику без GUI
3. **Переиспользование** - SporeLogic можно использовать в других контекстах
4. **Гибкость** - легко менять визуализацию без затрагивания логики
5. **Производительность** - возможность оптимизации каждой части отдельно
6. **Масштабируемость** - можно создавать сотни SporeLogic без визуализации

## Риски и меры по их снижению

1. **Нарушение обратной совместимости**
   - ✅ Сохранить все публичные методы в `Spore`
   - ✅ Протестировать с существующими скриптами

2. **Усложнение кода**
   - ✅ Хорошая документация и примеры
   - ✅ Четкие интерфейсы между классами

3. **Снижение производительности**
   - ✅ Минимизировать копирование данных
   - ✅ Профилирование до и после

## Критерии успеха

- [ ] Существующий скрипт `scripts/1.py` работает без изменений
- [ ] Все математические операции дают те же результаты
- [ ] Визуализация работает корректно
- [ ] SporeLogic можно создать без GUI зависимостей
- [ ] Код стал более читаемым и поддерживаемым
- [ ] Покрытие тестами увеличилось

## Примечания по реализации

1. **Координаты**: SporeLogic работает только в 2D (`position_2d`), SporeVisual конвертирует в 3D
2. **Синхронизация**: SporeVisual.sync_with_logic конвертирует 2D→3D через `y_coordinate`
3. **Клонирование**: Каждый clone создает новую SporeLogic с тем же 2D состоянием
4. **Совместимость**: evolve_3d и calc_3d_pos сохранены для обратной совместимости
5. **Конвертация**: Все 3D↔2D конвертации используют формулу: 2D=(x,z), 3D=(x,y_coord,z) 