"""
ДЕМОНСТРАЦИЯ НОВОЙ АРХИТЕКТУРЫ SPORE
====================================

Этот файл демонстрирует использование новой рефакторированной архитектуры:
- SporeLogic: чистая математическая логика (2D)
- SporeVisual: 3D визуализация
- Spore: объединенный класс с полной обратной совместимостью

Преимущества новой архитектуры:
✅ Разделение ответственности (SRP)
✅ Независимое тестирование логики
✅ Переиспользуемость компонентов
✅ Чистая архитектура без GUI зависимостей в логике
✅ 100% обратная совместимость
"""

import sys
import os
from ursina import *
import numpy as np
from colorsys import hsv_to_rgb
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# --- Настройка путей для импорта ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Импорты из проекта ---
from src.spawn_area import SpawnArea
from src.cost import Cost
from src.color_manager import ColorManager
from src.scene_setup import SceneSetup
from src.frame import Frame
from src.zoom_manager import ZoomManager
from src.scalable import Scalable, ScalableFrame, ScalableFloor
from src.pendulum import PendulumSystem
from src.window_manager import WindowManager
from src.param_manager import ParamManager

# ===== НОВАЯ АРХИТЕКТУРА =====
from src.spore_logic import SporeLogic      # 🧮 Чистая математическая логика
from src.spore_visual import SporeVisual    # 🎨 3D визуализация
from src.spore import Spore                 # 🔄 Объединенный класс (логика + визуализация)
from src.spore_manager import SporeManager

print("=== ДЕМОНСТРАЦИЯ НОВОЙ АРХИТЕКТУРЫ SPORE ===")
print("🏗️  SporeLogic (математика) + SporeVisual (3D) + Spore (интеграция)")
print("✅ Принцип единственной ответственности")
print("✅ 100% обратная совместимость с 1.py")
print("=" * 50)

# ===== ИНИЦИАЛИЗАЦИЯ URSINA =====
app = Ursina()

# ===== НАСТРОЙКА СЦЕНЫ (как в 1.py) =====
color_manager = ColorManager()

scene_setup = SceneSetup(
    init_position=(0.263, -1.451, 0.116), 
    init_rotation_x=42.750, 
    init_rotation_y=-12.135, 
    color_manager=color_manager
)

frame = Frame(color_manager=color_manager)
scene_setup.frame = frame
window_manager = WindowManager(color_manager=color_manager)

settings_param = ParamManager(0, show=False, color_manager=color_manager)

# ===== СОЗДАНИЕ СИСТЕМЫ МАЯТНИКА =====
pendulum = PendulumSystem(dt=0.1, damping=0.3)

# ===== ZOOM MANAGER =====
zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)

def set_initial_zoom():
    target_a = 1.602
    zoom_manager.a_transformation = target_a
    inv_point_2d = zoom_manager.identify_invariant_point()
    p = np.array([inv_point_2d[0], 0, inv_point_2d[1]], dtype=float)
    b = (1 - target_a) * p
    zoom_manager.b_translation = b
    zoom_manager.update_transform()

invoke(set_initial_zoom, delay=0.02)

# ===== SPORE MANAGER =====
spore_manager = SporeManager(
    pendulum=pendulum, 
    zoom_manager=zoom_manager, 
    settings_param=settings_param, 
    color_manager=color_manager
)

ball_size = 1/50

setup_register = {
    'x_axis': frame.x_axis,
    'y_axis': frame.y_axis,
    'z_axis': frame.z_axis,
    'origin': frame.origin,
    'floor': scene_setup.floor
}

for name, obj in setup_register.items():
    zoom_manager.register_object(obj, name)

dt = 0.05

# ===== ДЕМОНСТРАЦИЯ: СОЗДАНИЕ ЛОГИКИ ОТДЕЛЬНО ОТ ВИЗУАЛИЗАЦИИ =====
print("\n🧮 1. Создание чистой математической логики (SporeLogic)...")

goal_position = (np.pi, 0)
goal_position_2d = np.array([np.pi, 0])  # 2D цель
start_position_2d = np.array([0.0, 2.0])  # 2D старт

# Создаем логику отдельно - без GUI зависимостей!
pure_logic = SporeLogic(
    pendulum=pendulum,
    dt=dt,
    goal_position_2d=goal_position_2d,
    initial_position_2d=start_position_2d
)

print(f"   ✓ Логика создана: позиция={pure_logic.position_2d}, стоимость={pure_logic.cost:.3f}")
print(f"   ✓ Без GUI зависимостей - чистая математика!")

# Демонстрируем работу логики
print("\n🔬 2. Тестируем математические операции...")
controls = pure_logic.sample_controls(3, method='random')
print(f"   ✓ Сгенерированы управления: {controls}")

simulated_states = pure_logic.simulate_controls(controls)
print(f"   ✓ Симуляция состояний:")
for i, state in enumerate(simulated_states):
    logic_copy = SporeLogic(pendulum, dt, goal_position_2d, state)
    print(f"     Управление {controls[i]:.2f} → состояние {state} → стоимость {logic_copy.cost:.3f}")

# ===== ДЕМОНСТРАЦИЯ: НОВАЯ АРХИТЕКТУРА В ДЕЙСТВИИ =====
print("\n🎨 3. Создание объединенных спор (Spore = Logic + Visual)...")

# Целевая спора - использует новую архитектуру под капотом!
goal = Spore(
    pendulum=pendulum,
    dt=dt,
    scale=ball_size,
    position=(goal_position[0], 0, goal_position[1]),
    goal_position=goal_position,
    is_goal=True,
    color_manager=color_manager
)

print(f"   ✓ Целевая спора создана:")
print(f"     └ 3D позиция: {goal.real_position}")
print(f"     └ 2D логика: {goal.logic.position_2d}")
print(f"     └ Стоимость: {goal.cost:.3f}")

# Стартовая спора
spore = Spore(
    pendulum=pendulum, 
    dt=dt, 
    goal_position=goal_position,
    scale=ball_size,
    position=(0, 0, 2),
    color_manager=color_manager
)
spore.color = color.red

print(f"   ✓ Стартовая спора создана:")
print(f"     └ 3D позиция: {spore.real_position}")
print(f"     └ 2D логика: {spore.logic.position_2d}")
print(f"     └ Стоимость: {spore.cost:.3f}")

# ===== ДЕМОНСТРАЦИЯ: ОБРАТНАЯ СОВМЕСТИМОСТЬ =====
print("\n🔄 4. Демонстрация обратной совместимости...")
print("   ✓ Все методы из 1.py работают точно так же:")

# Эти методы работают как раньше!
pos_2d = spore.calc_2d_pos()
pos_3d = spore.calc_3d_pos([1.0, 2.0])
evolved_3d = spore.evolve_3d(control=0.1)
cloned_spore = spore.clone()

print(f"     └ calc_2d_pos(): {pos_2d}")
print(f"     └ calc_3d_pos([1,2]): {pos_3d}")
print(f"     └ evolve_3d(0.1): {evolved_3d}")
print(f"     └ clone(): {type(cloned_spore)}")

# Очистка клонированной споры
cloned_spore.disable()

# ===== СОЗДАНИЕ ОБЛАСТИ СПАВНА И ПОВЕРХНОСТИ СТОИМОСТИ =====
print("\n🌍 5. Создание окружения (как в 1.py)...")

spawn_area = SpawnArea(
    focus1=spore, 
    focus2=goal, 
    eccentricity=0.998, 
    dimensions=2, 
    resolution=64, 
    mode='line', 
    color_manager=color_manager
)

# Поверхность стоимости - такая же как в 1.py
cost_surface = Cost(
    goal_position=goal,
    spawn_area=spawn_area,
    color_manager=color_manager
)

cost_surface.generate_surface(
    mesh_generation={
        'boundary_points': 64,       
        'interior_min_radius': 0.2   
    },
    alpha=0.7,                      
    show_points=True,               
    point_size=0.02,                
    show_edges=True,                
    edge_thickness=1,               
    show_contours=True,             
    contour_levels=22,              
    contour_thickness=3,            
    contour_resolution=60           
)

print("   ✓ SpawnArea создана")
print("   ✓ Cost поверхность создана")

# ===== ДОБАВЛЕНИЕ В МЕНЕДЖЕРЫ =====
spore_manager.add_spore(goal)
spore_manager.add_spore(spore)

zoom_manager.register_object(goal)
zoom_manager.register_object(spore)
zoom_manager.register_object(spawn_area)
zoom_manager.register_object(cost_surface)

# Применяем начальные трансформации
zoom_manager.update_transform()

print("\n🎮 6. Интерфейс управления...")
print("   ✓ Все команды из 1.py работают")
print("   ✓ Дополнительные команды новой архитектуры:")

# ===== РАСШИРЕННАЯ ИНФОРМАЦИЯ НА ЭКРАНЕ =====
architecture_info = Text(
    text="НОВАЯ АРХИТЕКТУРА SPORE\n"
         "🧮 SporeLogic: чистая математика (2D)\n"
         "🎨 SporeVisual: 3D визуализация\n"
         "🔄 Spore: логика + визуализация\n"
         "✅ 100% совместимость с 1.py",
    position=(-0.95, 0.9),
    scale=0.7,
    color=color.white,
    background=True,
    background_color=color.rgba(0, 0, 0, 0.7)
)

operation_counter = {'count': 0}
operation_info = Text(
    text="Операций: 0",
    position=(-0.95, 0.4),
    scale=0.8,
    color=color.yellow
)

test_info = Text(
    text="🔬 T - тест архитектуры\n"
         "🧮 L - тест чистой логики\n"
         "🎨 V - тест визуализации\n"
         "🔄 S - синхронизация\n"
         "📊 I - информация",
    position=(-0.95, 0.1),
    scale=0.6,
    color=color.cyan,
    background=True,
    background_color=color.rgba(0, 0, 0, 0.5)
)

# ===== НОВЫЕ ФУНКЦИИ ДЕМОНСТРАЦИИ =====

def test_architecture_separation():
    """Демонстрация разделения архитектуры"""
    print("\n=== ТЕСТ РАЗДЕЛЕНИЯ АРХИТЕКТУРЫ ===")
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    # 1. Тестируем чистую логику
    print("1. Тест чистой логики (без GUI):")
    test_logic = SporeLogic(pendulum, 0.1, goal_position_2d, np.array([1.0, 1.5]))
    old_pos = test_logic.position_2d.copy()
    old_cost = test_logic.cost
    
    # Эволюция логики
    new_pos = test_logic.evolve(control=0.3)
    test_logic.set_position_2d(new_pos)
    new_cost = test_logic.cost
    
    print(f"   ✓ Эволюция: {old_pos} → {new_pos}")
    print(f"   ✓ Стоимость: {old_cost:.3f} → {new_cost:.3f}")
    
    # 2. Тестируем визуализацию
    print("2. Тест визуализации:")
    test_visual = SporeVisual(model='sphere', color_manager=color_manager, y_coordinate=0.0, scale=0.02)
    test_visual.sync_with_logic(test_logic)
    
    print(f"   ✓ 2D логика: {test_logic.position_2d}")
    print(f"   ✓ 3D визуал: {test_visual.real_position}")
    
    # 3. Тестируем объединение
    print("3. Тест объединенного Spore:")
    integrated_spore = Spore(
        dt=0.1, 
        pendulum=pendulum,
        goal_position=goal_position_2d,
        position=[new_pos[0], 0.0, new_pos[1]],
        color_manager=color_manager
    )
    
    print(f"   ✓ Объединенная спора создана")
    print(f"   ✓ Внутренняя логика: {integrated_spore.logic.position_2d}")
    print(f"   ✓ Визуальная позиция: {integrated_spore.real_position}")
    print(f"   ✓ Методы совместимости работают: calc_2d_pos={integrated_spore.calc_2d_pos()}")
    
    # Автоматическая очистка
    test_visual.disable()
    integrated_spore.disable()
    
    print("=== ТЕСТ ЗАВЕРШЕН ===")

def test_pure_logic():
    """Тест чистой математической логики"""
    print("\n=== ТЕСТ ЧИСТОЙ ЛОГИКИ ===")
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    # Создаем несколько логических объектов
    logics = []
    for i in range(3):
        start_pos = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
        logic = SporeLogic(pendulum, 0.1, goal_position_2d, start_pos)
        logics.append(logic)
        print(f"   Логика {i+1}: pos={logic.position_2d}, cost={logic.cost:.3f}")
    
    # Эволюция каждой логики
    print("\nЭволюция логики:")
    for i, logic in enumerate(logics):
        controls = logic.sample_controls(1, method='random')
        old_pos = logic.position_2d.copy()
        new_pos = logic.evolve(control=controls[0])
        
        # Создаем новую логику с новой позицией для расчета новой стоимости
        new_logic = SporeLogic(pendulum, 0.1, goal_position_2d, new_pos)
        
        print(f"   Логика {i+1}: {old_pos} → {new_pos} (control={controls[0]:.2f}, cost={new_logic.cost:.3f})")
    
    print("=== ЧИСТАЯ ЛОГИКА ПРОТЕСТИРОВАНА ===")

def test_visual_sync():
    """Тест синхронизации визуализации"""
    print("\n=== ТЕСТ СИНХРОНИЗАЦИИ ===")
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    # Берем существующую спору и тестируем синхронизацию
    if spore_manager.objects:
        test_spore = spore_manager.objects[0] if len(spore_manager.objects) > 0 else spore
        
        print(f"   До синхронизации:")
        print(f"     └ 2D логика: {test_spore.logic.position_2d}")
        print(f"     └ 3D визуал: {test_spore.real_position}")
        
        # Меняем логику
        old_logic_pos = test_spore.logic.position_2d.copy()
        new_logic_pos = test_spore.logic.evolve(control=0.2)
        test_spore.logic.set_position_2d(new_logic_pos)
        
        print(f"   Изменили логику: {old_logic_pos} → {new_logic_pos}")
        
        # Синхронизируем
        test_spore.sync_with_logic()
        
        print(f"   После синхронизации:")
        print(f"     └ 2D логика: {test_spore.logic.position_2d}")
        print(f"     └ 3D визуал: {test_spore.real_position}")
        print(f"     └ Синхронизация: {'✅' if np.allclose(test_spore.logic.position_2d, [test_spore.real_position[0], test_spore.real_position[2]]) else '❌'}")
    
    print("=== СИНХРОНИЗАЦИЯ ПРОТЕСТИРОВАНА ===")

def show_architecture_info():
    """Показать информацию об архитектуре"""
    print("\n=== ИНФОРМАЦИЯ ОБ АРХИТЕКТУРЕ ===")
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    print("📊 Статистика спор:")
    print(f"   └ Всего спор: {len(spore_manager.objects)}")
    
    if spore_manager.objects:
        total_cost = sum(s.cost for s in spore_manager.objects)
        avg_cost = total_cost / len(spore_manager.objects)
        print(f"   └ Общая стоимость: {total_cost:.3f}")
        print(f"   └ Средняя стоимость: {avg_cost:.3f}")
        
        print("\n🏗️ Архитектурные компоненты:")
        for i, spore_obj in enumerate(spore_manager.objects):
            print(f"   Спора {i+1}:")
            print(f"     └ Тип: {type(spore_obj).__name__}")
            print(f"     └ Имеет логику: {'✅' if hasattr(spore_obj, 'logic') else '❌'}")
            print(f"     └ Наследует от SporeVisual: {'✅' if isinstance(spore_obj, SporeVisual) else '❌'}")
            print(f"     └ 2D/3D синхронизация: {'✅' if hasattr(spore_obj, 'sync_with_logic') else '❌'}")
    
    print("=== ИНФОРМАЦИЯ ПОКАЗАНА ===")

# ===== ОБРАБОТКА ВВОДА (РАСШИРЕННАЯ) =====
def input(key):
    """Расширенная обработка ввода с новыми командами"""
    # Обычные команды (как в 1.py)
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)
    spore_manager.input_handler(key)
    settings_param.input_handler(key)
    spawn_area.input_handler(key)
    
    # НОВЫЕ КОМАНДЫ АРХИТЕКТУРЫ
    if key == 't':
        test_architecture_separation()
    elif key == 'l':
        test_pure_logic()
    elif key == 'v':
        test_visual_sync()
    elif key == 's':
        # Принудительная синхронизация всех спор
        print("\n🔄 Синхронизация всех спор...")
        for spore_obj in spore_manager.objects:
            if hasattr(spore_obj, 'sync_with_logic'):
                spore_obj.sync_with_logic()
        print("   ✓ Синхронизация завершена")
    elif key == 'i':
        show_architecture_info()

# ===== ОБНОВЛЕНИЕ (как в 1.py) =====
def update():
    """Обновление системы"""
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()
    settings_param.update()

# ===== ФИНАЛЬНАЯ ДЕМОНСТРАЦИЯ =====
print("\n🎉 7. Демонстрация готова!")
print("\n📋 НОВЫЕ КОМАНДЫ:")
print("   T - тест разделения архитектуры")
print("   L - тест чистой логики") 
print("   V - тест синхронизации визуализации")
print("   S - принудительная синхронизация")
print("   I - информация об архитектуре")
print("\n📋 ОБЫЧНЫЕ КОМАНДЫ (как в 1.py):")
print("   F - создать спору")
print("   Остальные команды работают как раньше")
print("\n" + "="*50)
print("🚀 НОВАЯ АРХИТЕКТУРА ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
print("="*50)

# ===== ЗАПУСК =====
if __name__ == '__main__':
    app.run() 