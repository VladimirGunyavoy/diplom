"""
Полноценный тест новой архитектуры Spore со всеми классами
Воссоздает функциональность оригинального 1.py с новой архитектурой
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

# --- Основные импорты из нашего проекта ---
from src.spawn_area import SpawnArea
from src.cost import Cost
from src.color_manager import ColorManager
from src.scene_setup import SceneSetup
from src.frame import Frame
from src.zoom_manager import ZoomManager
from src.scalable import Scalable, ScalableFrame, ScalableFloor
from src.pendulum import PendulumSystem
from src.window_manager import WindowManager
from src.spore import Spore  # НОВАЯ АРХИТЕКТУРА!
from src.spore_manager import SporeManager
from src.param_manager import ParamManager

print("=== ТЕСТ НОВОЙ АРХИТЕКТУРЫ SPORE ===")
print("Загружаем новый Spore с разделенной логикой и визуализацией...")

# create app
app = Ursina()

# create color manager
color_manager = ColorManager()

# create scene setup
scene_setup = SceneSetup(
    init_position=(0.263, -1.451, 0.116), 
    init_rotation_x=42.750, 
    init_rotation_y=-12.135, 
    color_manager=color_manager
)

frame = Frame(color_manager=color_manager)
scene_setup.frame = frame  # Добавляем frame в scene_setup
window_manager = WindowManager(color_manager=color_manager)

settings_param = ParamManager(0, show=False, color_manager=color_manager)

# create pendulum
pendulum = PendulumSystem(dt=0.1, damping=0.3)

zoom_manager = ZoomManager(scene_setup, color_manager=color_manager)

def set_initial_zoom():
    # Set initial zoom from image
    target_a = 1.602
    zoom_manager.a_transformation = target_a
    inv_point_2d = zoom_manager.identify_invariant_point()
    p = np.array([inv_point_2d[0], 0, inv_point_2d[1]], dtype=float)
    b = (1 - target_a) * p
    zoom_manager.b_translation = b
    zoom_manager.update_transform()

invoke(set_initial_zoom, delay=0.02)

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

# === ДЕМОНСТРАЦИЯ НОВОЙ АРХИТЕКТУРЫ ===
print("\n1. Создание целевой споры...")
goal_position = (np.pi, 0)
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
print(f"   ✓ 3D позиция: {goal.real_position}")
print(f"   ✓ 2D логика: {goal.logic.position_2d}")
print(f"   ✓ Тип объекта: {type(goal)}")

print("\n2. Создание стартовой споры...")
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
print(f"   ✓ 3D позиция: {spore.real_position}")
print(f"   ✓ 2D логика: {spore.logic.position_2d}")
print(f"   ✓ Стоимость: {spore.cost:.3f}")
print(f"   ✓ Тип объекта: {type(spore)}")

print("\n3. Создание области генерации...")
spawn_area = SpawnArea(
    focus1=spore, 
    focus2=goal, 
    eccentricity=0.998, 
    dimensions=2, 
    resolution=64, 
    mode='line', 
    color_manager=color_manager
)

print("\n4. Создание поверхности стоимости...")
# --- Поверхность стоимости (Cost) ---
cost_surface = Cost(
    goal_position=goal,
    spawn_area=spawn_area,
    color_manager=color_manager
)

# === НАСТРОЙКА РАДУЖНОЙ ПОВЕРХНОСТИ ===
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

print("\n5. Добавление в менеджеры...")
spore_manager.add_spore(goal)
spore_manager.add_spore(spore)

print("\n6. Проверка ghost spores...")
print(f"   ✓ Ghost spore создана: {spore_manager.ghost_spore is not None}")
print(f"   ✓ Ghost spores массив: {spore_manager.ghost_spores is not None}")
if spore_manager.ghost_spores:
    print(f"   ✓ Количество ghost spores: {len(spore_manager.ghost_spores)}")
print(f"   ✓ Ghost link создан: {spore_manager.ghost_link is not None}")

zoom_manager.register_object(goal)
zoom_manager.register_object(spore)
zoom_manager.register_object(spawn_area)
zoom_manager.register_object(cost_surface)

# apply initial transform to all registered objects
zoom_manager.update_transform()

print("\n7. UI элементы отключены...")
# UI элементы закомментированы по запросу пользователя
# architecture_info = Text(...)
# test_info = Text(...)
# operation_info = Text(...)

# Заглушки для совместимости
operation_counter = {'count': 0}
class FakeText:
    def __init__(self):
        self.text = ""
operation_info = FakeText()

def test_architecture_methods():
    """Тестирование методов новой архитектуры"""
    print("\n=== ТЕСТ МЕТОДОВ НОВОЙ АРХИТЕКТУРЫ ===")
    
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    # Тест SporeLogic методов
    print("1. Тестирование SporeLogic:")
    old_pos = spore.logic.position_2d.copy()
    old_cost = spore.logic.get_cost()
    
    # Эволюция
    control = np.random.uniform(-1.0, 1.0)
    new_pos = spore.logic.evolve(state=old_pos, control=control)
    # Обновляем позицию после эволюции
    spore.logic.set_position_2d(new_pos)
    new_cost = spore.logic.get_cost()
    
    print(f"   • evolve({control:.2f}): {old_pos} -> {new_pos}")
    print(f"   • cost: {old_cost:.3f} -> {new_cost:.3f}")
    
    # Синхронизация с визуализацией
    spore.sync_with_logic()
    print(f"   • sync_with_logic: 3D={spore.real_position}")
    
    # Тест клонирования
    clone = spore.clone()
    print(f"   • clone: type={type(clone)}, cost={clone.cost:.3f}")
    
    # Тест генерации управлений
    controls = spore.logic.sample_controls(5)
    print(f"   • sample_controls(5): {controls}")
    
    # Тест обратной совместимости
    print("\n2. Тестирование обратной совместимости:")
    pos_2d = spore.calc_2d_pos()
    pos_3d = spore.calc_3d_pos([1.0, 2.0])
    evolved_3d = spore.evolve_3d(control=0.5)
    
    print(f"   • calc_2d_pos: {pos_2d}")
    print(f"   • calc_3d_pos([1,2]): {pos_3d}")
    print(f"   • evolve_3d(0.5): {evolved_3d}")
    
    print("=== ТЕСТ ЗАВЕРШЕН ===")
    
    # Клон автоматически очистится сборщиком мусора

def create_test_spore():
    """Создание тестовой споры"""
    try:
        operation_counter['count'] += 1
        operation_info.text = f"Операций: {operation_counter['count']}"
        
        # Случайная позиция
        x = np.random.uniform(-2, 2)
        z = np.random.uniform(-2, 4)
        
        # Конвертируем goal_position в правильный формат
        goal_pos_3d = np.array([goal_position[0], 0, goal_position[1]])
        
        print(f"\nСоздание споры в позиции ({x:.2f}, 0, {z:.2f})...")
        
        test_spore = Spore(
            pendulum=pendulum,
            dt=dt,
            goal_position=goal_pos_3d,
            scale=ball_size,
            position=(x, 0, z),
            color_manager=color_manager
        )
        
        spore_manager.add_spore(test_spore)
        zoom_manager.register_object(test_spore)
        
        print(f"✓ Новая тестовая спора создана:")
        print(f"   3D: {test_spore.real_position}")
        print(f"   2D: {test_spore.logic.position_2d}")
        print(f"   Cost: {test_spore.cost:.3f}")
        print(f"   Всего спор: {len(spore_manager.objects)}")
        
    except Exception as e:
        print(f"ОШИБКА при создании споры: {e}")
        import traceback
        traceback.print_exc()

def evolve_all_spores():
    """Эволюция всех спор"""
    operation_counter['count'] += 1
    operation_info.text = f"Операций: {operation_counter['count']}"
    
    print(f"\nЭволюция {len(spore_manager.objects)} спор:")
    for i, s in enumerate(spore_manager.objects):
        if hasattr(s, 'logic') and hasattr(s.logic, 'evolve'):  # Только новые споры с логикой
            try:
                old_cost = s.cost
                control = np.random.uniform(-1.0, 1.0)
                
                # Проверяем позицию перед эволюцией
                pos_2d = s.logic.position_2d
                print(f"   Спора {i}: позиция {pos_2d}, тип {type(pos_2d)}")
                
                # Эволюцируем и обновляем позицию
                new_pos_2d = s.logic.evolve(state=pos_2d, control=control)
                s.logic.set_position_2d(new_pos_2d)
                s.sync_with_logic()
                
                print(f"   Спора {i}: cost {old_cost:.3f} -> {s.cost:.3f}")
            except Exception as e:
                print(f"   ОШИБКА в споре {i}: {e}")
                print(f"   Тип споры: {type(s)}")
                if hasattr(s, 'logic'):
                    print(f"   Тип логики: {type(s.logic)}")
                    if hasattr(s.logic, 'position_2d'):
                        print(f"   position_2d: {s.logic.position_2d}, тип: {type(s.logic.position_2d)}")
        else:
            print(f"   Спора {i}: пропускаем (старая архитектура или нет логики)")

def print_architecture_info():
    """Вывод информации об архитектуре"""
    print("\n=== ИНФОРМАЦИЯ О НОВОЙ АРХИТЕКТУРЕ ===")
    print("1. SporeLogic:")
    print("   • Чистая 2D математика")
    print("   • Эволюция, стоимость, генерация управлений")
    print("   • Без зависимостей от GUI")
    print("\n2. SporeVisual:")
    print("   • 3D визуализация")
    print("   • Конвертация координат 2D ↔ 3D")
    print("   • Управление цветами и трансформациями")
    print("\n3. Spore (объединенный):")
    print("   • Агрегирует SporeLogic + наследует SporeVisual")
    print("   • 100% обратная совместимость")
    print("   • Принцип единственной ответственности")
    print("=====================================")

def print_ghost_info():
    """Вывод информации о ghost spores"""
    print("\n=== ДИАГНОСТИКА GHOST SPORES ===")
    print(f"Всего объектов в spore_manager: {len(spore_manager.objects)}")
    
    # Детальная информация о всех объектах
    for i, obj in enumerate(spore_manager.objects):
        is_goal = getattr(obj, 'is_goal', False)
        print(f"   [{i}] Позиция: {obj.real_position}, is_goal: {is_goal}")
    
    # Проверяем функцию get_last_active_spore
    last_active = spore_manager.get_last_active_spore()
    if last_active:
        print(f"\nПоследняя активная спора: {last_active.real_position}")
        
        # Проверяем эволюцию от этой споры
        test_evolution = last_active.evolve_3d()
        print(f"Эволюция с нулевым управлением: {test_evolution}")
    else:
        print("\nПоследняя активная спора: НЕ НАЙДЕНА!")
    
    print(f"\nGhost spore создана: {spore_manager.ghost_spore is not None}")
    if spore_manager.ghost_spore:
        print(f"   Позиция ghost spore: {spore_manager.ghost_spore.real_position}")
    
    print(f"Ghost spores массив: {spore_manager.ghost_spores is not None}")
    if spore_manager.ghost_spores:
        print(f"   Количество: {len(spore_manager.ghost_spores)}")
        for i, ghost in enumerate(spore_manager.ghost_spores):
            print(f"   Ghost {i}: {ghost.real_position}")
    
    print(f"Ghost link: {spore_manager.ghost_link is not None}")
    if spore_manager.ghost_link:
        print(f"   От (parent): {spore_manager.ghost_link.parent_spore.real_position}")
        print(f"   К (child): {spore_manager.ghost_link.child_spore.real_position}")
    print("=====================================")

def update():
    """Основной цикл обновления"""
    scene_setup.update(time.dt)
    zoom_manager.identify_invariant_point()
    settings_param.update()
    
    # Обновляем ghost spores каждый кадр для отображения разных управлений
    if len(spore_manager.objects) > 0:
        spore_manager.update_ghost_spore()  # Основная ghost spore (нулевое управление)
        spore_manager.sample_ghost_spores(5)  # 5 ghost spores с разными управлениями

def input(key):
    """Обработчик пользовательского ввода"""
    # Базовые обработчики (F обрабатывается здесь как в оригинале!)
    scene_setup.input_handler(key)
    zoom_manager.input_handler(key)
    spore_manager.input_handler(key)  # Тут обрабатывается оригинальная F
    settings_param.input_handler(key)
    spawn_area.input_handler(key)
    
    # Наши тестовые команды (НЕ перехватываем F!)
    if key == 't':
        test_architecture_methods()
    elif key == 'shift+f':  # Используем Shift+F для тестовой функции
        create_test_spore()
    elif key == 'g':
        evolve_all_spores()
    elif key == 'i':
        print_architecture_info()
    elif key == 'h':
        print_ghost_info()
    elif key == 'u':
        # Принудительное обновление ghost spores
        print("\nПринудительное обновление ghost spores...")
        spore_manager.update_ghost_spore()
        spore_manager.sample_ghost_spores(5)
        print("Готово! Нажмите H для проверки.")

print("\n=== ГОТОВ К ТЕСТИРОВАНИЮ ===")
print("ОРИГИНАЛЬНАЯ ЛОГИКА ВОССТАНОВЛЕНА:")
print("• F - спора в позиции ghost link (как в оригинале)")
print("• Shift+F - случайная тестовая спора")
print("• T - тест методов новой архитектуры")
print("• G - эволюция всех спор")
print("• I - информация об архитектуре")
print("• H - диагностика ghost spores")
print("• U - принудительное обновление ghost spores")
print("• Все остальные команды работают как в оригинале")
print("=============================================")

if __name__ == '__main__':
    app.run() 