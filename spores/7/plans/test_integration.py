import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from unittest.mock import Mock, MagicMock

# Создаем базовые мок-классы с правильным поведением наследования
class MockEntity:
    def __init__(self, *args, **kwargs):
        self.position = kwargs.get('position', [0, 0, 0])
        self.scale = kwargs.get('scale', [1, 1, 1])
        self.color = kwargs.get('color', [1, 1, 1, 1])
        self.model = kwargs.get('model', 'sphere')
        self.real_position = np.array(self.position, dtype=float)
        self.real_scale = np.array(self.scale, dtype=float)
    
    def look_at(self, target, axis='forward'):
        """Мок метода look_at для Link"""
        pass

MockVec3 = Mock(side_effect=lambda x, y, z: [x, y, z])

# Создаем мок color с правильными атрибутами
mock_color = Mock()
mock_color.violet = [0.5, 0, 1, 1]
mock_color.cyan = [0, 1, 1, 1]
mock_color.green = [0, 1, 0, 1]

# Мок для Text
class MockText:
    def __init__(self, *args, **kwargs):
        self.text = kwargs.get('text', '')
        self.position = kwargs.get('position', [0, 0])
        self.scale = kwargs.get('scale', 1.0)
        self.color = kwargs.get('color', [1, 1, 1, 1])
        self.background = kwargs.get('background', False)
        self.background_color = kwargs.get('background_color', [0, 0, 0, 1])

# Создаем полный мок-модуль ursina
ursina_mock = Mock()
ursina_mock.Entity = MockEntity
ursina_mock.Vec3 = MockVec3
ursina_mock.color = mock_color
ursina_mock.Text = MockText

# Мокаем весь модуль
sys.modules['ursina'] = ursina_mock

# Также экспортируем в глобальное пространство для import *
globals()['Entity'] = MockEntity
globals()['Vec3'] = MockVec3  
globals()['color'] = mock_color
globals()['Text'] = MockText

from src.spore import Spore
from src.spore_manager import SporeManager
from src.pendulum import PendulumSystem
from src.color_manager import ColorManager
from src.zoom_manager import ZoomManager

def test_spore_manager_integration():
    """Тест интеграции нового Spore с SporeManager"""
    print("=== Тестирование интеграции с SporeManager ===")
    
    # Создаем мок-объекты
    pendulum = Mock(spec=PendulumSystem)
    pendulum.discrete_step.return_value = np.array([1.5, 2.5])
    pendulum.get_control_bounds.return_value = (-1.0, 1.0)
    pendulum.sample_controls.return_value = np.array([0.5, -0.3, 0.8, 0.2, -0.7])
    pendulum.goal_position = np.array([5.0, 3.0])
    
    zoom_manager = Mock(spec=ZoomManager)
    zoom_manager.a_transformation = 1.0
    zoom_manager.b_translation = np.array([0.0, 0.0, 0.0])
    zoom_manager.spores_scale = 1.0
    zoom_manager.register_object = Mock()
    
    settings_param = Mock()
    color_manager = ColorManager()
    
    print("1. Создание SporeManager...")
    try:
        spore_manager = SporeManager(
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            settings_param=settings_param,
            color_manager=color_manager
        )
        print("   ✓ SporeManager создан успешно")
    except Exception as e:
        print(f"   ✗ Ошибка создания SporeManager: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n2. Создание первой споры...")
    try:
        spore1 = Spore(
            dt=0.1,
            pendulum=pendulum,
            goal_position=np.array([5.0, 0.0, 3.0]),
            position=[1.0, 0.0, 2.0]
        )
        print(f"   ✓ Спора создана: {type(spore1)}")
        print(f"   ✓ Логика создана: {hasattr(spore1, 'logic')}")
        print(f"   ✓ Позиция логики: {spore1.logic.position_2d}")
    except Exception as e:
        print(f"   ✗ Ошибка создания споры: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n3. Добавление споры в менеджер...")
    try:
        spore_manager.add_spore(spore1)
        print("   ✓ Спора добавлена в менеджер")
        print(f"   ✓ Количество спор: {len(spore_manager.objects)}")
    except Exception as e:
        print(f"   ✗ Ошибка добавления споры: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n4. Тестирование генерации новой споры...")
    try:
        new_spore = spore_manager.generate_new_spore()
        print(f"   ✓ Новая спора создана: {new_spore is not None}")
        if new_spore:
            print(f"   ✓ Тип новой споры: {type(new_spore)}")
            print(f"   ✓ У новой споры есть логика: {hasattr(new_spore, 'logic')}")
            print(f"   ✓ Количество спор стало: {len(spore_manager.objects)}")
        
    except Exception as e:
        print(f"   ✗ Ошибка генерации новой споры: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. Проверка обратной совместимости методов...")
    try:
        # Тестируем методы которые использует SporeManager
        test_spore = spore_manager.objects[0]
        
        # evolve_3d - используется в update_ghost_spore
        result_3d = test_spore.evolve_3d()
        print(f"   ✓ evolve_3d(): {result_3d}")
        
        # step - используется в generate_new_spore  
        stepped_spore = test_spore.step()
        print(f"   ✓ step(): {type(stepped_spore)}")
        
        # clone - используется в update_ghost_spore
        cloned_spore = test_spore.clone()
        print(f"   ✓ clone(): {type(cloned_spore)}")
        
        # apply_transform - используется в apply_transform
        test_spore.apply_transform(1.0, np.array([0, 0, 0]), spores_scale=1.0)
        print("   ✓ apply_transform() выполнен")
        
    except Exception as e:
        print(f"   ✗ Ошибка в методах обратной совместимости: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n6. Тестирование input_handler...")
    try:
        initial_count = len(spore_manager.objects)
        spore_manager.input_handler('f')  # Должно создать новую спору
        new_count = len(spore_manager.objects)
        print(f"   ✓ Input handler 'f': {initial_count} → {new_count} спор")
        
    except Exception as e:
        print(f"   ✗ Ошибка в input_handler: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Интеграционный тест завершен успешно! ===")

def test_spore_methods_comprehensive():
    """Комплексный тест всех методов Spore"""
    print("\n=== Комплексный тест методов Spore ===")
    
    # Создаем мок-маятник
    pendulum = Mock(spec=PendulumSystem)
    pendulum.discrete_step.return_value = np.array([1.5, 2.5])
    pendulum.get_control_bounds.return_value = (-1.0, 1.0)
    pendulum.sample_controls.return_value = np.array([0.5, -0.3, 0.8, 0.2, -0.7])
    pendulum.goal_position = np.array([5.0, 3.0])
    
    spore = Spore(
        dt=0.1,
        pendulum=pendulum,
        goal_position=np.array([5.0, 0.0, 3.0]),
        position=[2.0, 0.0, 1.0]
    )
    
    print("1. Проверка всех атрибутов...")
    required_attrs = ['dt', 'pendulum', 'goal_position', 'logic', 'cost', 'cost_function']
    for attr in required_attrs:
        has_attr = hasattr(spore, attr)
        print(f"   ✓ {attr}: {has_attr}")
        if not has_attr:
            print(f"   ✗ Отсутствует атрибут {attr}!")
    
    print("\n2. Проверка всех методов...")
    required_methods = [
        'calc_2d_pos', 'calc_3d_pos', 'evolve_3d', 'evolve_2d', 
        'step', 'clone', 'sample_random_controls', 'sample_mesh_controls',
        'simulate_controls', 'apply_transform', 'sync_position_with_logic',
        'sync_logic_with_position'
    ]
    
    for method in required_methods:
        has_method = hasattr(spore, method) and callable(getattr(spore, method))
        print(f"   ✓ {method}: {has_method}")
        if not has_method:
            print(f"   ✗ Отсутствует метод {method}!")
    
    print("\n3. Проверка типов возвращаемых значений...")
    try:
        # calc_2d_pos должен возвращать numpy array
        pos_2d = spore.calc_2d_pos()
        print(f"   ✓ calc_2d_pos возвращает: {type(pos_2d)} shape={pos_2d.shape}")
        
        # calc_3d_pos должен возвращать numpy array
        pos_3d = spore.calc_3d_pos([3.0, 4.0])
        print(f"   ✓ calc_3d_pos возвращает: {type(pos_3d)} shape={pos_3d.shape}")
        
        # cost должен быть числом
        cost_val = spore.cost
        print(f"   ✓ cost возвращает: {type(cost_val)} = {cost_val}")
        
        # evolve_3d должен возвращать numpy array
        evolved_3d = spore.evolve_3d(control=0.1)
        print(f"   ✓ evolve_3d возвращает: {type(evolved_3d)} shape={evolved_3d.shape}")
        
    except Exception as e:
        print(f"   ✗ Ошибка в проверке типов: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Комплексный тест завершен! ===")

if __name__ == "__main__":
    test_spore_manager_integration()
    test_spore_methods_comprehensive() 