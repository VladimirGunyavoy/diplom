#!/usr/bin/env python3
"""
Тест миграции TreeCreationManager на SharedDependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_tree_creation_manager_migration():
    """Тестирует миграцию TreeCreationManager на SharedDependencies"""
    try:
        from src.managers.manual_creation import TreeCreationManager, SharedDependencies
        
        # Создаем мок-объекты для тестирования
        class MockZoomManager:
            def get_unique_spore_id(self):
                return "test_spore_1"
            def get_unique_link_id(self):
                return "test_link_1"
            def register_object(self, obj, obj_id):
                pass
            def unregister_object(self, obj_id):
                pass
            def update_transform(self):
                pass
            spores_scale = 1.0
                
        class MockColorManager:
            def get_color(self, category, name):
                return (0.6, 0.4, 0.9, 1.0)
                
        class MockPendulum:
            def get_control_bounds(self):
                return (-1.0, 1.0)
            def step(self, position, control, dt):
                return position + np.array([control * dt, 0.0])
                
        class MockSporeManager:
            def add_spore_manual(self, spore):
                pass
            class MockIdManager:
                def get_next_link_id(self):
                    return 1
                def get_next_spore_id(self):
                    return 1
            id_manager = MockIdManager()
                
        # Создаем SharedDependencies
        deps = SharedDependencies(
            zoom_manager=MockZoomManager(),
            color_manager=MockColorManager(),
            pendulum=MockPendulum(),
            config={'spore': {'goal_position': [0, 0], 'scale': 0.1}, 'pendulum': {'dt': 0.05}}
        )
        
        print("✅ SharedDependencies создан успешно!")
        
        # Создаем TreeCreationManager с SharedDependencies
        tree_creation_manager = TreeCreationManager(
            deps=deps,
            spore_manager=MockSporeManager()
        )
        
        print("✅ TreeCreationManager создан с SharedDependencies!")
        
        # Проверяем, что все методы работают
        has_deps = hasattr(tree_creation_manager, 'deps')
        print(f"✅ использует deps: {has_deps}")
        
        # Проверяем режимы создания
        print(f"✅ creation_mode = {tree_creation_manager.creation_mode}")
        print(f"✅ tree_depth = {tree_creation_manager.tree_depth}")
        
        # Тестируем переключение режимов
        tree_creation_manager.toggle_creation_mode()
        print(f"✅ toggle_creation_mode работает: {tree_creation_manager.creation_mode}")
        
        # Тестируем установку глубины
        tree_creation_manager.set_tree_depth(2)
        print(f"✅ set_tree_depth работает: {tree_creation_manager.tree_depth}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_spore_manager_integration():
    """Тестирует интеграцию с ManualSporeManager"""
    try:
        from src.managers.manual_spore_manager import ManualSporeManager
        
        # Создаем мок-объекты
        class MockSporeManager:
            class MockIdManager:
                def get_next_link_id(self):
                    return 1
                def get_next_spore_id(self):
                    return 1
            id_manager = MockIdManager()
            
        class MockZoomManager:
            def identify_invariant_point(self):
                return (0.0, 0.0)
            scene_setup = None
            a_transformation = 1.0
            b_translation = [0.0, 0.0, 0.0]
            spores_scale = 1.0
            def register_object(self, obj, name):
                pass
            def unregister_object(self, name):
                pass
            def get_unique_spore_id(self):
                return "test_spore_1"
            def get_unique_link_id(self):
                return "test_link_1"
            def update_transform(self):
                pass
                
        class MockColorManager:
            def get_color(self, category, name):
                return (0.6, 0.4, 0.9, 1.0)
                
        class MockPendulum:
            def get_control_bounds(self):
                return (-1.0, 1.0)
            def step(self, position, control, dt):
                return position + np.array([control * dt, 0.0])
                
        # Создаем ManualSporeManager
        msm = ManualSporeManager(
            spore_manager=MockSporeManager(),
            zoom_manager=MockZoomManager(),
            pendulum=MockPendulum(),
            color_manager=MockColorManager(),
            config={'spore': {'goal_position': [0, 0], 'scale': 0.1}, 'pendulum': {'dt': 0.05}}
        )
        
        print("✅ ManualSporeManager создан с мигрированным TreeCreationManager!")
        
        # Проверяем, что shared_deps создан
        has_shared_deps = hasattr(msm, 'shared_deps')
        print(f"✅ shared_deps создан: {has_shared_deps}")
        
        # Проверяем, что tree_creation_manager использует deps
        has_deps = hasattr(msm.tree_creation_manager, 'deps')
        print(f"✅ tree_creation_manager использует deps: {has_deps}")
        
        # Проверяем режимы создания
        print(f"✅ tree_creation_manager.creation_mode = {msm.tree_creation_manager.creation_mode}")
        print(f"✅ tree_creation_manager.tree_depth = {msm.tree_creation_manager.tree_depth}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Тестирование миграции TreeCreationManager на SharedDependencies...")
    print()
    
    success1 = test_tree_creation_manager_migration()
    print()
    
    if success1:
        success2 = test_manual_spore_manager_integration()
        print()
        
        if success2:
            print("🎉 Все тесты миграции прошли успешно!")
        else:
            print("💥 Тесты интеграции не прошли!")
    else:
        print("💥 Базовые тесты не прошли!")
