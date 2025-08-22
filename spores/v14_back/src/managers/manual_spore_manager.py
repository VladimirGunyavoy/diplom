from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2, scene

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..logic.pendulum import PendulumSystem
from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
from ..logic.tree.pairs.extract_optimal_times_from_pairs import extract_optimal_times_from_pairs
from ..visual.prediction_visualizer import PredictionVisualizer
from ..visual.link import Link
from .manual_components import CursorTracker, PreviewManager, CreationHistory, SporeCreator, TreeOptimizer


class ManualSporeManager:
    """
    Менеджер для ручного создания спор с превью по курсору.
    
    Функциональность:
    - Показывает полупрозрачную спору в позиции курсора мыши
    - Отображает предсказания при min/max управлении
    - Создает споры + цепочки родителей по ЛКМ
    """
    
    def __init__(self, 
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager,
                 pendulum: PendulumSystem,
                 color_manager: ColorManager,
                 config: dict):
        
        self.spore_manager = spore_manager
        self.zoom_manager = zoom_manager
        self.pendulum = pendulum
        self.color_manager = color_manager
        self.config = config

        # Создаем трекер курсора
        self.cursor_tracker = CursorTracker(zoom_manager)

        # Создаем менеджер превью
        self.preview_manager = PreviewManager(
            cursor_tracker=self.cursor_tracker,
            pendulum=pendulum,
            zoom_manager=zoom_manager,
            color_manager=color_manager,
            config=config
        )

        # Создаем историю создания
        self.creation_history = CreationHistory(
            spore_manager=spore_manager,
            zoom_manager=zoom_manager
        )

                # Создаем создатель спор
        self.spore_creator = SporeCreator(
            cursor_tracker=self.cursor_tracker,
            spore_manager=spore_manager,
            zoom_manager=zoom_manager,
            color_manager=color_manager,
            pendulum=pendulum,
            config=config
        )

        # Создаем оптимизатор деревьев
        self.tree_optimizer = TreeOptimizer(pendulum, config)
        # dt_manager будет установлен позже из main_demo.py
        self.dt_manager = None
        

        

        
        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])









        print(f"   ✓ Manual Spore Manager создан (управление: {self.min_control} .. {self.max_control})")
        print(f"   📚 История групп инициализирована")





        # 🆕 Добавляем поддержку деревьев
        # НЕ НУЖНО: self.creation_mode и self.tree_depth теперь properties!
        # НЕ НУЖНО: self.active_trees = [] - споры идут в общий граф!
    
        print("   🌲 Поддержка деревьев добавлена")

        # Безопасные методы для регистрации/удаления объектов
        self._safe_register = lambda obj, reg_id: self._safe_register_impl(obj, reg_id)
        self._safe_unregister_and_destroy = lambda obj: self._safe_unregister_and_destroy_impl(obj)

    def _safe_register_impl(self, obj, reg_id: str):
        """Безопасная регистрация объекта в zoom_manager."""
        try:
            self.zoom_manager.register_object(obj, reg_id)
            setattr(obj, '_reg_id', reg_id)
        except Exception as e:
            print(f"⚠️ Регистрация {reg_id} не удалась: {e}")

    def _safe_unregister_and_destroy_impl(self, obj):
        """Безопасное удаление объекта с отменой регистрации."""
        try:
            if hasattr(obj, '_reg_id'):
                try:
                    self.zoom_manager.unregister_object(obj._reg_id)
                except:
                    pass
            destroy(obj)
        except Exception as e:
            print(f"⚠️ Удаление {getattr(obj, '_reg_id', '<no_id>')} не удалось: {e}")

# В том же файле добавить методы:










    

    
    def update_cursor_position(self) -> None:
        """Обновляет позицию превью споры по позиции курсора мыши."""
        self.cursor_tracker.update_position()
        self.preview_manager.update_preview(self.tree_optimizer)
    

    



    def _destroy_preview(self) -> None:
        """Уничтожает превью спору, предсказания и их линки. Делегирует в PreviewManager."""
        self.preview_manager.clear_all()
    
 

    def clear_all(self):
        """Очищает все ресурсы."""
        print("🧹 ManualSporeManager: очистка всех ресурсов...")

        # 🆕 ДИАГНОСТИКА: Проверяем состояние сцены Ursina перед очисткой
        try:
            from ursina import scene, camera
            all_entities = [e for e in scene.entities if hasattr(e, 'model') and e.model]
            arrow_entities = [e for e in all_entities if 'arrow' in str(e.model).lower()]
            print(f"   📊 Entities в сцене до очистки: {len(scene.entities)}")
            print(f"   🏹 Arrow entities: {len(arrow_entities)}")
        except Exception as e:
            print(f"   ⚠️ Ошибка диагностики сцены: {e}")

        self.preview_manager.clear_all()
        self.creation_history.clear_all()
        self.spore_creator.created_links.clear()

        print("   ✓ Все ресурсы очищены")

        # 🆕 ДИАГНОСТИКА: Проверяем состояние сцены Ursina после очистки
        try:
            from ursina import scene, camera
            all_entities = [e for e in scene.entities if hasattr(e, 'model') and e.model]
            arrow_entities = [e for e in all_entities if 'arrow' in str(e.model).lower()]
            print(f"   📊 Entities в сцене после очистки: {len(scene.entities)}")
            print(f"   🏹 Arrow entities: {len(arrow_entities)}")
        except Exception as e:
            print(f"   ⚠️ Ошибка диагностики сцены: {e}")

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.clear_all()  # Используем новый метод
        print("   ✓ Manual Spore Manager уничтожен")

    def delete_last_spore_group(self) -> bool:
        return self.creation_history.delete_last_group()

    def get_groups_history_stats(self) -> dict:
        return self.creation_history.get_stats()

    def print_groups_history_stats(self) -> None:
        self.creation_history.print_stats()

    def create_spore_at_cursor(self):
        """Создает споры или дерево в зависимости от режима."""
        result = self.spore_creator.create_spore_family_at_cursor() if self.creation_mode == 'spores' else self.spore_creator.create_tree_at_cursor(self.tree_optimizer)
        if result:
            spores, links = result
            self.creation_history.add_group(spores, links)
            return spores
        return None

    def toggle_creation_mode(self) -> None:
        """Переключает режим создания."""
        current_mode = self.spore_creator.get_creation_mode()
        if current_mode == 'spores':
            self.spore_creator.set_creation_mode('tree')
            self.preview_manager.set_creation_mode('tree')
            print("🌲 Режим: Создание деревьев")
        else:
            self.spore_creator.set_creation_mode('spores')
            self.preview_manager.set_creation_mode('spores')
            print("🌟 Режим: Создание спор")

    def set_tree_depth(self, depth: int) -> None:
        """Устанавливает глубину дерева."""
        self.spore_creator.set_tree_depth(depth)
        print(f"🌲 Глубина дерева: {self.spore_creator.get_tree_depth()}")

    @property
    def creation_mode(self) -> str:
        """Возвращает текущий режим создания."""
        return self.spore_creator.get_creation_mode()

    @property
    def tree_depth(self) -> int:
        """Возвращает текущую глубину дерева."""
        return self.spore_creator.get_tree_depth()

    @property
    def created_links(self) -> List[Link]:
        """Возвращает список созданных линков."""
        return self.spore_creator.created_links

    def update_ghost_tree_with_optimal_pairs(self):
        """Обновляет призрачное дерево оптимальными dt."""
        return self.tree_optimizer.update_ghost_tree_with_optimal_pairs(self.preview_manager.preview_spore)

    def reset_ghost_tree_optimization(self):
        self.tree_optimizer.reset_ghost_tree_optimization()

    def set_dt_manager(self, dt_manager) -> None:
        """Устанавливает dt_manager после инициализации."""
        self.dt_manager = dt_manager
        if dt_manager:
            self.tree_optimizer.set_dt_manager(dt_manager)
        else:
            print("⚠️ Предупреждение: dt_manager равен None")

    # 🆕 ДОБАВЛЕНО: Методы для тестирования компонентов
    def test_cursor_tracker(self) -> None:
        """Тестирует CursorTracker: получение позиции мыши."""
        try:
            pos = self.cursor_tracker.get_current_position()
            print(f"✅ CursorTracker: позиция получена {pos}")
            return True
        except Exception as e:
            print(f"❌ CursorTracker: ошибка {e}")
            return False

    def test_preview_manager(self) -> None:
        """Тестирует PreviewManager: отображение превью и предсказаний."""
        try:
            # Проверяем что превью спора создана
            if self.preview_manager.preview_spore:
                print("✅ PreviewManager: превью спора создана")
                return True
            else:
                print("❌ PreviewManager: превью спора не создана")
                return False
        except Exception as e:
            print(f"❌ PreviewManager: ошибка {e}")
            return False

    def test_spore_creator(self) -> None:
        """Тестирует SporeCreator: создание семей спор и деревьев."""
        try:
            # Проверяем что создатель инициализирован
            if hasattr(self.spore_creator, 'created_links'):
                print("✅ SporeCreator: инициализирован корректно")
                return True
            else:
                print("❌ SporeCreator: отсутствуют необходимые атрибуты")
                return False
        except Exception as e:
            print(f"❌ SporeCreator: ошибка {e}")
            return False

    def test_creation_history(self) -> None:
        """Тестирует CreationHistory: управление группами."""
        try:
            stats = self.creation_history.get_stats()
            print(f"✅ CreationHistory: статистика получена {stats}")
            return True
        except Exception as e:
            print(f"❌ CreationHistory: ошибка {e}")
            return False

    def test_tree_optimizer(self) -> None:
        """Тестирует TreeOptimizer: оптимизация dt."""
        try:
            # Проверяем что оптимизатор инициализирован
            if self.tree_optimizer and hasattr(self.tree_optimizer, 'set_dt_manager'):
                print("✅ TreeOptimizer: инициализирован корректно")
                return True
            else:
                print("❌ TreeOptimizer: отсутствуют необходимые методы")
                return False
        except Exception as e:
            print(f"❌ TreeOptimizer: ошибка {e}")
            return False

    def run_full_test_suite(self) -> None:
        """Запускает полный набор тестов всех компонентов."""
        print("\n🧪 ЗАПУСК ПОЛНОГО ТЕСТИРОВАНИЯ КОМПОНЕНТОВ")
        print("=" * 50)

        tests = [
            ("CursorTracker", self.test_cursor_tracker),
            ("PreviewManager", self.test_preview_manager),
            ("SporeCreator", self.test_spore_creator),
            ("CreationHistory", self.test_creation_history),
            ("TreeOptimizer", self.test_tree_optimizer)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n📋 Тестируем {test_name}:")
            if test_func():
                passed += 1

        print("\n" + "=" * 50)
        print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {passed}/{total} тестов пройдено")

        if passed == total:
            print("🎉 Все тесты пройдены успешно!")
        else:
            print(f"⚠️  {total - passed} тест(ов) не пройдено")

        print("=" * 50)
