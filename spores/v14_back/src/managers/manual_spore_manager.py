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
        self.preview_manager.update_preview()
    

    



    def _destroy_preview(self) -> None:
        """Уничтожает превью спору, предсказания и их линки. Делегирует в PreviewManager."""
        self.preview_manager.clear_all()
    
 

    def clear_all(self) -> None:
        """Очищает все ресурсы созданные ManualSporeManager."""
        print("🧹 ManualSporeManager: очистка всех ресурсов...")
        
        # 🆕 ДИАГНОСТИКА: Проверяем состояние до удаления
        print(f"   📊 Линков для удаления: {len(self.created_links)}")
        
        # 1. Уничтожаем все созданные линки
        for i, link in enumerate(self.created_links):
            try:
                # 🆕 ДИАГНОСТИКА: Проверяем состояние линка
                print(f"   🔍 Линк {i+1}: enabled={getattr(link, 'enabled', 'N/A')}, visible={getattr(link, 'visible', 'N/A')}")
                print(f"            parent={getattr(link, 'parent', 'N/A')}, model={getattr(link, 'model', 'N/A')}")
                
                # Пробуем разные способы удаления
                link.enabled = False  # Отключаем
                link.visible = False  # Скрываем
                link.parent = None    # Отвязываем от родителя
                
                destroy(link)  # Уничтожаем
                print(f"   ✅ Линк {i+1} обработан")
                
            except Exception as e:
                print(f"   ❌ Ошибка с линком {i+1}: {e}")
        
        # 🆕 ДИАГНОСТИКА: Проверяем глобальное состояние Ursina
        try:
            from ursina import scene, camera
            all_entities = [e for e in scene.entities if hasattr(e, 'model') and e.model]
            arrow_entities = [e for e in all_entities if 'arrow' in str(e.model).lower()]
            print(f"   📊 Всего entities в сцене: {len(scene.entities)}")
            print(f"   🏹 Entities со стрелками: {len(arrow_entities)}")
        except Exception as e:
            print(f"   ⚠️ Ошибка диагностики сцены: {e}")

        # 🆕 Очищаем историю групп
        self.creation_history.clear_all()
        
        self.created_links.clear()

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
        if self.creation_mode == 'tree':
            return self.spore_creator.create_tree_at_cursor()
        else:
            return self.spore_creator.create_spore_family_at_cursor()

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
        return self.tree_optimizer.update_ghost_tree_with_optimal_pairs(None)

    def reset_ghost_tree_optimization(self):
        self.tree_optimizer.reset_ghost_tree_optimization()



    @property
    def tree_depth(self):
        return self.spore_creator.tree_depth

    @tree_depth.setter
    def tree_depth(self, value):
        self.spore_creator.tree_depth = value

    def set_dt_manager(self, dt_manager) -> None:
        """Устанавливает dt_manager после инициализации."""
        self.dt_manager = dt_manager
        if self.tree_optimizer:
            self.tree_optimizer.set_dt_manager(dt_manager)
