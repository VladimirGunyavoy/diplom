from ursina import *
import numpy as np
from typing import Dict, Optional, Tuple

from ..core.spore import Spore
from ..utils.scalable import Scalable
from .color_manager import ColorManager
from ..visual.ui_manager import UIManager
from ..visual.scene_setup import SceneSetup
# from .link import Link

class ZoomManager:
    def __init__(self, scene_setup: SceneSetup, 
                 color_manager: Optional[ColorManager] = None, 
                 ui_manager: Optional[UIManager] = None):
        self.zoom_x: float = 1
        self.zoom_y: float = 1
        self.zoom_z: float = 1

        self.zoom_fact: float = 1 + 1/8

        self.a_transformation: float = 1.0
        self.b_translation: np.ndarray = np.array([0, 0, 0], dtype=float)

        self.spores_scale: float = 1.0
        self.common_scale: float = 1.0

        # Используем переданный ColorManager или создаем новый
        self.color_manager: ColorManager = color_manager if color_manager is not None else ColorManager()

        # Используем переданный UIManager или создаем новый
        self.ui_manager: UIManager = ui_manager if ui_manager is not None else UIManager(self.color_manager)

        self.objects: Dict[str, Scalable] = {}
        self.scene_setup: SceneSetup = scene_setup

        # UI элементы для zoom manager теперь создаются в UI_setup.py
        self.ui_elements: Dict = {}

        self.invariant_point: Tuple[float, float, float] = (0, 0, 0)

        # Глобальные счетчики для уникальных ID
        self._global_spore_counter = 0
        self._global_link_counter = 0
        self._global_object_counter = 0
        
        # Флаг для отключения автоматического вывода
        self.auto_print_enabled = True

    def register_object(self, obj: Scalable, name: Optional[str] = None) -> None:
        if name is None:
            name = f"obj_{len(self.objects)}"
        self.objects[name] = obj
        obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)
        
        # Проверяем, является ли объект призрачным
        is_ghost = (
            getattr(obj, 'is_ghost', False) or
            'ghost' in name.lower() or
            (hasattr(obj, 'color') and hasattr(obj.color, 'a') and getattr(obj.color, 'a', 1.0) < 0.8)
        )
        
        # 🔍 ОТЛАДКА: Показываем информацию для ВСЕХ объектов (включая призраки)
        if self.auto_print_enabled:
            if not is_ghost:
                # Для обычных объектов показываем полную информацию
                self.print_quick_info(name, obj)
    
    def print_quick_info(self, name: str, obj: Scalable) -> None:
        """
        Показывает краткую информацию о добавленном объекте.
        Вызывается автоматически при регистрации не-призрачных объектов.
        """
        obj_type = type(obj).__name__
        obj_id = getattr(obj, 'id', 'N/A')
        
        print(f"🔍 Добавлен объект: {name} ({obj_type}, ID: {obj_id})")
        print(f"   📊 Всего объектов в системе: {len(self.objects)}")
        
        # 🔍 ДЕТАЛЬНАЯ ОТЛАДКА ПОЗИЦИЙ ДЛЯ СПОР
        if obj_type == 'Spore':
            print(f"   📍 ДЕТАЛЬНАЯ ПОЗИЦИЯ СПОРЫ:")
            print(f"      x, y, z: ({obj.x:.6f}, {obj.y:.6f}, {obj.z:.6f})")
            print(f"      real_position: {obj.real_position}")
            if hasattr(obj, 'logic') and obj.logic:
                print(f"      logic.position_2d: {obj.logic.position_2d}")
            print(f"      is_ghost: {getattr(obj, 'is_ghost', False)}")
            print(f"      color: {getattr(obj, 'color', 'N/A')}")
        
        # Показываем количество призраков, если они есть
        ghost_count = 0
        for n, o in self.objects.items():
            if (getattr(o, 'is_ghost', False) or
                'ghost' in n.lower() or
                (hasattr(o, 'color') and hasattr(o.color, 'a') and getattr(o.color, 'a', 1.0) < 0.8)):
                ghost_count += 1
        
        if ghost_count > 0:
            print(f"   👻 Призрачных объектов: {ghost_count}")
        
        print()  # Пустая строка для читаемости
    
    def enable_auto_print(self) -> None:
        """Включает автоматический вывод при добавлении объектов."""
        self.auto_print_enabled = True
        print("🔍 Автоматический вывод включен")
    
    def disable_auto_print(self) -> None:
        """Отключает автоматический вывод при добавлении объектов."""
        self.auto_print_enabled = False
        print("🔍 Автоматический вывод отключен")
    
    def print_all_objects(self) -> None:
        """
        Выводит информацию о всех объектах, зарегистрированных в ZoomManager.
        Показывает количество объектов, их типы и имена (исключая призрачные объекты).
        """
        print(f"\n🔍 ZoomManager: Объекты после добавления нового")
        print(f"   📊 Всего объектов: {len(self.objects)}")
        
        if not self.objects:
            print("   📭 Нет зарегистрированных объектов")
            return
        
        # Фильтруем призрачные объекты
        non_ghost_objects = {}
        ghost_count = 0
        
        for name, obj in self.objects.items():
            # Проверяем, является ли объект призрачным
            is_ghost = (
                getattr(obj, 'is_ghost', False) or  # Атрибут is_ghost
                'ghost' in name.lower() or          # Имя содержит "ghost"
                (hasattr(obj, 'color') and hasattr(obj.color, 'a') and getattr(obj.color, 'a', 1.0) < 0.8)  # Прозрачность
            )
            
            if is_ghost:
                ghost_count += 1
            else:
                non_ghost_objects[name] = obj
        
        # Показываем статистику по призракам
        if ghost_count > 0:
            print(f"   👻 Призрачных объектов: {ghost_count} (скрыты из вывода)")
        
        if not non_ghost_objects:
            print("   📭 Нет не-призрачных объектов")
            return
        
        # Группируем не-призрачные объекты по типам
        object_types = {}
        for name, obj in non_ghost_objects.items():
            obj_type = type(obj).__name__
            if obj_type not in object_types:
                object_types[obj_type] = []
            object_types[obj_type].append(name)
        
        # Выводим статистику по типам
        print(f"   📋 Типы объектов (не-призраки):")
        for obj_type, names in object_types.items():
            print(f"      • {obj_type}: {len(names)} шт.")
            # Показываем первые несколько имен для каждого типа
            if len(names) <= 5:
                for name in names:
                    print(f"        - {name}")
            else:
                for name in names[:3]:
                    print(f"        - {name}")
                print(f"        ... и еще {len(names) - 3} объектов")
        
        # Показываем последние добавленные не-призрачные объекты
        print(f"   🆕 Последние добавленные объекты (не-призраки):")
        recent_objects = list(non_ghost_objects.items())[-3:]  # Последние 3 не-призрачных объекта
        for name, obj in recent_objects:
            obj_type = type(obj).__name__
            obj_id = getattr(obj, 'id', 'N/A')
            print(f"      • {name} ({obj_type}, ID: {obj_id})")
        
        print()  # Пустая строка для читаемости
    
    def show_all_objects(self) -> None:
        """
        Ручной вызов для показа всех объектов в ZoomManager.
        Аналогичен print_all_objects(), но не вызывается автоматически.
        """
        self.print_all_objects()
    
    def show_all_objects_with_ghosts(self) -> None:
        """
        Показывает ВСЕ объекты, включая призрачные.
        Используется для полной отладки.
        """
        print(f"\n🔍 ZoomManager: ВСЕ объекты (включая призраки)")
        print(f"   📊 Всего объектов: {len(self.objects)}")
        
        if not self.objects:
            print("   📭 Нет зарегистрированных объектов")
            return
        
        # Группируем все объекты по типам
        object_types = {}
        ghost_objects = {}
        
        for name, obj in self.objects.items():
            obj_type = type(obj).__name__
            if obj_type not in object_types:
                object_types[obj_type] = []
            object_types[obj_type].append(name)
            
            # Отдельно собираем призрачные объекты
            is_ghost = (
                getattr(obj, 'is_ghost', False) or
                'ghost' in name.lower() or
                (hasattr(obj, 'color') and hasattr(obj.color, 'a') and getattr(obj.color, 'a', 1.0) < 0.8)
            )
            if is_ghost:
                ghost_objects[name] = obj
        
        # Выводим статистику по типам
        print(f"   📋 Типы объектов:")
        for obj_type, names in object_types.items():
            print(f"      • {obj_type}: {len(names)} шт.")
            if len(names) <= 5:
                for name in names:
                    is_ghost = name in ghost_objects
                    ghost_marker = " 👻" if is_ghost else ""
                    print(f"        - {name}{ghost_marker}")
            else:
                for name in names[:3]:
                    is_ghost = name in ghost_objects
                    ghost_marker = " 👻" if is_ghost else ""
                    print(f"        - {name}{ghost_marker}")
                print(f"        ... и еще {len(names) - 3} объектов")
        
        # Показываем последние добавленные объекты
        print(f"   🆕 Последние добавленные объекты:")
        recent_objects = list(self.objects.items())[-3:]
        for name, obj in recent_objects:
            obj_type = type(obj).__name__
            obj_id = getattr(obj, 'id', 'N/A')
            is_ghost = name in ghost_objects
            ghost_marker = " 👻" if is_ghost else ""
            print(f"      • {name} ({obj_type}, ID: {obj_id}){ghost_marker}")
        
        print()  # Пустая строка для читаемости
    
    def print_all_object_names(self) -> None:
        """
        Выводит имена ВСЕХ объектов, зарегистрированных в ZoomManager.
        Включает как обычные, так и призрачные объекты.
        """
        print(f"\n🔍 ZoomManager: Имена всех объектов")
        print(f"   📊 Всего объектов: {len(self.objects)}")
        
        if not self.objects:
            print("   📭 Нет зарегистрированных объектов")
            return
        
        # Группируем объекты по типам
        object_types = {}
        ghost_objects = {}
        
        for name, obj in self.objects.items():
            obj_type = type(obj).__name__
            if obj_type not in object_types:
                object_types[obj_type] = []
            object_types[obj_type].append(name)
            
            # Определяем, является ли объект призрачным
            is_ghost = (
                getattr(obj, 'is_ghost', False) or
                'ghost' in name.lower() or
                (hasattr(obj, 'color') and hasattr(obj.color, 'a') and getattr(obj.color, 'a', 1.0) < 0.8)
            )
            if is_ghost:
                ghost_objects[name] = obj
        
        # Выводим все объекты по типам с пометками призраков
        print(f"   📋 Все объекты по типам:")
        for obj_type, names in object_types.items():
            print(f"      • {obj_type}: {len(names)} шт.")
            for name in names:
                is_ghost = name in ghost_objects
                ghost_marker = " 👻" if is_ghost else ""
                print(f"        - {name}{ghost_marker}")
        
        # Показываем статистику по призракам
        ghost_count = len(ghost_objects)
        if ghost_count > 0:
            print(f"\n   👻 Призрачных объектов: {ghost_count}")
            print("      Призраки помечены 👻")
        else:
            print(f"\n   ✅ Призрачных объектов нет")
        
        print()  # Пустая строка для читаемости
    
    def unregister_object(self, name: str) -> None:
        """Удаляет объект из менеджера масштабирования."""
        if name in self.objects:
            del self.objects[name]

    def get_unique_spore_id(self) -> str:
        """Возвращает уникальный ID для споры."""
        self._global_spore_counter += 1
        return f"spore_{self._global_spore_counter}"

    def get_unique_link_id(self) -> str:
        """Возвращает уникальный ID для линка."""
        self._global_link_counter += 1
        return f"link_{self._global_link_counter}"

    def get_unique_object_id(self, prefix: str = "obj") -> str:
        """Возвращает уникальный ID для любого объекта."""
        self._global_object_counter += 1
        return f"{prefix}_{self._global_object_counter}"


    def identify_invariant_point(self) -> Tuple[float, float]:
        player = self.scene_setup.player
        psi = np.radians(self.scene_setup.player.rotation_y)
        phi = np.radians(self.scene_setup.player.camera_pivot.rotation_x)

        h = self.scene_setup.player.camera_pivot.world_position.y
        
        if np.tan(phi) == 0:
            # Avoid division by zero if camera is looking straight ahead
            return 0, 0

        d = (h / np.tan(phi)) 

        dx = d * np.sin(psi)
        dy = d * np.cos(psi) 

        x_0 = self.scene_setup.player.camera_pivot.world_position.x + dx
        z_0 = self.scene_setup.player.camera_pivot.world_position.z + dy

        # Обновление UI происходит автоматически через ui_manager.update_dynamic_elements()
        return x_0, z_0

    def update_transform(self) -> None:
        from src.visual.link import Link
        for obj in self.objects.values():
            try:
                # Проверяем что объект существует и имеет валидный NodePath
                if isinstance(obj, Link):
                    obj.update_geometry()
                if hasattr(obj, 'enabled') and obj.enabled and hasattr(obj, 'position'):
                    obj.apply_transform(self.a_transformation, self.b_translation, spores_scale=self.spores_scale)
                
            except (AssertionError, AttributeError, RuntimeError) as e:
                # Объект невалиден - пропускаем без краша
                continue
        # self.scene_setup.player.speed = self.scene_setup.base_speed * self.a_transformation
    
    def change_zoom(self, sign: int) -> None:
        inv = np.array(self.identify_invariant_point())
        inv_3d = np.array([inv[0], 0, inv[1]])

        zoom_multiplier = self.zoom_fact ** sign
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d
        self.update_transform()

    def reset_all(self) -> None:
        self.a_transformation = 1
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.spores_scale = 1
        self.update_transform()
        self.scene_setup.player.speed = int(self.scene_setup.base_speed)
        self.scene_setup.player.position = self.scene_setup.base_position



    def scale_spores(self, sign: int) -> None:
        self.spores_scale *= self.zoom_fact ** sign
        
        self.update_transform()

    def zoom_in(self) -> None:
        """Увеличивает масштаб (приближает)."""
        self.change_zoom(1)

    def zoom_out(self) -> None:
        """Уменьшает масштаб (отдаляет)."""
        self.change_zoom(-1)

    def reset_zoom(self) -> None:
        """Сбрасывает все трансформации к исходному состоянию."""
        self.reset_all()

    def increase_spores_scale(self) -> None:
        """Увеличивает масштаб только для спор."""
        self.scale_spores(1)

    def decrease_spores_scale(self) -> None:
        """Уменьшает масштаб только для спор."""
        self.scale_spores(-1)



