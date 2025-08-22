from typing import Optional, List, Tuple
import numpy as np
from ursina import scene

from ...core.spore import Spore
from ...managers.spore_manager import SporeManager
from ...managers.zoom_manager import ZoomManager
from ...managers.color_manager import ColorManager
from ...logic.pendulum import PendulumSystem
from ...visual.link import Link
from .cursor_tracker import CursorTracker
from .creation_history import CreationHistory


class SporeCreator:
    """
    Класс для создания реальных объектов (спор и деревьев).
    Отвечает за создание физических объектов в сцене.
    """

    def __init__(self,
                 cursor_tracker: CursorTracker,
                 spore_manager: SporeManager,
                 zoom_manager: ZoomManager,
                 color_manager: ColorManager,
                 pendulum: PendulumSystem,
                 config: dict):
        """
        Инициализация создателя спор.

        Args:
            cursor_tracker: Трекер курсора для получения позиции
            spore_manager: Менеджер спор для добавления объектов
            zoom_manager: Менеджер зума для регистрации объектов
            color_manager: Менеджер цветов
            pendulum: Система маятника
            config: Конфигурация
        """
        self.cursor_tracker = cursor_tracker
        self.spore_manager = spore_manager
        self.zoom_manager = zoom_manager
        self.color_manager = color_manager
        self.pendulum = pendulum
        self.config = config

        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])

        # Счетчики для уникальных ID
        self._link_counter = 0
        self._spore_counter = 0
        self._global_tree_counter = 0

        # Режим создания и параметры деревьев
        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2

        # Оптимальные dt векторы для деревьев
        self.ghost_tree_dt_vector = None
        self.ghost_tree_optimized = False

        # Список созданных линков (для отслеживания)
        self.created_links: List[Link] = []

    def create_spore_at_cursor(self, preview_spore: Optional[Spore] = None) -> Optional[List[Spore]]:
        """
        Создает споры или дерево в зависимости от режима.

        Args:
            preview_spore: Превью спора (для валидации)

        Returns:
            Список созданных спор или None при ошибке
        """
        if not preview_spore:
            return None

        if self.creation_mode == 'tree':
            return self.create_tree_at_cursor()
        else:
            return self.create_spore_family_at_cursor()

    def create_spore_family_at_cursor(self) -> Optional[List[Spore]]:
        """
        Создает полную семью спор:
        - Центральная спора в позиции курсора
        - 2 дочерние споры (forward min/max control)
        - 2 родительские споры (backward min/max control)
        - Все соединительные линки с правильными цветами

        Returns:
            Список созданных спор [center, forward_min, forward_max, backward_min, backward_max] или None при ошибке
        """
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})
            dt = pendulum_config.get('dt', 0.1)

            created_spores = []
            created_links = []

            # 1. Создаем ЦЕНТРАЛЬНУЮ спору в позиции курсора
            center_id = self._get_next_spore_id()
            center_pos_3d = self.cursor_tracker.get_current_position_3d()
            center_spore = Spore(
                pendulum=self.pendulum,
                dt=dt,
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=center_pos_3d,
                color_manager=self.color_manager,
                config=spore_config
            )

            # Добавляем центральную спору в систему
            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_id}')
            created_spores.append(center_spore)

            current_pos = self.cursor_tracker.get_current_position()
            print(f"   ✓ Создана центральная спора в позиции ({current_pos[0]:.3f}, {current_pos[1]:.3f})")

            # 2. Создаем 4 связанные споры (2 вперед + 2 назад)
            child_configs = [
                # Дочерние споры (вперед)
                {'control': self.min_control, 'name': 'forward_min', 'color': 'child_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'child_max', 'direction': 'forward'},
                # Родительские споры (назад)
                {'control': self.min_control, 'name': 'backward_min', 'color': 'parent_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'parent_max', 'direction': 'backward'}
            ]

            for i, config in enumerate(child_configs):
                # Вычисляем позицию дочерней/родительской споры
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    child_pos_2d = self.pendulum.step(
                        current_pos,
                        config['control'],
                        dt,
                        method='jit'
                    )
                else:  # backward
                    # Шаг назад во времени
                    child_pos_2d = self.pendulum.step(
                        current_pos,
                        config['control'],
                        -dt,
                        method='jit'
                    )

                # Создаем дочернюю/родительскую спору
                child_id = self._get_next_spore_id()
                child_spore = Spore(
                    pendulum=self.pendulum,
                    dt=dt,
                    goal_position=goal_position,
                    scale=spore_config.get('scale', 0.1),
                    position=(child_pos_2d[0], 0.0, child_pos_2d[1]),
                    color_manager=self.color_manager,
                    config=spore_config
                )

                # Устанавливаем цвет споры
                child_spore.color = self.color_manager.get_color('spore', config['color'])

                # Добавляем в систему
                self.spore_manager.add_spore_manual(child_spore)
                self.zoom_manager.register_object(child_spore, f'manual_{config["name"]}_{child_id}')
                created_spores.append(child_spore)

                # 3. Создаем линк между центральной и дочерней/родительской спорой
                if config['direction'] == 'forward':
                    # Вперед: центр → дочерняя
                    parent_spore = center_spore
                    child_spore_link = child_spore
                    link_color = 'child_min' if 'min' in config['name'] else 'child_max'
                else:  # backward
                    # Назад: родительская → центр
                    parent_spore = child_spore
                    child_spore_link = center_spore
                    link_color = 'parent_min' if 'min' in config['name'] else 'parent_max'

                link = Link(
                    parent_spore=parent_spore,
                    child_spore=child_spore_link,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    config=self.config
                )

                # Устанавливаем цвет линка
                link.color = self.color_manager.get_color('link', link_color)

                # Обновляем геометрию и регистрируем
                link.update_geometry()
                link_id = self._get_next_link_id()
                self.zoom_manager.register_object(link, f'manual_link_{config["name"]}_{link_id}')

                # Добавляем в списки
                created_links.append(link)
                self.created_links.append(link)

                print(f"   ✓ Создана спора {config['name']} и линк")

            print(f"   🎯 Создание завершено: {len(created_spores)} спор + {len(created_links)} линков")
            return created_spores

        except Exception as e:
            print(f"Ошибка создания семьи спор: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_tree_at_cursor(self) -> Optional[List[Spore]]:
        """
        Создает дерево в позиции курсора.

        Returns:
            Список созданных спор или None при ошибке
        """
        try:
            from ...visual.spore_tree_visual import SporeTreeVisual
            from ...logic.tree.spore_tree import SporeTree
            from ...logic.tree.spore_tree_config import SporeTreeConfig

            self._global_tree_counter += 1

            # Получаем текущий dt
            dt = self._get_current_dt()

            # Правильная позиция для дерева
            current_pos = self.cursor_tracker.get_current_position()
            tree_position = np.array([current_pos[0], current_pos[1]])

            # Валидация ghost_tree_dt_vector
            if self.ghost_tree_dt_vector is not None:
                assert len(self.ghost_tree_dt_vector) == 12, "ghost_tree_dt_vector должен быть длины 12"

            # Создаем логику дерева с правильными dt векторами
            if self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12:
                # Используем оптимизированные времена
                dt_children = np.abs(self.ghost_tree_dt_vector[0:4])
                dt_grandchildren = np.abs(self.ghost_tree_dt_vector[4:12]) if self.tree_depth == 2 else None

                print(f"   🎯 Используем оптимизированные dt")
                print(f"      📊 dt_children: {[f'{dt:.4f}' for dt in dt_children]}")
                if dt_grandchildren is not None:
                    print(f"      📊 dt_grandchildren: {[f'{dt:.4f}' for dt in dt_grandchildren]}")
            else:
                # Используем стандартные времена
                config_obj = SporeTreeConfig()
                config_obj.initial_position = tree_position
                dt_vector = config_obj.get_default_dt_vector()
                dt_children = dt_vector[0:4]
                dt_grandchildren = dt_vector[4:12] if self.tree_depth == 2 else None

                print(f"   📋 Используем стандартные dt")

            # Создаем объект конфигурации
            tree_config = SporeTreeConfig()
            tree_config.initial_position = tree_position

            # Создаем логическое дерево
            tree_logic = SporeTree(
                pendulum=self.pendulum,
                config=tree_config,
                dt_children=dt_children,
                dt_grandchildren=dt_grandchildren if self.tree_depth == 2 else None,
                show=False
            )

            # Создаем визуализацию
            tree_visual = SporeTreeVisual(
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=self.config
            )

            tree_visual.set_tree_logic(tree_logic)
            tree_visual.create_visual()

            # =============================================
            # БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ ОБЪЕКТОВ (ИСПРАВЛЕНИЕ PANDA3D)
            # =============================================

            created_spores = []
            created_links = []

            # Собираем все споры и линки
            all_spores = []
            all_links = []

            if tree_visual.root_spore:
                all_spores.append(tree_visual.root_spore)
            all_spores.extend([s for s in tree_visual.child_spores if s])
            all_spores.extend([s for s in tree_visual.grandchild_spores if s])
            all_links.extend([l for l in tree_visual.child_links if l])
            all_links.extend([l for l in tree_visual.grandchild_links if l])

            print(f"🔧 Безопасное извлечение: {len(all_spores)} спор + {len(all_links)} линков")

            # КРИТИЧЕСКИ ВАЖНО: Сначала отвязываем от tree_visual
            for spore in all_spores:
                if hasattr(spore, 'parent') and spore.parent:
                    spore.parent = scene  # Немедленно перевешиваем к корню

            for link in all_links:
                if hasattr(link, 'parent') and link.parent:
                    link.parent = scene  # Немедленно перевешиваем к корню

            # Добавляем споры в общую систему
            for i, spore in enumerate(all_spores):
                try:
                    self.spore_manager.add_spore_manual(spore)
                    created_spores.append(spore)
                    # Регистрируем с защитой от ошибок
                    reg_id = f"tree_spore_{self._global_tree_counter}_{i}"
                    self.zoom_manager.register_object(spore, reg_id)
                except Exception as e:
                    print(f"⚠️ Ошибка добавления споры {i}: {e}")

            # Добавляем линки с защитой от дублирования
            for i, link in enumerate(all_links):
                try:
                    # Проверяем, что линк еще не добавлен
                    if link not in self.created_links:
                        self.created_links.append(link)
                        created_links.append(link)
                        # Регистрируем с защитой от ошибок
                        reg_id = f"tree_link_{self._global_tree_counter}_{i}"
                        self.zoom_manager.register_object(link, reg_id)
                except Exception as e:
                    print(f"⚠️ Ошибка добавления линка {i}: {e}")

            # tree_visual больше не нужен - все объекты уже отвязаны
            tree_visual = None
            tree_logic = None

            print(f"🌲 Дерево создано в ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
            print(f"   📊 Глубина: {self.tree_depth}, dt: {dt:.4f}")
            print(f"   🎯 Добавлено в граф: {len(created_spores)} спор + {len(created_links)} линков")

            return created_spores

        except Exception as e:
            print(f"❌ Ошибка создания дерева: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_current_dt(self) -> float:
        """Получает текущий dt из конфига."""
        return self.config.get('pendulum', {}).get('dt', 0.1)

    def _get_next_link_id(self) -> int:
        """Возвращает уникальный ID для линка"""
        self._link_counter += 1
        return self._link_counter

    def _get_next_spore_id(self) -> int:
        """Возвращает уникальный ID для споры"""
        self._spore_counter += 1
        return self._spore_counter

    def set_creation_mode(self, mode: str) -> None:
        """Устанавливает режим создания."""
        self.creation_mode = mode

    def set_tree_depth(self, depth: int) -> None:
        """Устанавливает глубину дерева."""
        self.tree_depth = max(1, min(depth, 2))

    def set_ghost_tree_dt_vector(self, dt_vector: Optional[np.ndarray]) -> None:
        """Устанавливает оптимальный dt вектор для деревьев."""
        self.ghost_tree_dt_vector = dt_vector
        self.ghost_tree_optimized = dt_vector is not None

    def get_creation_mode(self) -> str:
        """Возвращает текущий режим создания."""
        return self.creation_mode

    def get_tree_depth(self) -> int:
        """Возвращает текущую глубину дерева."""
        return self.tree_depth

    def clear_created_links(self) -> None:
        """Очищает список созданных линков."""
        self.created_links.clear()

    def get_created_links_count(self) -> int:
        """Возвращает количество созданных линков."""
        return len(self.created_links)
