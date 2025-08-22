from typing import Optional, List
import numpy as np
from ursina import Vec3, destroy

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..logic.pendulum import PendulumSystem
from ..visual.link import Link
from ..visual.spore_tree_visual import SporeTreeVisual
from ..logic.tree.spore_tree import SporeTree
from ..logic.tree.spore_tree_config import SporeTreeConfig
from .cursor_tracker import CursorTracker
from .creation_history import CreationHistory
from .tree_optimizer import TreeOptimizer


class SporeCreator:
    """
    Класс для создания спор и деревьев в позиции курсора.
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
            spore_manager: Менеджер спор
            zoom_manager: Менеджер зума
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

        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2
        self.created_links: List[Link] = []
        self._spore_counter = 0
        self._link_counter = 0
        self._global_tree_counter = 0

        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])

    def set_creation_mode(self, mode: str) -> None:
        """Устанавливает режим создания."""
        if mode in ['spores', 'tree']:
            self.creation_mode = mode
        else:
            print(f"⚠️ Некорректный режим создания: {mode}. Оставлен текущий: {self.creation_mode}")

    def get_creation_mode(self) -> str:
        """Возвращает текущий режим создания."""
        return self.creation_mode

    def set_tree_depth(self, depth: int) -> None:
        """Устанавливает глубину дерева."""
        self.tree_depth = max(1, min(depth, 2))

    def get_tree_depth(self) -> int:
        """Возвращает текущую глубину дерева."""
        return self.tree_depth

    def create_spore_family_at_cursor(self) -> Optional[List[Spore]]:
        """
        Создает семью спор: центральная + 2 дочерние (forward) + 2 родительские (backward).
        """
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})
            dt = pendulum_config.get('dt', 0.1)
            current_pos = self.cursor_tracker.get_current_position()

            created_spores = []
            created_links = []

            # Создаем центральную спору
            center_id = self._get_next_spore_id()
            center_spore = Spore(
                pendulum=self.pendulum,
                dt=dt,
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(current_pos[0], 0.0, current_pos[1]),
                color_manager=self.color_manager,
                config=spore_config
            )

            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_id}')
            created_spores.append(center_spore)
            print(f"   ✓ Создана центральная спора в позиции ({current_pos[0]:.3f}, {current_pos[1]:.3f})")

            # Создаем дочерние и родительские споры
            spore_configs = [
                {'control': self.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                {'control': self.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]

            for config in spore_configs:
                child_id = self._get_next_spore_id()
                child_pos_2d = self.pendulum.step(
                    current_pos,
                    config['control'],
                    dt if config['direction'] == 'forward' else -dt,
                    method='jit'
                )

                child_spore = Spore(
                    pendulum=self.pendulum,
                    dt=dt,
                    goal_position=goal_position,
                    scale=spore_config.get('scale', 0.1),
                    position=(child_pos_2d[0], 0.0, child_pos_2d[1]),
                    color_manager=self.color_manager,
                    config=spore_config
                )

                self.spore_manager.add_spore_manual(child_spore)
                self.zoom_manager.register_object(child_spore, f'manual_{config["name"]}_{child_id}')
                child_spore.logic.optimal_control = np.array([config['control']])
                created_spores.append(child_spore)
                print(f"   ✓ Создана спора {config['name']} в позиции ({child_pos_2d[0]:.3f}, {child_pos_2d[1]:.3f})")

                # Создаем линк
                link_id = self._get_next_link_id()
                parent_spore = center_spore if config['direction'] == 'forward' else child_spore
                child_link_spore = child_spore if config['direction'] == 'forward' else center_spore

                spore_link = Link(
                    parent_spore=parent_spore,
                    child_spore=child_link_spore,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    config=self.config
                )

                link_color_name = 'ghost_min' if 'min' in config['name'] else 'ghost_max'
                spore_link.color = self.color_manager.get_color('link', link_color_name)
                spore_link.update_geometry()
                self.zoom_manager.register_object(spore_link, f'manual_link_{config["name"]}_{link_id}')
                created_links.append(spore_link)
                self.created_links.append(spore_link)
                print(f"   ✓ Создан {link_color_name} линк для {config['name']} (направление: {config['direction']})")

            print(f"   🎯 Создано ВСЕГО: {len(created_spores)} спор + {len(created_links)} линков")
            return created_spores, created_links

        except Exception as e:
            print(f"❌ Ошибка создания семьи спор: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_tree_at_cursor(self, tree_optimizer: TreeOptimizer) -> Optional[List[Spore]]:
        """
        Создает дерево в позиции курсора, используя оптимизированные dt из TreeOptimizer.
        """
        try:
            current_pos = self.cursor_tracker.get_current_position()
            dt = self.config.get('pendulum', {}).get('dt', 0.1)
            self._global_tree_counter += 1

            # Проверяем изменился ли dt
            dt_changed = tree_optimizer.check_dt_changed()
            use_optimized_dt = tree_optimizer.should_use_optimized_dt(dt_changed)

            # Получаем оптимизированные dt, если есть
            dt_children, dt_grandchildren = tree_optimizer.get_optimized_dt_vectors()

            # Создаем конфиг дерева
            tree_config = SporeTreeConfig(
                initial_position=current_pos.copy(),
                dt_base=dt,
                dt_grandchildren_factor=0.05,
                show_debug=False
            )

            if use_optimized_dt and dt_children is not None and dt_grandchildren is not None:
                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children,
                    dt_grandchildren=dt_grandchildren,
                    auto_create=True,
                    show=False
                )
            else:
                dt_children_std = np.array([dt, -dt, dt, -dt], dtype=float)
                dt_grandchildren_std = np.array([dt * 0.2, -dt * 0.2] * 4, dtype=float)
                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children_std,
                    dt_grandchildren=dt_grandchildren_std,
                    auto_create=True,
                    show=False
                )

            # Сохраняем dt вектор, если он еще не сохранен
            tree_optimizer.save_dt_vector_from_tree(tree_logic, dt)

            # Создаем визуализацию
            tree_visual = SporeTreeVisual(
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=self.config
            )
            tree_visual.set_tree_logic(tree_logic)
            tree_visual.create_visual()

            # Собираем споры и линки
            created_spores = [tree_visual.root_spore] + tree_visual.child_spores + tree_visual.grandchild_spores
            created_links = tree_visual.child_links + tree_visual.grandchild_links

            # Добавляем в общий граф
            for i, spore in enumerate(created_spores):
                if spore:
                    self.spore_manager.add_spore_manual(spore)
                    self.zoom_manager.register_object(spore, f"tree_spore_{self._global_tree_counter}_{i}")

            for i, link in enumerate(created_links):
                if link:
                    self.created_links.append(link)
                    self.zoom_manager.register_object(link, f"tree_link_{self._global_tree_counter}_{i}")

            # Очищаем визуализацию
            tree_visual.root_spore = None
            tree_visual.child_spores.clear()
            tree_visual.grandchild_spores.clear()
            tree_visual.child_links.clear()
            tree_visual.grandchild_links.clear()
            tree_visual.visual_created = False

            print(f"🌲 Дерево создано в ({current_pos[0]:.3f}, {current_pos[1]:.3f})")
            print(f"   📊 Глубина: {self.tree_depth}, dt: {dt:.4f}")
            return created_spores, created_links

        except Exception as e:
            print(f"❌ Ошибка создания дерева: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_next_spore_id(self) -> int:
        self._spore_counter += 1
        return self._spore_counter

    def _get_next_link_id(self) -> int:
        self._link_counter += 1
        return self._link_counter