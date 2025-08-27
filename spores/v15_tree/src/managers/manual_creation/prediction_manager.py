from typing import Optional, List
import numpy as np
from ursina import destroy

from .shared_dependencies import SharedDependencies
from ...visual.prediction_visualizer import PredictionVisualizer
from ...visual.link import Link


class PredictionManager:
    """
    Управляет предсказаниями min/max управления и призрачными деревьями.

    Ответственности:
    - Создание предсказаний min/max управления (4 направления)
    - Создание призрачных деревьев для превью
    - Управление линками к предсказаниям
    - Очистка всех предсказаний
    """

    def __init__(self, deps: SharedDependencies):
        self.deps = deps

        # Предсказания min/max управления
        self.prediction_visualizers: List[PredictionVisualizer] = []
        self.prediction_links: List[Link] = []  # Линки от превью споры к призракам
        self.show_predictions = True

        # Используем кэшированные значения из SharedDependencies
        self.min_control = deps.min_control
        self.max_control = deps.max_control

        # Текущая глубина дерева для предсказаний
        self.tree_depth = 2

        print(f"   ✓ Prediction Manager создан (управление: {self.min_control} .. {self.max_control})")

    def update_predictions(self, preview_spore, preview_position_2d: np.ndarray, creation_mode: str, tree_depth: int) -> None:
        """
        Обновляет предсказания в зависимости от режима создания.

        Args:
            preview_spore: Превью спора для предсказаний
            preview_position_2d: Позиция превью в 2D
            creation_mode: 'spores' или 'tree'
            tree_depth: Глубина дерева для режима 'tree'
        """
        # print(f"🔍 PredictionManager.update_predictions вызван:")
        # print(f"   show_predictions: {self.show_predictions}")
        # print(f"   preview_spore: {preview_spore is not None}")
        # print(f"   creation_mode: {creation_mode}")
        # print(f"   tree_depth: {tree_depth}")
        
        if not self.show_predictions:
            print("   ❌ Предсказания отключены (show_predictions=False)")
            return
        
        if not preview_spore:
            print("   ❌ Нет превью споры")
            return

        self.tree_depth = tree_depth
        # print(f"   ✅ Все проверки пройдены, создаем предсказания...")

        if creation_mode == 'tree':
            # print("   🌲 Режим дерева")
            self._update_tree_preview(preview_spore, preview_position_2d)
        else:
            # print("   🧬 Режим спор")
            self._update_spore_predictions(preview_spore, preview_position_2d)
        
        # print(f"   📊 Создано visualizers: {len(self.prediction_visualizers)}")
        # print(f"   📊 Создано links: {len(self.prediction_links)}")

    def _update_spore_predictions(self, preview_spore, preview_position_2d: np.ndarray) -> None:
        """Обновляет предсказания: 2 вперед (min/max) + 2 назад (min/max)."""
        
        # print(f"🔍 _update_spore_predictions начал работу")
        
        # Очищаем старые предсказания
        self.clear_predictions()

        if not preview_spore:
            return

        try:
            # Получаем текущий dt
            dt = self._get_current_dt()

            # Конфигурации для 4 предсказаний: forward/backward + min/max
            prediction_configs = [
                {'name': 'forward_min', 'control': self.min_control, 'dt': dt, 'direction': 'forward'},
                {'name': 'forward_max', 'control': self.max_control, 'dt': dt, 'direction': 'forward'},
                {'name': 'backward_min', 'control': self.min_control, 'dt': -dt, 'direction': 'backward'},
                {'name': 'backward_max', 'control': self.max_control, 'dt': -dt, 'direction': 'backward'}
            ]

            for config in prediction_configs:
                # Создаем предсказание
                predicted_pos_2d = self.deps.pendulum.step(
                    preview_position_2d,
                    config['control'],
                    config['dt']
                )

                # Создаем визуализатор предсказания
                prediction_viz = PredictionVisualizer(
                    parent_spore=preview_spore,
                    color_manager=self.deps.color_manager,
                    zoom_manager=self.deps.zoom_manager,
                    cost_function=None,
                    config={
                        'spore': {'show_ghosts': True},
                        'angel': {'show_angels': False, 'show_pillars': False}
                    },
                    spore_id=self.deps.zoom_manager.get_unique_spore_id()
                )

                # Обновляем позицию предсказания
                prediction_viz.update(predicted_pos_2d)

                # Создаем линк от превью споры к призраку
                if prediction_viz.ghost_spore:
                    # Для обратного направления меняем местами parent и child
                    if config['direction'] == 'forward':
                        # Вперед: превью → будущее состояние
                        parent_spore = preview_spore
                        child_spore = prediction_viz.ghost_spore
                    else:  # backward
                        # Назад: прошлое состояние → превью (показываем откуда пришли)
                        parent_spore = prediction_viz.ghost_spore
                        child_spore = preview_spore

                    prediction_link = Link(
                        parent_spore=parent_spore,
                        child_spore=child_spore,
                        color_manager=self.deps.color_manager,
                        zoom_manager=self.deps.zoom_manager,
                        config=self.deps.config
                    )

                    # Устанавливаем цвет линка в зависимости от управления
                    if config['direction'] == 'forward':
                        # Обычные цвета для линков вперед
                        if 'min' in config['name']:
                            link_color_name = 'ghost_min'  # Синий для min
                        else:  # max
                            link_color_name = 'ghost_max'  # Красный для max
                    else:  # backward
                        # Те же цвета для линков назад
                        if 'min' in config['name']:
                            link_color_name = 'ghost_min'  # Синий для min
                        else:  # max
                            link_color_name = 'ghost_max'  # Красный для max

                    prediction_link.color = self.deps.color_manager.get_color('link', link_color_name)

                    # Обновляем геометрию и регистрируем в zoom manager
                    prediction_link.update_geometry()
                    link_id = self.deps.zoom_manager.get_unique_link_id()
                    self.deps.zoom_manager.register_object(prediction_link, link_id)
                    prediction_link._zoom_manager_key = link_id  # Сохраняем для удаления

                    self.prediction_links.append(prediction_link)

                self.prediction_visualizers.append(prediction_viz)

        except Exception as e:
            print(f"Ошибка обновления предсказаний: {e}")
            import traceback
            traceback.print_exc()

    def _update_tree_preview(self, preview_spore, preview_position_2d: np.ndarray) -> None:
        """Создает призрачное дерево для превью."""

        # Очищаем старые предсказания
        self.clear_predictions()

        if not preview_spore:
            return

        try:
            # Импорты для дерева
            from ...logic.tree.spore_tree import SporeTree
            from ...logic.tree.spore_tree_config import SporeTreeConfig

            # Получаем текущий dt
            dt = self._get_current_dt()

            # Создаем конфиг дерева
            tree_config = SporeTreeConfig(
                initial_position=preview_position_2d.copy(),
                dt_base=dt,
                dt_grandchildren_factor=0.2,
                show_debug=False
            )

            # Создаем логику дерева
            tree_logic = SporeTree(
                pendulum=self.deps.pendulum,
                config=tree_config,
                auto_create=False
            )

            # Создаем детей
            tree_logic.create_children()

            # Создаем внуков если нужна глубина 2
            if self.tree_depth >= 2:
                tree_logic.create_grandchildren()

            # Конвертируем в призрачные предсказания
            self._create_ghost_tree_from_logic(tree_logic, preview_spore)

        except Exception as e:
            print(f"Ошибка создания призрачного дерева: {e}")

    def _create_ghost_tree_from_logic(self, tree_logic, preview_spore):
        """Создает призрачные споры и линки из логики дерева."""

        # Создаем призрачные споры для детей
        child_ghosts = []
        for i, child_data in enumerate(tree_logic.children):
            ghost_viz = self._create_ghost_spore_from_data(child_data, f"child_{i}", 0.4)
            if ghost_viz and ghost_viz.ghost_spore:
                child_ghosts.append(ghost_viz.ghost_spore)

        # Создаем призрачные споры для внуков (если есть)
        grandchild_ghosts = []
        if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            for i, grandchild_data in enumerate(tree_logic.grandchildren):
                ghost_viz = self._create_ghost_spore_from_data(grandchild_data, f"grandchild_{i}", 0.3)
                if ghost_viz and ghost_viz.ghost_spore:
                    grandchild_ghosts.append(ghost_viz.ghost_spore)

        # Создаем призрачные линки от корня к детям с цветом по управлению
        for i, child_ghost in enumerate(child_ghosts):
            if child_ghost and i < len(tree_logic.children):
                # Получаем управление ребенка из данных дерева
                child_control = tree_logic.children[i]['control']

                # Выбираем цвет в зависимости от знака управления
                if child_control >= 0:
                    link_color = 'ghost_max'  # Положительное управление - красный
                else:
                    link_color = 'ghost_min'  # Отрицательное управление - синий

                self._create_ghost_link(
                    preview_spore,
                    child_ghost,
                    f"root_to_child_{i}",
                    link_color
                )

        # Создаем призрачные линки от детей к внукам (если tree_depth >= 2)
        if self.tree_depth >= 2 and grandchild_ghosts:
            for i, grandchild_ghost in enumerate(grandchild_ghosts):
                if grandchild_ghost:
                    # Определяем родителя внука из данных
                    grandchild_data = tree_logic.grandchildren[i]
                    parent_idx = grandchild_data['parent_idx']

                    if parent_idx < len(child_ghosts) and child_ghosts[parent_idx]:
                        self._create_ghost_link(
                            child_ghosts[parent_idx],
                            grandchild_ghost,
                            f"child_{parent_idx}_to_grandchild_{i}",
                            'ghost_min' if i % 2 == 0 else 'ghost_max'  # Чередуем цвета
                        )

    def _create_ghost_spore_from_data(self, spore_data, name_suffix, alpha):
        """Создает одну призрачную спору из данных дерева."""

        # Получаем финальную позицию споры
        final_position = spore_data['position']  # должно быть [x, z]

        # Создаем визуализатор предсказания
        prediction_viz = PredictionVisualizer(
            parent_spore=None,  # Для призраков дерева не нужен parent
            color_manager=self.deps.color_manager,
            zoom_manager=self.deps.zoom_manager,
            cost_function=None,
            config={
                'spore': {'show_ghosts': True},
                'angel': {'show_angels': False, 'show_pillars': False}
            },
            spore_id=self.deps.zoom_manager.get_unique_spore_id()
        )

        # Устанавливаем позицию призрака
        if prediction_viz.ghost_spore:
            # Устанавливаем полупрозрачность
            base_color = self.deps.color_manager.get_color('spore', 'default')
            try:
                prediction_viz.ghost_spore.color = (base_color.r, base_color.g, base_color.b, alpha)
            except:
                prediction_viz.ghost_spore.color = (0.6, 0.4, 0.9, alpha)

            # Обновляем позицию
            prediction_viz.update(final_position)

            # Призрачные споры больше не регистрируются в ZoomManager - они постоянные
            # Просто устанавливаем ID
            prediction_viz.ghost_spore.id = f"tree_ghost_{name_suffix}"

        # Добавляем в список предсказаний
        self.prediction_visualizers.append(prediction_viz)
        return prediction_viz

    def _create_ghost_link(self, parent_spore, child_spore, link_suffix, color_name):
        """Создает призрачный линк между двумя спорами."""
        try:
            ghost_link = Link(
                parent_spore=parent_spore,
                child_spore=child_spore,
                color_manager=self.deps.color_manager,
                zoom_manager=self.deps.zoom_manager,
                config=self.deps.config
            )

            # Устанавливаем цвет линка
            ghost_link.color = self.deps.color_manager.get_color('link', color_name)

            # Делаем линк полупрозрачным
            if hasattr(ghost_link, 'alpha'):
                ghost_link.alpha = 0.6

            # Обновляем геометрию и регистрируем
            ghost_link.update_geometry()
            link_id = self.deps.zoom_manager.get_unique_link_id()
            self.deps.zoom_manager.register_object(ghost_link, link_id)
            ghost_link._zoom_manager_key = link_id  # Сохраняем для удаления

            # Добавляем в список для очистки
            self.prediction_links.append(ghost_link)

        except Exception as e:
            print(f"Ошибка создания призрачного линка {link_suffix}: {e}")

    def clear_predictions(self) -> None:
        """Очищает все предсказания и их линки."""

        # Призрачные споры больше не регистрируются в ZoomManager, просто уничтожаем визуализаторы
        for viz in self.prediction_visualizers:
            viz.destroy()
        self.prediction_visualizers.clear()

        # Очищаем линки предсказаний
        for i, link in enumerate(self.prediction_links):
            # Используем сохраненный ключ если есть
            if hasattr(link, '_zoom_manager_key'):
                try:
                    self.deps.zoom_manager.unregister_object(link._zoom_manager_key)
                except:
                    pass
            else:
                # Fallback - используем сохраненный ключ линка если есть
                link_key = getattr(link, '_zoom_manager_key', None)
                if link_key:
                    try:
                        self.deps.zoom_manager.unregister_object(link_key)
                    except:
                        pass

            try:
                destroy(link)
            except:
                pass
        self.prediction_links.clear()

    def set_show_predictions(self, enabled: bool) -> None:
        """Включает/выключает показ предсказаний."""
        self.show_predictions = enabled
        if not enabled:
            self.clear_predictions()

    def _get_current_dt(self):
        """Получает текущий dt из конфига."""
        return self.deps.config.get('pendulum', {}).get('dt', 0.1)

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.clear_predictions()
        print("   ✓ Prediction Manager уничтожен")
