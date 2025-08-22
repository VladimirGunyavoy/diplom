from typing import Optional, List
import numpy as np
from ursina import Vec3, destroy

from ...core.spore import Spore
from ...managers.zoom_manager import ZoomManager
from ...managers.color_manager import ColorManager
from ...logic.pendulum import PendulumSystem
from ...visual.prediction_visualizer import PredictionVisualizer
from ...visual.link import Link
from .cursor_tracker import CursorTracker


class PreviewManager:
    """
    Класс для управления превью объектами и предсказаниями.
    Отвечает за полупрозрачные споры, призраков min/max управления и их линки.
    """

    def __init__(self,
                 cursor_tracker: CursorTracker,
                 pendulum: PendulumSystem,
                 zoom_manager: ZoomManager,
                 color_manager: ColorManager,
                 config: dict):
        """
        Инициализация менеджера превью.

        Args:
            cursor_tracker: Трекер курсора для получения позиции
            pendulum: Система маятника для вычислений
            zoom_manager: Менеджер зума для регистрации объектов
            color_manager: Менеджер цветов
            config: Конфигурация
        """
        self.cursor_tracker = cursor_tracker
        self.pendulum = pendulum
        self.zoom_manager = zoom_manager
        self.color_manager = color_manager
        self.config = config

        # Настройки превью
        self.preview_enabled = True
        self.preview_alpha = 0.5  # Полупрозрачность
        self.show_predictions = True
        self.debug_mode = False  # Флаг для отладочного вывода

        # Превью объекты
        self.preview_spore: Optional[Spore] = None
        self.prediction_visualizers: List[PredictionVisualizer] = []
        self.prediction_links: List[Link] = []

        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])

        # Режим создания (для разных типов предсказаний)
        self.creation_mode = 'spores'  # будет устанавливаться извне

    def update_preview(self) -> None:
        """
        Обновляет превью спору и предсказания.
        """
        if not self.preview_enabled:
            return

        # Создаем или обновляем превью спору
        self._update_preview_spore()

        # Обновляем предсказания
        if self.show_predictions:
            self._update_predictions()

    def set_preview_enabled(self, enabled: bool) -> None:
        """Включает/выключает превью спор."""
        self.preview_enabled = enabled
        if not enabled:
            self._destroy_preview()

    def set_creation_mode(self, mode: str) -> None:
        """Устанавливает режим создания для предсказаний."""
        self.creation_mode = mode

    def force_update_predictions(self) -> None:
        """Принудительно обновляет предсказания (например, при изменении dt)."""
        # Сбрасываем кэш позиции для принудительного обновления
        if hasattr(self, '_last_update_pos'):
            delattr(self, '_last_update_pos')
        self._update_predictions()

    def _update_preview_spore(self) -> None:
        """Создает или обновляет превью спору."""
        if not self.preview_spore:
            self._create_preview_spore()
        else:
            # Обновляем позицию существующей споры
            current_pos = self.cursor_tracker.get_current_position()
            self.preview_spore.real_position = Vec3(
                current_pos[0],
                0.0,
                current_pos[1]
            )
            # Применяем трансформации zoom manager
            self.preview_spore.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation,
                spores_scale=self.zoom_manager.spores_scale
            )

    def _create_preview_spore(self) -> None:
        """Создает новую превью спору."""
        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})

            current_pos_3d = self.cursor_tracker.get_current_position_3d()

            self.preview_spore = Spore(
                pendulum=self.pendulum,
                dt=pendulum_config.get('dt', 0.1),
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=current_pos_3d,
                color_manager=self.color_manager,
                is_ghost=True  # Делаем спору-призрак
            )

            base_color = self.color_manager.get_color('spore', 'default')

            # Способ 1: Используем атрибуты r, g, b
            try:
                self.preview_spore.color = (base_color.r, base_color.g, base_color.b, self.preview_alpha)
            except AttributeError:
                # Способ 2: Если это Vec4 или tuple
                try:
                    self.preview_spore.color = (base_color[0], base_color[1], base_color[2], self.preview_alpha)
                except (TypeError, IndexError):
                    # Способ 3: Fallback на стандартный цвет
                    self.preview_spore.color = (0.6, 0.4, 0.9, self.preview_alpha)  # Фиолетовый по умолчанию
                    print(f"   ⚠️ Использован fallback цвет для preview spore")

            # Регистрируем в zoom manager
            self.zoom_manager.register_object(self.preview_spore, name='manual_preview')

        except Exception as e:
            print(f"Ошибка создания превью споры: {e}")

    def _update_predictions(self) -> None:
        """Обновляет предсказания в зависимости от режима создания."""
        # Проверяем нужно ли принудительное обновление (при изменении dt)
        current_dt = self.config.get('pendulum', {}).get('dt', 0.1)
        force_update = False

        if not hasattr(self, '_last_dt') or self._last_dt != current_dt:
            # dt изменился - принудительное обновление
            force_update = True
            self._last_dt = current_dt

        # Защита от слишком частых обновлений при движении мыши
        if not force_update and hasattr(self, '_last_update_pos'):
            current_pos = self.cursor_tracker.get_current_position()
            # Обновляем только если курсор сдвинулся значительно (> 0.01)
            pos_diff = np.linalg.norm(current_pos - self._last_update_pos)
            if pos_diff < 0.01:
                return  # Пропускаем обновление если курсор почти не двигался

        # Сохраняем текущую позицию
        self._last_update_pos = self.cursor_tracker.get_current_position().copy()

        if self.creation_mode == 'tree':
            self._update_ghost_predictions()
        else:
            self._update_spore_predictions()

    def _update_spore_predictions(self) -> None:
        """Обновляет предсказания: 2 вперед (min/max) + 2 назад (min/max)."""
        if not self.preview_spore:
            return

        try:
            # Очищаем старые предсказания
            self._clear_predictions()

            # Получаем dt из конфига или dt_manager
            dt = self.config.get('pendulum', {}).get('dt', 0.1)

            # 4 предсказания: 2 вперед + 2 назад
            prediction_configs = [
                # Вперед
                {'control': self.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                # Назад
                {'control': self.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]

            current_pos = self.cursor_tracker.get_current_position()

            for i, config in enumerate(prediction_configs):
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    predicted_pos_2d = self.pendulum.step(
                        current_pos,
                        config['control'],
                        dt,
                        method='jit'
                    )
                else:  # backward
                    # Шаг назад во времени
                    predicted_pos_2d = self.pendulum.step(
                        current_pos,
                        config['control'],
                        -dt,
                        method='jit'
                    )

                # Создаем визуализатор предсказания
                prediction_viz = PredictionVisualizer(
                    parent_spore=self.preview_spore,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    cost_function=None,  # Не показываем cost для предсказаний
                    config={
                        'spore': {'show_ghosts': True},
                        'angel': {'show_angels': False, 'show_pillars': False}
                    },
                    spore_id=f'manual_prediction_{config["name"]}'
                )

                # Устанавливаем цвет призрака
                if prediction_viz.ghost_spore:
                    base_color = self.color_manager.get_color('spore', config['color'])

                    # Для призраков назад делаем цвет более тусклым
                    if config['direction'] == 'backward':
                        # Уменьшаем яркость для призраков назад
                        if hasattr(base_color, 'r'):
                            r, g, b = base_color.r * 0.6, base_color.g * 0.6, base_color.b * 0.6
                            prediction_viz.ghost_spore.color = (r, g, b, 0.7)
                        else:
                            # Fallback
                            prediction_viz.ghost_spore.color = (0.3, 0.3, 0.6, 0.7)  # Тусклый синий
                    else:
                        # Обычный цвет для призраков вперед
                        prediction_viz.ghost_spore.color = base_color

                # Обновляем позицию предсказания
                prediction_viz.update(predicted_pos_2d)

                # Создаем линк от превью споры к призраку
                if prediction_viz.ghost_spore:
                    # Для обратного направления меняем местами parent и child
                    if config['direction'] == 'forward':
                        # Вперед: превью → будущее состояние
                        parent_spore = self.preview_spore
                        child_spore = prediction_viz.ghost_spore
                    else:  # backward
                        # Назад: прошлое состояние → превью (показываем откуда пришли)
                        parent_spore = prediction_viz.ghost_spore
                        child_spore = self.preview_spore

                    prediction_link = Link(
                        parent_spore=parent_spore,
                        child_spore=child_spore,
                        color_manager=self.color_manager,
                        zoom_manager=self.zoom_manager,
                        config=self.config
                    )

                    # Устанавливаем цвет линка в зависимости от управления
                    if 'min' in config['name']:
                        link_color_name = 'ghost_min'  # Синий для min
                    else:  # max
                        link_color_name = 'ghost_max'  # Красный для max

                    prediction_link.color = self.color_manager.get_color('link', link_color_name)

                    # Обновляем геометрию
                    prediction_link.update_geometry()

                    # Сохраняем reg_id для правильной очистки
                    link_reg_id = f'manual_prediction_link_{config["name"]}'
                    prediction_link._reg_id = link_reg_id

                    # Регистрируем в zoom manager
                    self.zoom_manager.register_object(prediction_link, link_reg_id)

                    self.prediction_links.append(prediction_link)

                self.prediction_visualizers.append(prediction_viz)

        except Exception as e:
            print(f"Ошибка обновления предсказаний: {e}")
            import traceback
            traceback.print_exc()

    def _update_ghost_predictions(self):
        """Обновляет призрачные предсказания дерева."""
        if not self.preview_spore:
            return

        # Очищаем старые предсказания
        self._clear_predictions()

        try:
            from ...logic.tree.spore_tree import SporeTree
            from ...logic.tree.spore_tree_config import SporeTreeConfig

            # Получаем текущую позицию и dt
            current_pos = self.cursor_tracker.get_current_position()
            dt = self.config.get('pendulum', {}).get('dt', 0.1)

            # Создаем мини-дерево для предсказания
            tree_config = SporeTreeConfig()
            tree_config.initial_position = current_pos.copy()
            tree_config.dt_base = dt

            tree_logic = SporeTree(
                pendulum=self.pendulum,
                config=tree_config,
                show=False
            )

            # Создаем призрачные споры для детей
            if hasattr(tree_logic, 'children'):
                for i, child_data in enumerate(tree_logic.children):
                    child_pos = child_data['position']

                    prediction_viz = PredictionVisualizer(
                        parent_spore=self.preview_spore,
                        color_manager=self.color_manager,
                        zoom_manager=self.zoom_manager,
                        cost_function=None,
                        config={
                            'spore': {'show_ghosts': True},
                            'angel': {'show_angels': False, 'show_pillars': False}
                        },
                        spore_id=f'tree_child_{i}'
                    )

                    # Устанавливаем цвет и позицию
                    if prediction_viz.ghost_spore:
                        color = self.color_manager.get_color('spore', 'ghost_min' if i % 2 == 0 else 'ghost_max')
                        prediction_viz.ghost_spore.color = color
                        prediction_viz.update(child_pos)

                    self.prediction_visualizers.append(prediction_viz)

            # Аналогично для внуков, если есть
            if hasattr(tree_logic, 'grandchildren'):
                for i, gc_data in enumerate(tree_logic.grandchildren):
                    gc_pos = gc_data['position']

                    prediction_viz = PredictionVisualizer(
                        parent_spore=self.preview_spore,
                        color_manager=self.color_manager,
                        zoom_manager=self.zoom_manager,
                        cost_function=None,
                        config={
                            'spore': {'show_ghosts': True},
                            'angel': {'show_angels': False, 'show_pillars': False}
                        },
                        spore_id=f'tree_grandchild_{i}'
                    )

                    # Устанавливаем цвет и позицию
                    if prediction_viz.ghost_spore:
                        color = self.color_manager.get_color('spore', 'ghost_min' if i % 2 == 0 else 'ghost_max')
                        prediction_viz.ghost_spore.color = (color[0]*0.7, color[1]*0.7, color[2]*0.7, 0.5)  # Более тусклые
                        prediction_viz.update(gc_pos)

                    self.prediction_visualizers.append(prediction_viz)

        except Exception as e:
            print(f"Ошибка создания призрачного дерева: {e}")

    def _clear_predictions(self) -> None:
        """Очищает все предсказания и их линки."""
        # Очищаем визуализаторы предсказаний
        for viz in self.prediction_visualizers:
            if getattr(viz, 'ghost_spore', None):
                self._safe_unregister_and_destroy(viz.ghost_spore)
            try:
                viz.destroy()
            except:
                pass
        self.prediction_visualizers.clear()

        # Очищаем линки предсказаний с принудительной отменой регистрации
        for link in self.prediction_links:
            # Принудительно отменяем регистрацию по известному ID
            if hasattr(link, '_reg_id'):
                try:
                    self.zoom_manager.unregister_object(link._reg_id)
                except:
                    pass

            # Уничтожаем объект
            try:
                destroy(link)
            except:
                pass
        self.prediction_links.clear()

    def _destroy_preview(self) -> None:
        """Уничтожает все превью объекты."""
        # Очищаем предсказания
        self._clear_predictions()

        # Удаляем превью спору
        if self.preview_spore:
            self._safe_unregister_and_destroy(self.preview_spore)
            self.preview_spore = None

    def _safe_unregister_and_destroy(self, obj) -> None:
        """Безопасно удаляет объект из zoom manager и уничтожает его."""
        try:
            # Пытаемся разрегистрировать через zoom manager
            if hasattr(obj, '_reg_id'):
                self.zoom_manager.unregister_object(obj._reg_id)
            elif hasattr(obj, 'id'):
                self.zoom_manager.unregister_object(obj.id)
        except:
            pass

        try:
            # Пытаемся уничтожить объект
            if hasattr(obj, 'destroy'):
                obj.destroy()
            else:
                destroy(obj)
        except:
            pass

    def clear_all(self) -> None:
        """Очищает все превью объекты."""
        self._destroy_preview()

    def get_preview_spore(self) -> Optional[Spore]:
        """Возвращает текущую превью спору."""
        return self.preview_spore

    def add_prediction_visualizer(self, visualizer) -> None:
        """Добавляет визуализатор предсказания в список."""
        self.prediction_visualizers.append(visualizer)

    def add_prediction_link(self, link) -> None:
        """Добавляет линк предсказания в список."""
        self.prediction_links.append(link)