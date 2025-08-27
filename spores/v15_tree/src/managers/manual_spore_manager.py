from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..managers.manual_creation.preview_manager import PreviewManager
from ..managers.manual_creation.prediction_manager import PredictionManager
from ..managers.manual_creation.tree_creation_manager import TreeCreationManager
from ..logic.pendulum import PendulumSystem
from ..visual.link import Link


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

        # Создаем PreviewManager
        self.preview_manager = PreviewManager(
            zoom_manager=zoom_manager,
            pendulum=pendulum,
            color_manager=color_manager,
            config=config
        )

        # Создаем PredictionManager
        self.prediction_manager = PredictionManager(
            zoom_manager=zoom_manager,
            pendulum=pendulum,
            color_manager=color_manager,
            config=config
        )

        # Создаем TreeCreationManager
        self.tree_creation_manager = TreeCreationManager(
            spore_manager=spore_manager,
            zoom_manager=zoom_manager,
            pendulum=pendulum,
            color_manager=color_manager,
            config=config
        )
        
        # Синхронизируем ghost_tree_dt_vector
        # self.tree_creation_manager.ghost_tree_dt_vector = self._ghost_tree_dt_vector

        # Общие созданные линки
        self.created_links: List[Link] = []
        
        # Получаем id_manager от spore_manager
        self.id_manager = self.spore_manager.id_manager

        # Сохранение dt вектора от призрачного дерева
        self._ghost_tree_dt_vector = None

        # История созданных групп спор для возможности удаления
        self.spore_groups_history: List[List[Spore]] = []  # История групп спор
        self.group_links_history: List[List[Link]] = []    # История линков для каждой группы

        # Синхронизируем режимы с TreeCreationManager
        self.creation_mode = self.tree_creation_manager.creation_mode
        self.tree_depth = self.tree_creation_manager.tree_depth

        print(f"   ✓ Manual Spore Manager создан (управление: {self.prediction_manager.min_control} .. {self.prediction_manager.max_control})")
        print(f"   📚 История групп инициализирована")

    def toggle_creation_mode(self):
        """Переключает режим создания."""
        self.tree_creation_manager.toggle_creation_mode()
        # Синхронизируем режим
        self.creation_mode = self.tree_creation_manager.creation_mode

    def set_tree_depth(self, depth: int):
        """Устанавливает глубину дерева."""
        self.tree_creation_manager.set_tree_depth(depth)
        # Синхронизируем глубину
        self.tree_depth = self.tree_creation_manager.tree_depth

    def create_tree_at_cursor(self):
        """Создает дерево в позиции курсора."""
        if not self.preview_manager.preview_enabled or not self.preview_manager.has_preview():
            return None

        # Делегируем создание дерева в TreeCreationManager
        preview_position_2d = self.preview_manager.get_preview_position()
        created_spores = self.tree_creation_manager.create_tree_at_cursor(preview_position_2d)
        
        if created_spores:
            # Создаем линки если нужно (TreeCreationManager уже создал основные)
            created_links = []  # TreeCreationManager уже обработал линки
            

            
            # Сохраняем в историю групп
            self.spore_groups_history.append(created_spores.copy())
            self.group_links_history.append(created_links.copy())
            
            print(f"   📚 Группа #{len(self.spore_groups_history)} сохранена в истории")
            
            return created_spores
        
        return None

    @property
    def ghost_tree_dt_vector(self):
        """Получает dt вектор от призрачного дерева."""
        return self._ghost_tree_dt_vector

    @ghost_tree_dt_vector.setter
    def ghost_tree_dt_vector(self, value):
        """Устанавливает dt вектор и синхронизирует с TreeCreationManager."""
        self._ghost_tree_dt_vector = value
        if hasattr(self, 'tree_creation_manager'):
            self.tree_creation_manager.ghost_tree_dt_vector = value
            




    def _get_next_link_id(self) -> int:
        """Возвращает уникальный ID для линка"""
        return self.id_manager.get_next_link_id()

    def _get_next_spore_id(self) -> int:
        """Возвращает уникальный ID для споры"""
        return self.id_manager.get_next_spore_id()

    def get_mouse_world_position(self) -> Optional[Tuple[float, float]]:
        """
        Получает позицию курсора мыши с правильной трансформацией зума.

        Алгоритм:
        1. Определить точку на которую смотрит камера (без зума)
        2. Отнять позицию frame origin_cube
        3. Поделить на common_scale

        Returns:
            (x, z) координаты курсора в правильной системе координат
        """
        try:
            # 1. Получаем точку взгляда камеры (без зума)
            look_point_x, look_point_z = self.zoom_manager.identify_invariant_point()

            # 2. Получаем трансформированную позицию origin_cube из frame
            frame = getattr(self.zoom_manager.scene_setup, 'frame', None)
            if frame and hasattr(frame, 'origin_cube'):
                # Используем обычную position (после трансформаций) а не real_position
                origin_pos = frame.origin_cube.position
                if hasattr(origin_pos, 'x'):
                    origin_x, origin_z = origin_pos.x, origin_pos.z
                else:
                    origin_x, origin_z = origin_pos[0], origin_pos[2]
            else:
                # Fallback если frame не найден
                origin_x, origin_z = 0.0, 0.0

            # 3. Получаем масштаб трансформации
            transform_scale = getattr(self.zoom_manager, 'a_transformation', 1.0)

            # 4. ПРАВИЛЬНАЯ ФОРМУЛА: (look_point - frame_origin_cube) / scale
            corrected_x = (look_point_x - origin_x) / transform_scale
            corrected_z = (look_point_z - origin_z) / transform_scale

            return (corrected_x, corrected_z)

        except Exception as e:
            print(f"Ошибка определения позиции мыши: {e}")
            return (0.0, 0.0)

    def update_cursor_position(self) -> None:
        """
        Обновляет позицию превью споры по позиции курсора мыши.
        """
            # print(f"DEBUG: update_cursor_position вызван")
            # print(f"   preview_enabled: {self.preview_manager.preview_enabled}")
        
        if not self.preview_manager.preview_enabled:
            print("   STOP: preview отключен")
            return

        # Получаем правильную позицию курсора мыши
        mouse_pos = self.get_mouse_world_position()
        # print(f"   mouse_pos: {mouse_pos}")
        
        if mouse_pos is None:
            print("   STOP: mouse_pos = None")
            return

        # Используем PreviewManager для обновления позиции
        # print("   Вызываем preview_manager.update_cursor_position...")
        self.preview_manager.update_cursor_position(mouse_pos)

        # Обновляем предсказания
        # print(f"   show_predictions: {self.prediction_manager.show_predictions}")
        if self.prediction_manager.show_predictions:
            # print("   Вызываем _update_predictions...")
            self._update_predictions()
        else:
            print("   SKIP: предсказания отключены")

    def _update_predictions(self) -> None:
        """Обновляет предсказания в зависимости от режима создания."""
        self.prediction_manager.update_predictions(
            preview_spore=self.preview_manager.get_preview_spore(),
            preview_position_2d=self.preview_manager.get_preview_position(),
            creation_mode=self.tree_creation_manager.creation_mode,
            tree_depth=self.tree_creation_manager.tree_depth
        )

    def create_spore_at_cursor(self):
        """Создает споры или дерево в зависимости от режима."""
        if self.tree_creation_manager.creation_mode == 'tree':
            return self.create_tree_at_cursor()
        else:
            return self._create_spore_at_cursor_original()

    def _create_spore_at_cursor_original(self) -> Optional[List[Spore]]:
        """
        Создает полную семью спор:
        - Центральная спора в позиции курсора
        - 2 дочерние споры (forward min/max control)
        - 2 родительские споры (backward min/max control)
        - Все соединительные линки с правильными цветами

        Returns:
            Список созданных спор [center, forward_min, forward_max, backward_min, backward_max] или None при ошибке
        """
        if not self.preview_manager.preview_enabled or not self.preview_manager.has_preview():
            return None

        try:
            goal_position = self.config.get('spore', {}).get('goal_position', [0, 0])
            spore_config = self.config.get('spore', {})
            pendulum_config = self.config.get('pendulum', {})
            dt = pendulum_config.get('dt', 0.1)

            created_spores = []
            created_links = []

            # 1. Создаем ЦЕНТРАЛЬНУЮ спору в позиции курсора
            center_spore = Spore(
                pendulum=self.pendulum,
                dt=dt,
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_manager.get_preview_position()[0], 0.0, self.preview_manager.get_preview_position()[1]),
                color_manager=self.color_manager,
                config=spore_config
            )
            # Присваиваем уникальный ID
            center_spore.id = self._get_next_spore_id()

            # Добавляем центральную спору в систему
            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_spore.id}')
            created_spores.append(center_spore)
            print(f"   ✓ Создана центральная спора в позиции ({self.preview_manager.get_preview_position()[0]:.3f}, {self.preview_manager.get_preview_position()[1]:.3f})")

            # 2. Создаем ДОЧЕРНИЕ споры (forward) + РОДИТЕЛЬСКИЕ споры (backward)
            spore_configs = [
                # Дочерние (forward)
                {'control': self.prediction_manager.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.prediction_manager.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                # Родительские (backward)
                {'control': self.prediction_manager.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.prediction_manager.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]

            for config in spore_configs:
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    child_pos_2d = self.pendulum.step(
                        self.preview_manager.get_preview_position(),
                        config['control'],
                        dt,
                        method='jit'
                    )
                else:  # backward
                    # Шаг назад во времени
                    child_pos_2d = self.pendulum.step(
                        self.preview_manager.get_preview_position(),
                        config['control'],
                        -dt,
                        method='jit'
                    )

                # Создаем спору
                child_spore = Spore(
                    pendulum=self.pendulum,
                    dt=dt,
                    goal_position=goal_position,
                    scale=spore_config.get('scale', 0.1),
                    position=(child_pos_2d[0], 0.0, child_pos_2d[1]),
                    color_manager=self.color_manager,
                    config=spore_config
                )
                # Присваиваем уникальный ID
                child_spore.id = self._get_next_spore_id()

                # Добавляем спору в систему БЕЗ автоматических призраков
                self.spore_manager.add_spore_manual(child_spore)
                self.zoom_manager.register_object(child_spore, f'manual_{config["name"]}_{child_spore.id}')

                # Переопределяем управление на конкретное значение
                child_spore.logic.optimal_control = np.array([config['control']])

                created_spores.append(child_spore)
                print(f"   ✓ Создана спора {config['name']} в позиции ({child_pos_2d[0]:.3f}, {child_pos_2d[1]:.3f}) с управлением {config['control']:.2f}")

                # 3. Создаем ЛИНК с правильным направлением и цветом
                # Определяем направление стрелки
                if config['direction'] == 'forward':
                    # Вперед: центр → дочерняя
                    parent_spore = center_spore
                    child_link_spore = child_spore
                else:  # backward
                    # Назад: родительская → центр (показываем откуда пришли)
                    parent_spore = child_spore
                    child_link_spore = center_spore

                # Создаем линк
                spore_link = Link(
                    parent_spore=parent_spore,
                    child_spore=child_link_spore,
                    color_manager=self.color_manager,
                    zoom_manager=self.zoom_manager,
                    config=self.config
                )
                # Присваиваем уникальный ID
                spore_link.id = self._get_next_link_id()

                # Устанавливаем цвет линка в зависимости от управления
                if 'min' in config['name']:
                    # Синий цвет для минимального управления
                    link_color_name = 'ghost_min'
                else:  # max
                    # Красный цвет для максимального управления
                    link_color_name = 'ghost_max'

                spore_link.color = self.color_manager.get_color('link', link_color_name)

                # Обновляем геометрию и регистрируем линк
                spore_link.update_geometry()
                self.zoom_manager.register_object(spore_link, f'manual_link_{config["name"]}_{spore_link.id}')

                created_links.append(spore_link)
                self.created_links.append(spore_link)

                print(f"   ✓ Создан {link_color_name} линк для {config['name']} (направление: {config['direction']})")

            print(f"   🎯 Создано ВСЕГО: {len(created_spores)} спор + {len(created_links)} линков")
            print(f"   📊 Состав: 1 центральная + 2 дочерние (forward) + 2 родительские (backward)")

            # 🆕 СОХРАНЕНИЕ В ИСТОРИЮ (как обычную группу спор)
            self.spore_groups_history.append(created_spores.copy())
            self.group_links_history.append(created_links.copy())
            print(f"   📚 Группа #{len(self.spore_groups_history)} сохранена в истории")

            return created_spores

        except Exception as e:
            print(f"Ошибка создания семьи спор: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _destroy_preview(self) -> None:
        """Уничтожает предсказания и их линки."""
        self.prediction_manager.clear_predictions()

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.preview_manager.destroy()
        self._destroy_preview()
        print("   ✓ Manual Spore Manager уничтожен")

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
        cleared_groups = len(self.spore_groups_history)
        self.spore_groups_history.clear()
        self.group_links_history.clear()
        print(f"   📚 Очищена история: {cleared_groups} групп")

        self.created_links.clear()

    def delete_last_spore_group(self) -> bool:
        """
        Удаляет последнюю созданную группу спор и их линки.

        Returns:
            True если удаление успешно, False если нечего удалять
        """
        if not self.spore_groups_history:
            print("   ⚠️ Нет групп для удаления")
            return False

        try:
            # 1. Получаем последнюю группу из истории
            last_spores = self.spore_groups_history.pop()
            last_links = self.group_links_history.pop()

            print(f"   🗑️ Удаление группы #{len(self.spore_groups_history) + 1}")
            print(f"   📊 К удалению: {len(last_spores)} спор + {len(last_links)} линков")

            # 2. УДАЛЯЕМ ЛИНКИ (важно делать ДО удаления спор)
            deleted_links = 0
            for i, link in enumerate(last_links):
                try:
                    # Дерегистрируем из zoom_manager используя сохраненный ключ
                    if hasattr(link, '_zoom_manager_key'):
                        key = link._zoom_manager_key
                        self.zoom_manager.unregister_object(key)
                        print(f"   ✓ Линк {i+1} дерегистрирован: {key}")
                    else:
                        # Fallback: поиск по ссылке
                        if hasattr(self.zoom_manager, 'objects'):
                            for key, obj in list(self.zoom_manager.objects.items()):
                                if obj is link:
                                    self.zoom_manager.unregister_object(key)
                                    print(f"   ✓ Линк {i+1} дерегистрирован (fallback): {key}")
                                    break

                    # Удаляем из created_links
                    if link in self.created_links:
                        self.created_links.remove(link)

                    # Уничтожаем объект Ursina
                    destroy(link)
                    deleted_links += 1
                    print(f"   ✅ Линк {i+1} удален")

                except Exception as e:
                    print(f"   ❌ Ошибка удаления линка {i+1}: {e}")

            # 3. УДАЛЯЕМ СПОРЫ
            deleted_spores = 0
            for i, spore in enumerate(last_spores):
                try:
                    # Удаляем из SporeManager
                    if hasattr(self.spore_manager, 'remove_spore'):
                        removed = self.spore_manager.remove_spore(spore)
                        if removed:
                            print(f"   ✓ Спора {i+1} удалена из SporeManager")
                    else:
                        # Fallback если remove_spore еще не реализован
                        if hasattr(self.spore_manager, 'objects') and spore in self.spore_manager.objects:
                            self.spore_manager.objects.remove(spore)
                            print(f"   ✓ Спора {i+1} удалена из objects (fallback)")

                    # Дерегистрируем из zoom_manager используя сохраненный ключ
                    if hasattr(spore, '_zoom_manager_key'):
                        key = spore._zoom_manager_key
                        self.zoom_manager.unregister_object(key)
                        print(f"   ✓ Спора {i+1} дерегистрирована: {key}")
                    else:
                        # Fallback: поиск по ссылке
                        if hasattr(self.zoom_manager, 'objects'):
                            for key, obj in list(self.zoom_manager.objects.items()):
                                if obj is spore:
                                    self.zoom_manager.unregister_object(key)
                                    print(f"   ✓ Спора {i+1} дерегистрирована (fallback): {key}")
                                    break

                    # Уничтожаем объект Ursina
                    destroy(spore)
                    deleted_spores += 1
                    print(f"   ✅ Спора {i+1} уничтожена")

                except Exception as e:
                    print(f"   ❌ Ошибка удаления споры {i+1}: {e}")

            # 4. ИТОГОВАЯ СТАТИСТИКА
            print(f"   🎯 УДАЛЕНИЕ ЗАВЕРШЕНО:")
            print(f"      📊 Спор удалено: {deleted_spores}/{len(last_spores)}")
            print(f"      🔗 Линков удалено: {deleted_links}/{len(last_links)}")
            print(f"      📚 Групп осталось в истории: {len(self.spore_groups_history)}")

            # Проверяем что удаление было успешным
            if deleted_spores == len(last_spores) and deleted_links == len(last_links):
                print(f"   ✅ Группа успешно удалена!")
                return True
            else:
                print(f"   ⚠️ Удаление частично неуспешно")
                return False

        except Exception as e:
            print(f"   ❌ Критическая ошибка удаления группы: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_groups_history_stats(self) -> dict:
        """
        Возвращает статистику по истории созданных групп.

        Returns:
            Словарь с информацией о количестве групп, спор и линков в истории
        """
        total_groups = len(self.spore_groups_history)
        total_spores = sum(len(group) for group in self.spore_groups_history)
        total_links = sum(len(links) for links in self.group_links_history)

        return {
            'total_groups': total_groups,
            'total_spores': total_spores,
            'total_links': total_links,
            'can_delete': total_groups > 0
        }

    def print_groups_history_stats(self) -> None:
        """Выводит детальную статистику истории групп."""
        stats = self.get_groups_history_stats()

        print(f"\n📚 СТАТИСТИКА ИСТОРИИ ГРУПП:")
        print(f"   🔢 Всего групп: {stats['total_groups']}")
        print(f"   🧬 Всего спор: {stats['total_spores']}")
        print(f"   🔗 Всего линков: {stats['total_links']}")
        print(f"   🗑️ Можно удалить: {'Да' if stats['can_delete'] else 'Нет'}")

        if stats['total_groups'] > 0:
            print(f"   📋 Последняя группа: {len(self.spore_groups_history[-1])} спор + {len(self.group_links_history[-1])} линков")
        print("========================")

    def _get_current_dt(self) -> float:
        """Получает текущий dt из конфигурации."""
        return self.config.get('pendulum', {}).get('dt', 0.1)