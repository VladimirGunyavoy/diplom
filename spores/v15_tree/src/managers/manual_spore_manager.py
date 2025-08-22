from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..managers.manual_creation.preview_manager import PreviewManager
from ..managers.manual_creation.prediction_manager import PredictionManager
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

        # Общие созданные линки
        self.created_links: List[Link] = []

        self._link_counter = 0
        self._spore_counter = 0

        # Сохранение dt вектора от призрачного дерева
        self.ghost_tree_dt_vector = None

        # История созданных групп спор для возможности удаления
        self.spore_groups_history: List[List[Spore]] = []  # История групп спор
        self.group_links_history: List[List[Link]] = []    # История линков для каждой группы

        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2
        self._global_tree_counter = 0

        print(f"   ✓ Manual Spore Manager создан (управление: {self.prediction_manager.min_control} .. {self.prediction_manager.max_control})")
        print(f"   📚 История групп инициализирована")

    def toggle_creation_mode(self):
        """Переключает режим создания."""
        if self.creation_mode == 'spores':
            self.creation_mode = 'tree'
            print("🌲 Режим: Создание деревьев")
        else:
            self.creation_mode = 'spores'
            print("🌟 Режим: Создание спор")

    def set_tree_depth(self, depth: int):
        """Устанавливает глубину дерева."""
        self.tree_depth = max(1, min(depth, 2))
        print(f"🌲 Глубина дерева: {self.tree_depth}")

    def create_tree_at_cursor(self):
        """
        Создает дерево в позиции курсора.

        ЛОГИКА: дерево → споры и линки → добавляем в общий граф → забываем дерево
        """
        if not self.preview_manager.preview_enabled or not self.preview_manager.has_preview():
            return None

        try:
            from ..visual.spore_tree_visual import SporeTreeVisual
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig

            self._global_tree_counter += 1

            # Получаем текущий dt
            dt = self._get_current_dt()

            # ИСПРАВЛЕНИЕ: Правильная позиция для дерева
            tree_position = np.array([self.preview_manager.get_preview_position()[0], self.preview_manager.get_preview_position()[1]])

            # Создаем логику дерева с правильными dt векторами
            if self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12:
                # ИСПРАВЛЕНИЕ: Берем абсолютные значения, т.к. create_children() применит знаки сам
                dt_children = np.abs(self.ghost_tree_dt_vector[:4])
                dt_grandchildren = np.abs(self.ghost_tree_dt_vector[4:12])

                print(f"🎯 Используем dt вектор от призрачного дерева:")
                print(f"   Исходный dt_children: {self.ghost_tree_dt_vector[:4]}")
                print(f"   Абсолютный dt_children: {dt_children}")
                print(f"   Исходный dt_grandchildren: {self.ghost_tree_dt_vector[4:12]}")
                print(f"   Абсолютный dt_grandchildren: {dt_grandchildren}")

                # Создаем конфиг дерева
                tree_config = SporeTreeConfig(
                    initial_position=tree_position,  # ИСПРАВЛЕНО
                    dt_base=dt,
                    dt_grandchildren_factor=0.05,
                    show_debug=False
                )

                # ИСПРАВЛЕНИЕ: Всегда используем auto_create=False
                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=None,  # Не передаем в конструктор
                    dt_grandchildren=None,  # Не передаем в конструктор
                    auto_create=False,  # ИСПРАВЛЕНО: Всегда False
                    show=False
                )

                # ПРИНУДИТЕЛЬНО создаем все элементы с правильными dt
                tree_logic.create_children(dt_children=dt_children, show=True)
                if self.tree_depth >= 2:
                    tree_logic.create_grandchildren(dt_grandchildren=dt_grandchildren, show=True)

            else:
                # Fallback: создаем дерево стандартным способом
                tree_config = SporeTreeConfig(
                    initial_position=tree_position,  # ИСПРАВЛЕНО
                    dt_base=dt,
                    dt_grandchildren_factor=0.05,
                    show_debug=False
                )

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    auto_create=False
                )

                # Создаем детей и внуков стандартно
                tree_logic.create_children(show=True)
                if self.tree_depth >= 2:
                    tree_logic.create_grandchildren(show=True)

            # ДОБАВЛЯЕМ ПРИНУДИТЕЛЬНУЮ ПРОВЕРКУ
            print(f"🔍 ПРОВЕРКА ДЕРЕВА:")
            print(f"   Ожидаем: 4 ребенка, получили: {len(getattr(tree_logic, 'children', []))}")
            print(f"   Ожидаем: 8 внуков, получили: {len(getattr(tree_logic, 'grandchildren', []))}")

            # Если детей меньше 4, пересоздаем
            if not hasattr(tree_logic, 'children') or len(tree_logic.children) != 4:
                print(f"⚠️ Неполное количество детей, принудительно пересоздаем...")
                tree_logic.create_children(dt_children=dt_children if 'dt_children' in locals() else None, show=True)
                print(f"   После пересоздания детей: {len(tree_logic.children)}")

            if self.tree_depth >= 2:
                # Если внуков меньше 8, пересоздаем
                if not hasattr(tree_logic, 'grandchildren') or len(tree_logic.grandchildren) != 8:
                    print(f"⚠️ Неполное количество внуков, принудительно пересоздаем...")
                    tree_logic.create_grandchildren(dt_grandchildren=dt_grandchildren if 'dt_grandchildren' in locals() else None, show=True)
                    print(f"   После пересоздания внуков: {len(tree_logic.grandchildren)}")

            print(f"✅ ФИНАЛЬНАЯ СТРУКТУРА ДЕРЕВА:")
            print(f"   Детей: {len(tree_logic.children)}")
            print(f"   Внуков: {len(getattr(tree_logic, 'grandchildren', []))}")
            print(f"   Корень в позиции: {tree_logic.root['position']}")

            # ДИАГНОСТИКА: Проверим позиции детей
            print(f"🔍 ПОЗИЦИИ ДЕТЕЙ:")
            for i, child in enumerate(tree_logic.children):
                print(f"   Ребенок {i}: {child['position']} (dt={child['dt']:.4f})")

            if hasattr(tree_logic, 'grandchildren'):
                print(f"🔍 ПОЗИЦИИ ВНУКОВ:")
                for i, gc in enumerate(tree_logic.grandchildren):
                    print(f"   Внук {i}: {gc['position']} (родитель {gc['parent_idx']}, dt={gc['dt']:.4f})")


            # Создаем визуализацию (ВРЕМЕННО)
            tree_visual = SporeTreeVisual(
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=self.config
            )

            tree_visual.set_tree_logic(tree_logic)
            tree_visual.create_visual()

            # =============================================
            # ЗАБИРАЕМ СОЗДАННЫЕ ОБЪЕКТЫ В ОБЩИЙ ГРАФ
            # =============================================

            created_spores = []
            created_links = []

            # Собираем все споры
            all_spores = [tree_visual.root_spore] + tree_visual.child_spores + tree_visual.grandchild_spores

            # Добавляем споры в общую систему (как обычные споры)
            for i, spore in enumerate(all_spores):
                if spore:
                    self.spore_manager.add_spore_manual(spore)
                    created_spores.append(spore)
                    # Регистрируем спору в ZoomManager с уникальным именем
                    self.zoom_manager.register_object(spore, f"tree_spore_{self._global_tree_counter}_{i}")

            # Добавляем линки в общий список (как обычные линки)
            all_links = tree_visual.child_links + tree_visual.grandchild_links
            for i, link in enumerate(all_links):
                if link:
                    self.created_links.append(link)
                    created_links.append(link)
                    # ✅ ДОБАВИТЬ РЕГИСТРАЦИЮ:
                    self.zoom_manager.register_object(link, f"tree_link_{self._global_tree_counter}_{i}")

            # =============================================
            # ОСВОБОЖДАЕМ SporeTreeVisual
            # =============================================
            #
            # Очищаем ссылки чтобы tree_visual не уничтожил объекты
            tree_visual.root_spore = None
            tree_visual.child_spores.clear()
            tree_visual.grandchild_spores.clear()
            tree_visual.child_links.clear()
            tree_visual.grandchild_links.clear()
            tree_visual.visual_created = False

            # tree_visual больше не нужен
            tree_visual = None
            tree_logic = None

            # =============================================
            # СОХРАНЯЕМ В ИСТОРИЮ (как обычную группу спор)
            # =============================================

            self.spore_groups_history.append(created_spores.copy())
            self.group_links_history.append(created_links.copy())

            print(f"🌲 Дерево создано в ({self.preview_manager.get_preview_position()[0]:.3f}, {self.preview_manager.get_preview_position()[1]:.3f})")
            print(f"   📊 Глубина: {self.tree_depth}, dt: {dt:.4f}")
            print(f"   🎯 Добавлено в граф: {len(created_spores)} спор + {len(created_links)} линков")
            print(f"   📚 Группа #{len(self.spore_groups_history)} сохранена в истории")

            return created_spores  # Возвращаем как обычную группу спор

        except Exception as e:
            print(f"❌ Ошибка создания дерева: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_current_dt(self):
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
        print(f"DEBUG: update_cursor_position вызван")
        print(f"   preview_enabled: {self.preview_manager.preview_enabled}")
        
        if not self.preview_manager.preview_enabled:
            print("   STOP: preview отключен")
            return

        # Получаем правильную позицию курсора мыши
        mouse_pos = self.get_mouse_world_position()
        print(f"   mouse_pos: {mouse_pos}")
        
        if mouse_pos is None:
            print("   STOP: mouse_pos = None")
            return

        # Используем PreviewManager для обновления позиции
        print("   Вызываем preview_manager.update_cursor_position...")
        self.preview_manager.update_cursor_position(mouse_pos)

        # Обновляем предсказания
        print(f"   show_predictions: {self.prediction_manager.show_predictions}")
        if self.prediction_manager.show_predictions:
            print("   Вызываем _update_predictions...")
            self._update_predictions()
        else:
            print("   SKIP: предсказания отключены")

    def _update_predictions(self) -> None:
        """Обновляет предсказания в зависимости от режима создания."""
        self.prediction_manager.update_predictions(
            preview_spore=self.preview_manager.get_preview_spore(),
            preview_position_2d=self.preview_manager.get_preview_position(),
            creation_mode=self.creation_mode,
            tree_depth=self.tree_depth
        )

    def create_spore_at_cursor(self):
        """Создает споры или дерево в зависимости от режима."""
        if self.creation_mode == 'tree':
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
            center_id = self._get_next_spore_id()
            center_spore = Spore(
                pendulum=self.pendulum,
                dt=dt,
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_manager.get_preview_position()[0], 0.0, self.preview_manager.get_preview_position()[1]),
                color_manager=self.color_manager,
                config=spore_config
            )

            # Добавляем центральную спору в систему
            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_id}')
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
                child_id = self._get_next_spore_id()

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

                # Добавляем спору в систему БЕЗ автоматических призраков
                self.spore_manager.add_spore_manual(child_spore)
                self.zoom_manager.register_object(child_spore, f'manual_{config["name"]}_{child_id}')

                # Переопределяем управление на конкретное значение
                child_spore.logic.optimal_control = np.array([config['control']])

                created_spores.append(child_spore)
                print(f"   ✓ Создана спора {config['name']} в позиции ({child_pos_2d[0]:.3f}, {child_pos_2d[1]:.3f}) с управлением {config['control']:.2f}")

                # 3. Создаем ЛИНК с правильным направлением и цветом
                link_id = self._get_next_link_id()

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
                self.zoom_manager.register_object(spore_link, f'manual_link_{config["name"]}_{link_id}')

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
                    # Дерегистрируем из zoom_manager
                    # Нужно найти правильный ключ - проверяем registered objects
                    if hasattr(self.zoom_manager, 'objects'):
                        for key, obj in list(self.zoom_manager.objects.items()):
                            if obj is link:
                                self.zoom_manager.unregister_object(key)
                                print(f"   ✓ Линк {i+1} дерегистрирован: {key}")
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

                    # Дерегистрируем из zoom_manager
                    if hasattr(self.zoom_manager, 'objects'):
                        for key, obj in list(self.zoom_manager.objects.items()):
                            if obj is spore:
                                self.zoom_manager.unregister_object(key)
                                print(f"   ✓ Спора {i+1} дерегистрирована: {key}")
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