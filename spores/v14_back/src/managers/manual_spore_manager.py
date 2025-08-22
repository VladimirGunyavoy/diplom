from typing import Optional, Tuple, List
import numpy as np
from ursina import Vec3, color, destroy, mouse, camera, raycast, Vec2

from ..core.spore import Spore
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..managers.color_manager import ColorManager
from ..logic.pendulum import PendulumSystem
from ..logic.tree.pairs.find_optimal_pairs import find_optimal_pairs
from ..logic.tree.pairs.extract_optimal_times_from_pairs import extract_optimal_times_from_pairs
from ..visual.prediction_visualizer import PredictionVisualizer
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
        
        # Настройки превью
        self.preview_enabled = True
        self.preview_alpha = 0.5  # Полупрозрачность
        self.debug_mode = False  # Флаг для отладочного вывода
        
        # Превью спора
        self.preview_spore: Optional[Spore] = None
        self.preview_position_2d = np.array([0.0, 0.0], dtype=float)
        
        # Предсказания min/max управления
        self.prediction_visualizers: List[PredictionVisualizer] = []
        self.prediction_links: List[Link] = []  # Линки от превью споры к призракам
        self.created_links: List[Link] = []
        self.show_predictions = True
        
        # Получаем границы управления из маятника
        control_bounds = self.pendulum.get_control_bounds()
        self.min_control = float(control_bounds[0])
        self.max_control = float(control_bounds[1])

        self._link_counter = 0
        self._spore_counter = 0

        # Сохранение dt вектора от призрачного дерева
        self.ghost_tree_dt_vector = None
        self.ghost_tree_optimized = False
        self.last_known_dt = None  # ✅ ДОБАВИТЬ: для отслеживания изменений dt
        self.tree_created_with_dt = None  # ✅ ДОБАВИТЬ: для отслеживания созданного dt

        # История созданных групп спор для возможности удаления
        self.spore_groups_history: List[List[Spore]] = []  # История групп спор
        self.group_links_history: List[List[Link]] = []    # История линков для каждой группы

        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2 
        self._global_tree_counter = 0

        print(f"   ✓ Manual Spore Manager создан (управление: {self.min_control} .. {self.max_control})")
        print(f"   📚 История групп инициализирована")




    
        # 🆕 Добавляем поддержку деревьев
        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2  # 1 или 2
        # НЕ НУЖНО: self.active_trees = [] - споры идут в общий граф!
    
    print(f"   🌲 Поддержка деревьев добавлена")

# В том же файле добавить методы:

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
        if not self.preview_enabled or not self.preview_spore:
            return None

        try:
            from ..visual.spore_tree_visual import SporeTreeVisual
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig

            self._global_tree_counter += 1

            # Получаем текущий dt
            dt = self._get_current_dt()

            # ИСПРАВЛЕНИЕ: Правильная позиция для дерева
            tree_position = np.array([self.preview_position_2d[0], self.preview_position_2d[1]])

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
            
            print(f"🌲 Дерево создано в ({self.preview_position_2d[0]:.3f}, {self.preview_position_2d[1]:.3f})")
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
        """Получает текущий dt из DTManager или конфига."""
        if hasattr(self, 'dt_manager') and self.dt_manager:
            return self.dt_manager.get_current_dt()
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
        if not self.preview_enabled:
            return


        
        # Получаем правильную позицию курсора мыши
        mouse_pos = self.get_mouse_world_position()
        if mouse_pos is None:
            return
            
        # Обновляем позицию
        self.preview_position_2d[0] = mouse_pos[0]
        self.preview_position_2d[1] = mouse_pos[1]
        
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
    
    def _update_preview_spore(self) -> None:
        """Создает или обновляет превью спору."""
        if not self.preview_spore:
            self._create_preview_spore()
        else:
            # Обновляем позицию существующей споры
            self.preview_spore.real_position = Vec3(
                self.preview_position_2d[0], 
                0.0, 
                self.preview_position_2d[1]
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
            
            self.preview_spore = Spore(
                pendulum=self.pendulum,
                dt=pendulum_config.get('dt', 0.1),
                goal_position=goal_position,
                scale=spore_config.get('scale', 0.1),
                position=(self.preview_position_2d[0], 0.0, self.preview_position_2d[1]),
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

            if hasattr(self, 'dt_manager') and self.dt_manager:
                dt = self.dt_manager.get_dt()
            else:
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

            for i, config in enumerate(prediction_configs):
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    predicted_pos_2d = self.pendulum.step(
                        self.preview_position_2d,
                        config['control'],
                        dt,
                        method='jit'
                    )
                else:  # backward
                    # Шаг назад во времени
                    predicted_pos_2d = self.pendulum.step(
                        self.preview_position_2d,
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

                # # Создаем линк от превью споры к призраку
                # if prediction_viz.ghost_spore:
                #     prediction_link = Link(
                #         parent_spore=self.preview_spore,
                #         child_spore=prediction_viz.ghost_spore,
                #         color_manager=self.color_manager,
                #         zoom_manager=self.zoom_manager,
                #         config=self.config
                #     )

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

                    prediction_link.color = self.color_manager.get_color('link', link_color_name)

                    # Для линков назад делаем более тонкими или пунктирными
                    if config['direction'] == 'backward':
                        # Можно добавить дополнительную стилизацию линков назад
                        pass

                    # Обновляем геометрию и регистрируем в zoom manager
                    prediction_link.update_geometry()
                    self.zoom_manager.register_object(
                        prediction_link,
                        f'manual_prediction_link_{config["name"]}'
                    )

                    self.prediction_links.append(prediction_link)

                self.prediction_visualizers.append(prediction_viz)

        except Exception as e:
            print(f"Ошибка обновления предсказаний: {e}")
            import traceback
            traceback.print_exc()

    def _clear_predictions(self) -> None:
        """Очищает все предсказания и их линки."""
        # Сначала дерегистрируем из zoom_manager, ПОТОМ уничтожаем
        for i, viz in enumerate(self.prediction_visualizers):
            if viz.ghost_spore:
                # Ищем и дерегистрируем ghost_spore
                ghost_id = getattr(viz.ghost_spore, 'id', f'tree_ghost_{i}')
                try:
                    self.zoom_manager.unregister_object(ghost_id)
                except:
                    pass
            viz.destroy()
        self.prediction_visualizers.clear()

        # Очищаем линки предсказаний
        for i, link in enumerate(self.prediction_links):
            # Пытаемся найти правильный ключ регистрации
            possible_keys = [
                f'manual_prediction_link_{["forward_min", "forward_max", "backward_min", "backward_max"][i] if i < 4 else i}',
                f'ghost_link_root_to_child_{i}',
                f'ghost_link_child_{i//2}_to_grandchild_{i}'
            ]

            for key in possible_keys:
                try:
                    self.zoom_manager.unregister_object(key)
                    break
                except:
                    continue

            try:
                destroy(link)
            except:
                pass
        self.prediction_links.clear()

    def _update_ghost_predictions(self):
        """Обновляет призрачные предсказания дерева."""
        # Очищаем старые предсказания
        self._clear_predictions()

        if not self.preview_spore:
            return

        # Получаем текущий dt
        current_dt = self._get_current_dt()

        # Проверяем изменение dt
        dt_changed = False
        if hasattr(self, 'last_known_dt') and self.last_known_dt is not None:
            if abs(current_dt - self.last_known_dt) > 1e-10:  # dt изменился
                dt_changed = True
                print(f"🔄 dt изменился: {self.last_known_dt:.6f} → {current_dt:.6f}")

                # Если была оптимизация - сбрасываем её
                if hasattr(self, 'ghost_tree_optimized') and self.ghost_tree_optimized:
                    print(f"🔓 Сброс оптимизации из-за изменения dt")
                    self.ghost_tree_optimized = False
                    self.ghost_tree_dt_vector = None
                    self.tree_created_with_dt = None  # ✅ Сбрасываем созданный dt

        # Сохраняем текущий dt для следующего раза
        self.last_known_dt = current_dt

        # Определяем режим создания дерева
        use_optimized_dt = (hasattr(self, 'ghost_tree_optimized') and
                            self.ghost_tree_optimized and
                            hasattr(self, 'ghost_tree_dt_vector') and
                            self.ghost_tree_dt_vector is not None and
                            not dt_changed)  # ✅ НЕ используем оптимизированные dt если dt изменился

        if use_optimized_dt:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print("🎯 Используем заблокированные оптимизированные dt")
        else:
            # Выводим сообщение только если dt изменился или это первый вызов
            if dt_changed:
                print(f"🔄 Пересоздаем дерево с новым dt: {current_dt:.6f}")
            elif not hasattr(self, 'tree_created_with_dt') or self.tree_created_with_dt != current_dt:
                print(f"🌲 Создаем дерево с dt: {current_dt:.6f}")
                self.tree_created_with_dt = current_dt
            # Иначе - дерево уже создано с этим dt, не выводим сообщение

        try:
            # Импорты для дерева
            from ..logic.tree.spore_tree import SporeTree
            from ..logic.tree.spore_tree_config import SporeTreeConfig

            # Получаем текущий dt
            dt = self._get_current_dt()

            # Создаем конфиг дерева
            tree_config = SporeTreeConfig(
                initial_position=self.preview_position_2d.copy(),
                dt_base=dt,
                dt_grandchildren_factor=0.2,
                show_debug=False
            )

            # ПРОВЕРЯЕМ: используем ли оптимизированные dt
            if use_optimized_dt:
                # Используем сохраненные оптимизированные dt с сохранением знаков
                dt_children = self.ghost_tree_dt_vector[:4]  # ✅ ИСПРАВЛЕНИЕ: сохраняем знаки
                dt_grandchildren = self.ghost_tree_dt_vector[4:12]  # ✅ ИСПРАВЛЕНИЕ: сохраняем знаки

                # Проверяем размеры векторов перед использованием
                assert len(dt_children) == 4, f"Ожидается 4 dt для детей, получено {len(dt_children)}"
                assert len(dt_grandchildren) == 8, f"Ожидается 8 dt для внуков, получено {len(dt_grandchildren)}"

                # Отладочные принты для проверки что dt действительно применяются
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"🎯 Создаем дерево с оптимизированными dt:")
                    print(f"   dt_children: {dt_children}")
                    print(f"   dt_grandchildren: {dt_grandchildren}")

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children,  # ✅ ИСПРАВЛЕНИЕ: передаем подписанные dt в конструктор
                    dt_grandchildren=dt_grandchildren,  # ✅ ИСПРАВЛЕНИЕ: передаем подписанные dt в конструктор
                    auto_create=True,  # ✅ ИСПРАВЛЕНИЕ: используем auto_create=True
                    show=False
                )
                # ✅ НЕ вызываем create_children() и create_grandchildren() - дерево уже создано с правильными dt!

                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print("🎯 Используем оптимизированные dt для призрачного дерева")
            else:
                # Создаем обычное дерево с стандартными dt (dt из dt-менеджера для детей, в 5 раз меньше для внуков)
                current_dt = self._get_current_dt()

                # Стандартные dt: детям - полный dt, внукам - в 5 раз меньше
                dt_children_std = np.array([current_dt, -current_dt, current_dt, -current_dt])  # ✅ с правильными знаками
                dt_grandchildren_std = np.array([
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_0
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_1
                    current_dt * 0.2, -current_dt * 0.2,  # внуки от child_2
                    current_dt * 0.2, -current_dt * 0.2   # внуки от child_3
                ])

                tree_logic = SporeTree(
                    pendulum=self.pendulum,
                    config=tree_config,
                    dt_children=dt_children_std,  # ✅ ИСПРАВЛЕНИЕ: передаем стандартные dt с правильными знаками
                    dt_grandchildren=dt_grandchildren_std,  # ✅ ИСПРАВЛЕНИЕ: передаем стандартные dt с правильными знаками
                    auto_create=True,  # ✅ ИСПРАВЛЕНИЕ: используем auto_create=True
                    show=False
                )

            # Сохраняем ссылку на призрачное дерево для дальнейшего использования
            self.current_ghost_tree = tree_logic

            # ✅ НЕ вызываем create_children() и create_grandchildren() - дерево уже создано с auto_create=True!

            # Сохраняем dt вектор ТОЛЬКО если его еще нет
            if hasattr(tree_logic, 'children') and hasattr(tree_logic, 'grandchildren'):
                if not hasattr(self, 'ghost_tree_dt_vector') or self.ghost_tree_dt_vector is None:
                    try:
                        # Извлекаем dt из дерева
                        dt_children = [child.get('dt', dt) for child in tree_logic.children]
                        dt_grandchildren = [gc.get('dt', dt * 0.2) for gc in tree_logic.grandchildren]
                        self.ghost_tree_dt_vector = np.hstack([dt_children, dt_grandchildren])
                    except Exception as e:
                        print(f"⚠️ Ошибка сохранения dt вектора: {e}")
                        self.ghost_tree_dt_vector = None

            # Конвертируем в призрачные предсказания
            self._create_ghost_tree_from_logic(tree_logic)

        except Exception as e:
            print(f"Ошибка создания призрачного дерева: {e}")

    def _create_ghost_tree_from_logic(self, tree_logic):
        """Создает призрачные споры и линки из логики дерева."""
        # Создаем призрачные споры для детей
        child_ghosts = []
        for i, child_data in enumerate(tree_logic.children):
            ghost_viz = self._create_ghost_spore_from_data(child_data, f"tree_child_{i}", 0.4)
            if ghost_viz and ghost_viz.ghost_spore:
                child_ghosts.append(ghost_viz.ghost_spore)

        # Создаем призрачные споры для внуков (если есть)
        grandchild_ghosts = []
        if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            for i, grandchild_data in enumerate(tree_logic.grandchildren):
                ghost_viz = self._create_ghost_spore_from_data(grandchild_data, f"tree_grandchild_{i}", 0.3)
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
                    self.preview_spore,
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
        from ..visual.prediction_visualizer import PredictionVisualizer

        # Получаем финальную позицию споры
        final_position = spore_data['position']  # должно быть [x, z]

        # Создаем визуализатор предсказания
        prediction_viz = PredictionVisualizer(
            parent_spore=self.preview_spore,
            color_manager=self.color_manager,
            zoom_manager=self.zoom_manager,
            cost_function=None,
            config={
                'spore': {'show_ghosts': True},
                'angel': {'show_angels': False, 'show_pillars': False}
            },
            spore_id=f'tree_ghost_{name_suffix}'
        )

        # Устанавливаем позицию призрака
        if prediction_viz.ghost_spore:
            # Устанавливаем полупрозрачность
            base_color = self.color_manager.get_color('spore', 'default')
            try:
                prediction_viz.ghost_spore.color = (base_color.r, base_color.g, base_color.b, alpha)
            except:
                prediction_viz.ghost_spore.color = (0.6, 0.4, 0.9, alpha)

            # Обновляем позицию
            prediction_viz.update(final_position)

            # Безопасно регистрируем призрака в zoom_manager
            try:
                ghost_id = f'tree_ghost_{name_suffix}'
                prediction_viz.ghost_spore.id = ghost_id
                self.zoom_manager.register_object(prediction_viz.ghost_spore, ghost_id)
            except Exception as e:
                print(f"Ошибка регистрации призрака {name_suffix}: {e}")

        # Добавляем в список предсказаний
        self.prediction_visualizers.append(prediction_viz)
        return prediction_viz

    def _create_ghost_link(self, parent_spore, child_spore, link_suffix, color_name):
        """Создает призрачный линк между двумя спорами."""
        try:
            from ..visual.link import Link

            ghost_link = Link(
                parent_spore=parent_spore,
                child_spore=child_spore,
                color_manager=self.color_manager,
                zoom_manager=self.zoom_manager,
                config=self.config
            )

            # Устанавливаем цвет линка
            ghost_link.color = self.color_manager.get_color('link', color_name)

            # Делаем линк полупрозрачным
            if hasattr(ghost_link, 'alpha'):
                ghost_link.alpha = 0.6

            # Обновляем геометрию и регистрируем
            ghost_link.update_geometry()
            link_id = f'ghost_link_{link_suffix}'
            self.zoom_manager.register_object(ghost_link, link_id)

            # Добавляем в список для очистки
            self.prediction_links.append(ghost_link)

        except Exception as e:
            print(f"Ошибка создания призрачного линка {link_suffix}: {e}")

    def create_spore_at_cursor(self):
        """Создает споры или дерево в зависимости от режима."""
        if self.creation_mode == 'tree':
            return self.create_tree_at_cursor()
        else:
            # ... весь существующий код создания спор ...
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
        if not self.preview_enabled or not self.preview_spore:
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
                position=(self.preview_position_2d[0], 0.0, self.preview_position_2d[1]),
                color_manager=self.color_manager,
                config=spore_config
            )
            
            # Добавляем центральную спору в систему
            self.spore_manager.add_spore_manual(center_spore)
            self.zoom_manager.register_object(center_spore, f'manual_center_{center_id}')
            created_spores.append(center_spore)
            print(f"   ✓ Создана центральная спора в позиции ({self.preview_position_2d[0]:.3f}, {self.preview_position_2d[1]:.3f})")
            
            # 2. Создаем ДОЧЕРНИЕ споры (forward) + РОДИТЕЛЬСКИЕ споры (backward)
            spore_configs = [
                # Дочерние (forward)
                {'control': self.min_control, 'name': 'forward_min', 'color': 'ghost_min', 'direction': 'forward'},
                {'control': self.max_control, 'name': 'forward_max', 'color': 'ghost_max', 'direction': 'forward'},
                # Родительские (backward) 
                {'control': self.min_control, 'name': 'backward_min', 'color': 'ghost_min', 'direction': 'backward'},
                {'control': self.max_control, 'name': 'backward_max', 'color': 'ghost_max', 'direction': 'backward'}
            ]
            
            for config in spore_configs:
                child_id = self._get_next_spore_id()
                
                # Вычисляем позицию в зависимости от направления
                if config['direction'] == 'forward':
                    # Обычный шаг вперед
                    child_pos_2d = self.pendulum.step(
                        self.preview_position_2d, 
                        config['control'], 
                        dt,
                        method='jit'
                    )
                else:  # backward
                    # Шаг назад во времени
                    child_pos_2d = self.pendulum.step(
                        self.preview_position_2d, 
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

            print(f"   🎯 Создано ВСЕГО: {len(created_spores)} спор + {len(created_links)} линков")
            print(f"   📊 Состав: 1 центральная + 2 дочерние (forward) + 2 родительские (backward)")
            
            # 🆕 СОХРАНЕНИЕ В ИСТОРИЮ для возможности удаления
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
        """Уничтожает превью спору, предсказания и их линки."""
        if self.preview_spore:
            self.zoom_manager.unregister_object('manual_preview')
            destroy(self.preview_spore)
            self.preview_spore = None
            
        self._clear_predictions()  # Это очистит и визуализаторы, и линки
    
    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
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

    def destroy(self) -> None:
        """Очищает все ресурсы менеджера."""
        self.clear_all()  # Используем новый метод
        print("   ✓ Manual Spore Manager уничтожен")

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

    def update_ghost_tree_with_optimal_pairs(self):
        """Обновляет призрачное дерево оптимальными dt из найденных пар."""
        if not self.preview_spore or not hasattr(self, 'current_ghost_tree'):
            print("⚠️ Нет призрачного дерева для обновления")
            return

        try:
            print("🔍 Поиск оптимальных пар для текущего призрачного дерева...")

            # Ищем оптимальные пары в уже созданном призрачном дереве
            pairs = find_optimal_pairs(self.current_ghost_tree, show=True)

            if pairs is None or len(pairs) == 0:
                print("⚠️ Не найдено оптимальных пар")
                return

            print(f"✅ Найдено {len(pairs)} оптимальных пар")

            # ИСПОЛЬЗУЕМ ПРАВИЛЬНУЮ ФУНКЦИЮ для извлечения dt
            dt_results = extract_optimal_times_from_pairs(pairs, self.current_ghost_tree, show=True)

            if dt_results is None:
                print("❌ Ошибка извлечения оптимальных времен")
                return

            # Получаем оптимизированные dt из результата
            optimal_dt_children = dt_results['dt_children']
            optimal_dt_grandchildren = dt_results['dt_grandchildren']

            print(f"🔍 РЕЗУЛЬТАТ ИЗВЛЕЧЕНИЯ:")
            print(f"   Обновлено внуков: {dt_results['stats']['changed_count']}/{dt_results['stats']['total_grandchildren']}")
            print(f"   Пар использовано: {len(pairs)}")
            print(f"   Неспаренных внуков: {len(dt_results['unpaired_grandchildren'])}")

            # Обновляем сохраненный dt вектор
            old_dt_vector = self.ghost_tree_dt_vector.copy() if self.ghost_tree_dt_vector is not None else None
            self.ghost_tree_dt_vector = np.hstack([optimal_dt_children, optimal_dt_grandchildren])

            print(f"🔍 СРАВНЕНИЕ dt векторов:")
            if old_dt_vector is not None:
                print(f"   Старый: {old_dt_vector}")
            print(f"   Новый:  {self.ghost_tree_dt_vector}")

            # БЛОКИРУЕМ автообновление при движении мыши
            self.ghost_tree_optimized = True

            # ✅ НОВОЕ: Фиксируем текущий dt для отслеживания изменений
            self.last_known_dt = self._get_current_dt()

            # Принудительно пересоздаем призрачную визуализацию
            print("🔄 Принудительное пересоздание призрачного дерева...")
            self._update_ghost_predictions()

            print(f"🎯 Призрачное дерево обновлено оптимальными dt из {len(pairs)} пар!")
            print(f"   📊 Изменено dt у {dt_results['stats']['changed_count']} внуков")

        except Exception as e:
            print(f"❌ Ошибка обновления призрачного дерева: {e}")
            import traceback
            traceback.print_exc()

    def reset_ghost_tree_optimization(self):
        """Сбрасывает блокировку оптимизированного призрачного дерева."""
        if hasattr(self, 'ghost_tree_optimized'):
            self.ghost_tree_optimized = False
            self.ghost_tree_dt_vector = None
            self.tree_created_with_dt = None  # ✅ Сбрасываем также созданный dt
            print("🔓 Оптимизация призрачного дерева сброшена")

    def _get_current_dt(self) -> float:
        """Получает текущий dt из DTManager или конфига."""
        if hasattr(self, 'dt_manager') and self.dt_manager:
            # Проверяем разные возможные названия метода
            if hasattr(self.dt_manager, 'get_current_dt'):
                return self.dt_manager.get_current_dt()
            elif hasattr(self.dt_manager, 'get_dt'):
                return self.dt_manager.get_dt()
            else:
                print("⚠️ DTManager не имеет метода get_dt() или get_current_dt()")
        return self.config.get('pendulum', {}).get('dt', 0.1)

