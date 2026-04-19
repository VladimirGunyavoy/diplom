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
        
        # 🔍 Флаг для детальной отладки призрачного дерева
        self.debug_ghost_tree = False

        print(f"   ✓ Prediction Manager создан (управление: {self.min_control} .. {self.max_control})")

        # ═══════════════════════════════════════════════════════════════
        # 🚀 ОПТИМИЗАЦИЯ ЭТАП 1: Кэширование объектов предсказаний
        # ═══════════════════════════════════════════════════════════════
        
        # Кэш созданных объектов предсказаний (избегаем пересоздания каждый кадр)
        self._prediction_objects_cache = {}  # {cache_key: {'visualizers': [...], 'links': [...]}}
        self._cache_initialized = False      # Флаг инициализации кэша
        self._current_mode = None           # Текущий режим создания для инвалидации
        self._current_tree_depth = None     # Текущая глубина дерева для инвалидации
        
        print("   🚀 Кэширование предсказаний инициализировано (Этап 1)")

    def update_predictions(self, preview_spore, preview_position_2d: np.ndarray, creation_mode: str, tree_depth: int, ghost_dt_vector=None) -> None:
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
            self._update_tree_preview(preview_spore, preview_position_2d, ghost_dt_vector)
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

    def _update_tree_preview(self, preview_spore, preview_position_2d: np.ndarray, ghost_dt_vector=None) -> None:
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

            # Создаем логику дерева с учетом ghost_tree_dt_vector
            if ghost_dt_vector is not None and len(ghost_dt_vector) == 12:
                # print(f"🎯 Используем оптимизированный ghost_tree_dt_vector для призрачного дерева")  # Отключен для избежания спама
                
                # Извлекаем dt из вектора (берем абсолютные значения для SporeTree)
                dt_children_abs = np.abs(ghost_dt_vector[:4])
                dt_grandchildren_abs = np.abs(ghost_dt_vector[4:12])
                
                tree_logic = SporeTree(
                    pendulum=self.deps.pendulum,
                    config=tree_config,
                    dt_children=dt_children_abs,
                    dt_grandchildren=dt_grandchildren_abs,
                    auto_create=False
                )
            else:
                # Стандартное создание
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

            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пересчитываем позиции с новыми dt
            if ghost_dt_vector is not None and len(ghost_dt_vector) == 12:
                self._recalculate_positions_with_new_dt(tree_logic, ghost_dt_vector, preview_position_2d)

            # DEBUG: Проверяем что dt правильно применились к дереву (отключено для избежания спама)
            # print(f"🔍 DEBUG: tree_logic создан:")
            # print(f"   Дети dt: {[child['dt'] for child in tree_logic.children]}")
            # if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            #     print(f"   Внуки dt: {[gc['dt'] for gc in tree_logic.grandchildren]}")

            # Конвертируем в призрачные предсказания
            self._create_ghost_tree_from_logic(tree_logic, preview_spore)

        except Exception as e:
            print(f"Ошибка создания призрачного дерева: {e}")

    def _recalculate_positions_with_new_dt(self, tree_logic, ghost_dt_vector, initial_position):
        """
        🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пересчитывает позиции всех узлов дерева с новыми dt.
        
        Args:
            tree_logic: SporeTree с новыми dt
            ghost_dt_vector: Вектор из 12 dt (4 детей + 8 внуков)
            initial_position: Начальная позиция корня дерева
        """
        try:
            if self.debug_ghost_tree:
                print(f"   🔧 ПЕРЕСЧЕТ ПОЗИЦИЙ: Начинаем пересчет с новыми dt")
                print(f"      Начальная позиция: {initial_position}")
                print(f"      Новые dt детей: {ghost_dt_vector[:4]}")
                print(f"      Новые dt внуков: {ghost_dt_vector[4:12]}")
            
            # Пересчитываем позиции детей
            for i, child_data in enumerate(tree_logic.children):
                if i < len(ghost_dt_vector[:4]):
                    new_dt = ghost_dt_vector[i]
                    # Получаем управление ребенка
                    control = child_data.get('control', 0.0)
                    # Используем pendulum для расчета новой позиции
                    new_position = self.deps.pendulum.step(initial_position, control, new_dt)
                    # Обновляем позицию в данных дерева
                    child_data['position'] = new_position
                    
                    if self.debug_ghost_tree:
                        old_pos = child_data.get('original_position', 'N/A')
                        print(f"      Ребенок {i}: dt={new_dt:+.6f}, control={control:+.6f}, pos={old_pos} → {new_position}")
                        # Сохраняем оригинальную позицию для сравнения
                        child_data['original_position'] = old_pos
            
            # Пересчитываем позиции внуков
            if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
                for i, grandchild_data in enumerate(tree_logic.grandchildren):
                    if i < len(ghost_dt_vector[4:12]):
                        new_dt = ghost_dt_vector[4 + i]
                        # Получаем управление внука
                        control = grandchild_data.get('control', 0.0)
                        # Получаем позицию родителя внука
                        parent_idx = grandchild_data['parent_idx']
                        if parent_idx < len(tree_logic.children):
                            parent_position = tree_logic.children[parent_idx]['position']
                            # Используем pendulum для расчета новой позиции внука
                            new_position = self.deps.pendulum.step(parent_position, control, new_dt)
                            # Обновляем позицию в данных дерева
                            grandchild_data['position'] = new_position
                            
                            if self.debug_ghost_tree:
                                old_pos = grandchild_data.get('original_position', 'N/A')
                                print(f"      Внук {i}: dt={new_dt:+.6f}, control={control:+.6f}, pos={old_pos} → {new_position}")
                                # Сохраняем оригинальную позицию для сравнения
                                grandchild_data['original_position'] = old_pos
            
            if self.debug_ghost_tree:
                print(f"   🔧 ПЕРЕСЧЕТ ПОЗИЦИЙ: Завершен")
                
        except Exception as e:
            print(f"❌ Ошибка пересчета позиций: {e}")
            import traceback
            traceback.print_exc()

    def _create_ghost_tree_from_logic(self, tree_logic, preview_spore):
        """Создает призрачные споры и линки из логики дерева."""

        # Создаем призрачные споры для детей
        child_ghosts = []
        for i, child_data in enumerate(tree_logic.children):
            # 🔍 ДИАГНОСТИКА: Показываем данные ребенка перед созданием призрака (если включена отладка)
            if hasattr(self, 'debug_ghost_tree') and self.debug_ghost_tree:
                print(f"   🔍 Создаем призрак ребенка {i}:")
                print(f"      Исходные данные: dt={child_data['dt']:+.6f}, pos={child_data['position']}")
            
            ghost_viz = self._create_ghost_spore_from_data(child_data, f"child_{i}", 0.4)
            if ghost_viz and ghost_viz.ghost_spore:
                child_ghosts.append(ghost_viz.ghost_spore)

        # DEBUG: Показываем позиции призрачных детей для проверки обновления (отключено для избежания спама)
        # if len(child_ghosts) > 0:
        #     print(f"🔍 DEBUG: Позиции призрачных детей:")
        #     for i, (child_ghost, child_data) in enumerate(zip(child_ghosts, tree_logic.children)):
        #         if child_ghost:
        #             actual_pos = (child_ghost.x, child_ghost.z)  
        #             expected_pos = (child_data['position'][0], child_data['position'][1])
        #             print(f"   Child {i}: expected={expected_pos}, actual={actual_pos}, dt={child_data['dt']:+.6f}")

        # Создаем призрачные споры для внуков (если есть)
        grandchild_ghosts = []
        if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            for i, grandchild_data in enumerate(tree_logic.grandchildren):
                # 🔍 ДИАГНОСТИКА: Показываем данные внука перед созданием призрака (если включена отладка)
                if self.debug_ghost_tree:
                    print(f"   🔍 Создаем призрак внука {i}:")
                    print(f"      Исходные данные: dt={grandchild_data['dt']:+.6f}, pos={grandchild_data['position']}")
                
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
        
        # 🔍 ДИАГНОСТИКА: Показываем что получаем для создания призрака (если включена отладка)
        if self.debug_ghost_tree:
            print(f"         🔍 Создание призрака:")
            print(f"            final_position: {final_position}")
            print(f"            spore_data keys: {list(spore_data.keys())}")

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

    # ═══════════════════════════════════════════════════════════════
    # 🚀 МЕТОДЫ КЭШИРОВАНИЯ - ЭТАП 1
    # ═══════════════════════════════════════════════════════════════

    def _recreate_prediction_cache(self, creation_mode: str, tree_depth: int) -> None:
        """
        Пересоздает кэш объектов предсказаний (вызывается редко - только при смене режима).
        
        Args:
            creation_mode: 'spores' или 'tree'  
            tree_depth: глубина дерева для tree режима
        """
        # Очищаем старый кэш
        self._clear_prediction_cache()
        
        cache_key = f"{creation_mode}_{tree_depth}"
        print(f"🔄 Пересоздание кэша предсказаний: {cache_key}")
        
        if creation_mode == 'tree':
            # TODO: Реализовать после тестирования spores режима
            print(f"   ⚠️ Tree режим пока не оптимизирован, используем старую логику")
            return
        else:
            # Для spores режима - создаем 4 стандартных предсказания
            self._create_spore_prediction_cache()
            
        # Обновляем состояние кэша
        self._cache_initialized = True
        self._current_mode = creation_mode
        self._current_tree_depth = tree_depth
        
        print(f"   ✅ Кэш создан: {len(self.prediction_visualizers)} предсказаний")

    def _create_spore_prediction_cache(self) -> None:
        """
        Создает кэшированные объекты для режима 'spores' - 4 направления min/max x forward/backward.
        """
        self.prediction_visualizers.clear()
        self.prediction_links.clear()
        
        # Временная позиция - будет обновлена в _update_cached_predictions_positions
        temp_pos = np.array([0.0, 0.0], dtype=float)
        
        # Конфигурации для 4 предсказаний (как в существующем коде)
        predictions_config = [
            {'name': 'forward_min', 'control': self.min_control, 'direction': 'forward'},
            {'name': 'forward_max', 'control': self.max_control, 'direction': 'forward'},
            {'name': 'backward_min', 'control': self.min_control, 'direction': 'backward'},
            {'name': 'backward_max', 'control': self.max_control, 'direction': 'backward'}
        ]
        
        for config in predictions_config:
            # Создаем PredictionVisualizer (аналогично существующему коду)
            prediction_viz = PredictionVisualizer(
                parent_spore=None,  # Пока None, обновим при первом использовании
                color_manager=self.deps.color_manager,
                zoom_manager=self.deps.zoom_manager,
                cost_function=None,
                config={
                    'spore': {'show_ghosts': True},
                    'angel': {'show_angels': False, 'show_pillars': False}
                },
                spore_id=self.deps.zoom_manager.get_unique_spore_id()
            )
            
            # Сохраняем конфигурацию в объекте для быстрого доступа
            prediction_viz.name = config['name']  # Добавляем name для идентификации
            
            # Временно устанавливаем позицию в [0,0]
            if prediction_viz.ghost_spore:
                prediction_viz.update_position_only(temp_pos)
            
            self.prediction_visualizers.append(prediction_viz)

    def _update_cached_predictions_positions(self, preview_spore, preview_position_2d: np.ndarray,
                                           creation_mode: str, tree_depth: int, ghost_dt_vector=None) -> None:
        """
        БЫСТРОЕ обновление позиций кэшированных объектов (каждый кадр).
        НЕ создает новые объекты, только обновляет positions используя update_position_only().
        
        Args:
            preview_spore: превью спора
            preview_position_2d: позиция превью
            creation_mode: режим создания
            tree_depth: глубина дерева
            ghost_dt_vector: вектор dt для tree режима
        """
        if creation_mode == 'tree':
            # TODO: быстрое обновление для tree режима
            print("   ⚠️ Tree режим - быстрое обновление пока не реализовано")
            return
            
        # Для spores режима - обновляем 4 предсказания
        dt = self._get_current_dt()
        
        for prediction_viz in self.prediction_visualizers:
            if not prediction_viz.ghost_spore or not hasattr(prediction_viz, 'name'):
                continue
                
            # Определяем параметры из имени (кэшированная конфигурация)
            if 'max' in prediction_viz.name:
                control = self.max_control
            else:
                control = self.min_control
                
            if 'forward' in prediction_viz.name:
                dt_direction = dt  # положительный dt
            else:
                dt_direction = -dt  # отрицательный dt для backward
            
            # Вычисляем новую позицию предсказания
            try:
                predicted_pos = self.deps.pendulum.step(
                    state=preview_position_2d,
                    control=control,
                    dt=dt_direction
                )
                
                # БЫСТРОЕ ОБНОВЛЕНИЕ - используем новый оптимизированный метод!
                prediction_viz.update_position_only(predicted_pos)
                
            except Exception as e:
                print(f"   ❌ Ошибка обновления предсказания {prediction_viz.name}: {e}")

    def _clear_prediction_cache(self) -> None:
        """
        Очищает кэш предсказаний и освобождает ресурсы.
        """
        # Очищаем visualizers
        for viz in self.prediction_visualizers:
            try:
                # Дерегистрируем из zoom_manager если было зарегистрировано
                if hasattr(viz, '_zoom_manager_key'):
                    self.deps.zoom_manager.unregister_object(viz._zoom_manager_key)
                # Уничтожаем объект
                viz.destroy()
            except Exception as e:
                print(f"   ⚠️ Ошибка при удалении visualizer: {e}")
                
        # Очищаем links (пока не используем в кэше)
        for link in self.prediction_links:
            try:
                if hasattr(link, '_zoom_manager_key'):
                    self.deps.zoom_manager.unregister_object(link._zoom_manager_key)
                destroy(link)
            except Exception as e:
                print(f"   ⚠️ Ошибка при удалении link: {e}")
                
        self.prediction_visualizers.clear()
        self.prediction_links.clear()

    def clear_predictions(self) -> None:
        """Публичный метод очистки (для обратной совместимости)."""
        # Вызываем новый метод очистки кэша
        self._clear_prediction_cache()
        
        # Сбрасываем состояние кэша
        self._cache_initialized = False
        self._current_mode = None
        self._current_tree_depth = None

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
