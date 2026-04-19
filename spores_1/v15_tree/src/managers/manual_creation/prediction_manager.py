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
                    # ✅ Направление строго по знаку dt:
                    #   dt > 0  → preview → ghost (forward)
                    #   dt < 0  → ghost → preview (backward)
                    if config['dt'] > 0:
                        parent_spore = preview_spore
                        child_spore = prediction_viz.ghost_spore
                    else:
                        parent_spore = prediction_viz.ghost_spore
                        child_spore = preview_spore

                    prediction_link = Link(
                        parent_spore=parent_spore,
                        child_spore=child_spore,
                        color_manager=self.deps.color_manager,
                        zoom_manager=self.deps.zoom_manager,
                        config=self.deps.config
                    )

                    # ✅ Цвет только по знаку управления
                    if config['control'] > 0:
                        link_color_name = 'ghost_max'  # Красный для max
                    else:
                        link_color_name = 'ghost_min'  # Синий для min

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
            factor = self.deps.config.get('tree', {}).get('dt_grandchildren_factor', 0.05)
            tree_config = SporeTreeConfig(
                initial_position=preview_position_2d.copy(),
                dt_base=dt,
                dt_grandchildren_factor=factor,
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

        # Создаем призрачные линки от корня к детям
        for i, child_ghost in enumerate(child_ghosts):
            if child_ghost and i < len(tree_logic.children):
                child_data = tree_logic.children[i]
                
                # ✅ Цвет только по знаку управления
                link_color = 'ghost_max' if child_data['control'] > 0 else 'ghost_min'

                # ✅ Направление строго по знаку dt:
                #   dt > 0  → root(preview) → child
                #   dt < 0  → child → root(preview)
                if child_data['dt'] > 0:
                    parent_spore = preview_spore
                    child_link_spore = child_ghost
                else:
                    parent_spore = child_ghost
                    child_link_spore = preview_spore

                self._create_ghost_link(
                    parent_spore,
                    child_link_spore,
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
                        # ✅ Цвет по знаку управления внука
                        link_color = 'ghost_max' if grandchild_data['control'] > 0 else 'ghost_min'

                        # ✅ Направление по знаку dt:
                        #   dt > 0  → child → grandchild
                        #   dt < 0  → grandchild → child
                        if grandchild_data['dt'] > 0:
                            parent_spore = child_ghosts[parent_idx]
                            child_link_spore = grandchild_ghost
                        else:
                            parent_spore = grandchild_ghost
                            child_link_spore = child_ghosts[parent_idx]

                        self._create_ghost_link(
                            parent_spore,
                            child_link_spore,
                            f"child_{parent_idx}_to_grandchild_{i}",
                            link_color
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

            # ✅ Цвет с альфой из colors.json
            ghost_link.color = self.deps.color_manager.get_color('link', color_name)
            # убрать принудительную установку alpha
            # (если очень нужно — делайте это через colors.json или config)

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
