from typing import Optional, List
import numpy as np
from ursina import destroy

from .shared_dependencies import SharedDependencies
from ...managers.spore_manager import SporeManager
from ...core.spore import Spore
from ...visual.link import Link


class TreeCreationManager:
    """
    Управляет созданием деревьев разной глубины.

    Ответственности:
    - Создание деревьев споров в позиции курсора
    - Управление режимами создания (spores/tree)
    - Настройка глубины дерева
    - Интеграция с SporeTree логикой
    """

    def __init__(self, deps: SharedDependencies, spore_manager: SporeManager, manual_spore_manager=None):
        self.deps = deps
        self.spore_manager = spore_manager
        self.manual_spore_manager = manual_spore_manager  # НОВОЕ: прямая ссылка

        # Настройки создания
        self.creation_mode = 'spores'  # 'spores' или 'tree'
        self.tree_depth = 2

        # Сохранение dt вектора от призрачного дерева
        self.ghost_tree_dt_vector = None

        # ДОБАВИТЬ: Отслеживание созданных объектов
        self.created_spores: List[Spore] = []
        self.created_links: List[Link] = []
        self.created_spore_keys: List[str] = []  # Ключи в zoom_manager
        self.created_link_keys: List[str] = []   # Ключи в zoom_manager

        print("   ✓ Tree Creation Manager создан")

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

    def create_tree_at_cursor(self, preview_position_2d: np.ndarray, depth: Optional[int] = None) -> Optional[List[Spore]]:
        """
        Создает дерево указанной глубины.
        
        Args:
            preview_position_2d: Позиция курсора в 2D координатах
            depth: Глубина дерева (1 или 2). Если None, использует self.tree_depth

        Returns:
            Список созданных спор или None при ошибке

        ЛОГИКА: дерево → споры и линки → добавляем в общий граф → забываем дерево
        """
        if depth is None:
            depth = self.tree_depth
        depth = int(depth)  # Убеждаемся что depth - это int
        print("🚨 ВЫЗВАН create_tree_at_cursor()!!! Начинаем создание дерева!")
        try:
            from ...visual.spore_tree_visual import SporeTreeVisual
            from ...logic.tree.spore_tree import SporeTree
            from ...logic.tree.spore_tree_config import SporeTreeConfig

            # Получаем текущий dt
            dt = self._get_current_dt()

            # Правильная позиция для дерева
            tree_position = np.array([preview_position_2d[0], preview_position_2d[1]])

            # 🔍 ДИАГНОСТИКА: Выбор пути создания дерева
            print(f"🔍 ДИАГНОСТИКА выбора пути:")
            print(f"   depth: {depth}")
            print(f"   self.ghost_tree_dt_vector is None: {self.ghost_tree_dt_vector is None}")
            if self.ghost_tree_dt_vector is not None:
                print(f"   self.ghost_tree_dt_vector length: {len(self.ghost_tree_dt_vector)}")
                print(f"   condition check: {self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12}")
            else:
                print(f"   self.ghost_tree_dt_vector: None")

            # Создаем логику дерева с учетом глубины
            if depth == 1:
                # Для глубины 1: используем dt из dt_manager для всех детей
                print(f"🛤️ ПУТЬ: Глубина 1 (только дети)")
                print(f"🌲 Создаем дерево глубины 1 с единым dt")
                
                tree_config = SporeTreeConfig(
                    initial_position=tree_position,
                    dt_base=dt
                )

                tree_logic = SporeTree(
                    pendulum=self.deps.pendulum,
                    config=tree_config,
                    auto_create=False  # Создаем вручную
                )
                
                # Создаем только детей с единым dt
                dt_children_uniform = np.ones(4) * dt
                tree_logic.create_children(dt_children=dt_children_uniform)
                
            elif self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12:
                print(f"🛤️ ПУТЬ: Призрачное дерево (с объединением)")
                
                # Получаем готовое призрачное дерево
                if hasattr(self.manual_spore_manager, '_last_tree_logic') and self.manual_spore_manager._last_tree_logic:
                    print(f"🔄 Используем готовое призрачное дерево (уже объединенное)")
                    tree_logic = self.manual_spore_manager._last_tree_logic
                    
                    # Обновляем позицию корня на позицию курсора
                    tree_logic.root['position'] = tree_position.copy()
                    
                    # Проверяем что дерево уже объединено
                    grandchildren_count = len(tree_logic.grandchildren) if hasattr(tree_logic, 'grandchildren') else 0
                    print(f"📊 Призрачное дерево содержит: {grandchildren_count} внуков")
                    
                    if hasattr(tree_logic, '_grandchildren_modified') and tree_logic._grandchildren_modified:
                        print(f"✅ Призрачное дерево уже объединено - используем как есть")
                    else:
                        print(f"⚠️ Призрачное дерево не объединено - применяем объединение")
                        if depth >= 2 and grandchildren_count > 0:
                            merge_result = tree_logic.merge_close_grandchildren(distance_threshold=2e-3)
                            if merge_result['total_merged'] > 0:
                                print(f"🔗 Дополнительно объединено {merge_result['total_merged']} пар")
                else:
                    print(f"❌ Призрачное дерево не найдено - создаем новое")
                    # Fallback: создаем как раньше
                    dt_children_abs = np.abs(self.ghost_tree_dt_vector[:4])
                    dt_grandchildren_abs = np.abs(self.ghost_tree_dt_vector[4:])

                    print(f"🌲 Используем dt из призрачного дерева")
                    print(f"   📊 dt_children: {dt_children_abs}")
                    print(f"   📊 dt_grandchildren: {dt_grandchildren_abs}")

                    tree_config = SporeTreeConfig(
                        initial_position=tree_position,
                        dt_base=dt
                    )

                    tree_logic = SporeTree(
                        pendulum=self.deps.pendulum,
                        config=tree_config,
                        dt_children=dt_children_abs,
                        dt_grandchildren=dt_grandchildren_abs,
                        auto_create=False
                    )
                    
                    # Создаем детей (всегда)
                    tree_logic.create_children()
                    
                    # Создаем внуков если нужна глубина 2
                    print(f"🔍 ДИАГНОСТИКА создания внуков: depth={depth}, depth >= 2: {depth >= 2}")
                    if depth >= 2:
                        print(f"🔍 Вызываем tree_logic.create_grandchildren()")
                        grandchildren = tree_logic.create_grandchildren(show=False)
                        print(f"🔍 Создано внуков: {len(grandchildren) if grandchildren else 0}")
                        
                        print(f"🔍 Вызываем merge_close_grandchildren с threshold=2e-3")
                        # Применяем объединение для призрачных деревьев
                        merge_result = tree_logic.merge_close_grandchildren(
                            distance_threshold=2e-3  # Увеличенный threshold для оптимизированных пар
                        )
                        print(f"🔍 Результат объединения: {merge_result}")

                        if merge_result['total_merged'] > 0:
                            print(f"🔗 Объединено {merge_result['total_merged']} пар внуков в реальном дереве")
                            print(f"📊 Внуков в реальном дереве: {merge_result['remaining_grandchildren']}")
                        else:
                            print(f"📊 Объединение в реальном дереве не требуется: все внуки далеко")
                    else:
                        print(f"❌ Внуки не создаются: depth={depth} < 2")
            else:
                print(f"🛤️ ПУТЬ: Обычное дерево (fallback)")
                # Создаем обычное дерево ТОЧНО как в превью: явные dt + единый пересчет
                print(f"🌲 Создаем дерево без паринга (dt как в превью)")

                factor = self.deps.config.get('tree', {}).get('dt_grandchildren_factor', 0.05)

                tree_config = SporeTreeConfig(
                    initial_position=tree_position,
                    dt_base=dt
                )

                # Положительные dt-массивы (как ожидает SporeTree.create_*):
                dt_children_abs = np.ones(4, dtype=float) * dt
                dt_grandchildren_abs = np.ones(8, dtype=float) * (dt * factor)

                tree_logic = SporeTree(
                    pendulum=self.deps.pendulum,
                    config=tree_config,
                    auto_create=False
                )

                # Создаем детей/внуков ровно один раз и с явными массивами
                tree_logic.create_children(dt_children=dt_children_abs)
                if depth >= 2:
                    grandchildren = tree_logic.create_grandchildren(dt_grandchildren=dt_grandchildren_abs, show=False)
                    
                    # НОВОЕ: Объединяем близких внуков
                    merge_result = tree_logic.merge_close_grandchildren(
                        distance_threshold=2e-3  # Консистентный threshold с призрачными деревьями
                    )
                    
                    if merge_result['total_merged'] > 0:
                        print(f"🔗 Объединено {merge_result['total_merged']} пар внуков")
                        print(f"📊 Внуков в дереве: {merge_result['remaining_grandchildren']}")
                        
                        # Обновляем статистику для UI
                        tree_logic._merge_stats = merge_result
                    else:
                        print("📊 Внуки не требуют объединения")

                # Синтетический подписанный вектор (как в превью)
                dt_children_signed = np.array([+dt, -dt, +dt, -dt], dtype=float)
                base_gc = dt * factor
                dt_grandchildren_signed = np.array(
                    [ +base_gc, -base_gc,  +base_gc, -base_gc,  +base_gc, -base_gc,  +base_gc, -base_gc ],
                    dtype=float
                )
                synthetic_dt_vector = np.concatenate([dt_children_signed, dt_grandchildren_signed])

                # Принудительно пересчитываем позиции и dt в данных дерева
                self._recalculate_positions_with_new_dt(tree_logic, synthetic_dt_vector, tree_position)

            # 🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пересчитываем позиции с новыми dt
            if self.ghost_tree_dt_vector is not None and len(self.ghost_tree_dt_vector) == 12:
                self._recalculate_positions_with_new_dt(tree_logic, self.ghost_tree_dt_vector, tree_position)

            # 🔍 ПРОВЕРКА ПОЛНОТЫ ДЕРЕВА ПЕРЕД ВИЗУАЛИЗАЦИЕЙ
            print(f"\n🔍 ПРОВЕРКА ПОЛНОТЫ ДЕРЕВА:")
            print(f"   📊 Корень: {tree_logic.root}")
            print(f"   📊 Детей: {len(tree_logic.children)}")
            if hasattr(tree_logic, 'grandchildren'):
                print(f"   📊 Внуков: {len(tree_logic.grandchildren)}")
                # Проверяем объединенные внуки
                merged_count = sum(1 for gc in tree_logic.grandchildren if 'merged_from' in gc)
                print(f"   🔗 Объединенных внуков: {merged_count}")
                if merged_count > 0:
                    print(f"   📍 Детали объединений:")
                    for i, gc in enumerate(tree_logic.grandchildren):
                        if 'merged_from' in gc:
                            print(f"      Внук {i}: объединен из {gc['merged_from']}")
            else:
                print(f"   📊 Внуков: нет")

            # Проверяем согласованность с призрачным деревом
            if hasattr(self.manual_spore_manager, '_last_tree_logic') and self.manual_spore_manager._last_tree_logic == tree_logic:
                print(f"   ✅ Используется то же самое дерево что и в призрачном режиме")
            else:
                print(f"   ⚠️ Используется другое дерево (возможна потеря объединений)")
                
            # Создаем визуализацию дерева
            goal_position = self.deps.config.get('spore', {}).get('goal_position', [0, 0])
            
            # DEBUG: Диагностика конфигурации
            print(f"🔍 DEBUG: Полный self.config['spore']: {self.deps.config.get('spore', 'НЕТ КЛЮЧА!')}")
            spore_config = self.deps.config.get('spore', {})
            print(f"🔍 DEBUG: spore_config: {spore_config}")
            spore_scale = spore_config.get('scale', 'НЕТ SCALE!')
            print(f"🔍 DEBUG: spore scale из конфигурации: {spore_scale}")
            print(f"🔍 DEBUG: zoom_manager.spores_scale: {self.deps.zoom_manager.spores_scale}")

            # ПРОВЕРКА: Если scale отсутствует - это баг!
            if 'scale' not in spore_config:
                print(f"❌ БАГ: scale отсутствует в spore_config! Принудительно устанавливаем 0.02")
                spore_config = spore_config.copy()
                spore_config['scale'] = 0.02

            # ВАЖНО: Передаем весь config, не только spore_config, чтобы включить goal_position
            visual_config = self.deps.config.copy()
            visual_config['spore']['goal_position'] = goal_position

            # Правильный конструктор SporeTreeVisual
            tree_visual = SporeTreeVisual(
                color_manager=self.deps.color_manager,
                zoom_manager=self.deps.zoom_manager,
                config=visual_config,
                id_manager=self.spore_manager.id_manager  # Передаем id_manager
            )

            # Устанавливаем логику дерева
            tree_visual.set_tree_logic(tree_logic)
            
            # Создаем визуализацию
            tree_visual.create_visual()

            # Извлекаем споры и линки из дерева
            created_spores = []
            created_links = []

            # Собираем споры
            if tree_visual.root_spore:
                created_spores.append(tree_visual.root_spore)

            created_spores.extend(tree_visual.child_spores)

            if depth >= 2:
                created_spores.extend(tree_visual.grandchild_spores)

            # ПРАВИЛЬНЫЙ ПОРЯДОК: Сначала добавляем в систему, потом применяем трансформации
            # 1. Добавляем споры в spore_manager
            for spore in created_spores:
                if spore:
                    self.spore_manager.add_spore_manual(spore)

            # 2. Регистрируем споры в zoom_manager (используем уже присвоенные ID)
            spore_keys = []
            for spore in created_spores:
                if spore:
                    key = f"tree_spore_{spore.id}"  # Используем уже присвоенный ID
                    self.deps.zoom_manager.register_object(spore, key)
                    spore_keys.append(key)
                    spore._zoom_manager_key = key

            # ДОБАВИТЬ: Сохраняем для отслеживания
            self.created_spores.extend(created_spores)
            self.created_spore_keys.extend(spore_keys)

            # Собираем линки
            created_links.extend(tree_visual.child_links)

            if depth >= 2:
                created_links.extend(tree_visual.grandchild_links)

            # 3. Регистрируем линки в zoom_manager (используем уже присвоенные ID)
            link_keys = []
            for link in created_links:
                if link:
                    key = f"tree_link_{link.id}"  # Используем уже присвоенный ID
                    self.deps.zoom_manager.register_object(link, key)
                    link_keys.append(key)
                    link._zoom_manager_key = key

            # ДОБАВИТЬ: Сохраняем для отслеживания
            self.created_links.extend(created_links)
            self.created_link_keys.extend(link_keys)

            # 4. Применяем трансформации ко всем объектам сразу
            self.deps.zoom_manager.update_transform()

            # Освобождаем SporeTreeVisual
            tree_visual.root_spore = None
            tree_visual.child_spores.clear()
            tree_visual.grandchild_spores.clear()
            tree_visual.child_links.clear()
            tree_visual.grandchild_links.clear()
            tree_visual.visual_created = False

            tree_visual = None
            
            # НОВОЕ: Сохраняем ссылку на дерево для отладки ПЕРЕД обнулением
            if self.manual_spore_manager:
                self.manual_spore_manager._last_tree_logic = tree_logic
            
            print(f"🌲 Дерево создано в ({preview_position_2d[0]:.3f}, {preview_position_2d[1]:.3f})")
            print(f"   📊 Глубина: {self.tree_depth}, dt: {dt:.4f}")
            print(f"   🎯 Создано: {len(created_spores)} спор + {len(created_links)} линков")
            print(f"💡 Нажмите L для построения графика дерева")

            # 🖱️ АВТОМАТИЧЕСКАЯ ГЕНЕРАЦИЯ ПОЛНОГО РЕАЛЬНОГО ГРАФА ПО ЛКМ
            print("\n🖱️ ГЕНЕРАЦИЯ ПОЛНОГО РЕАЛЬНОГО ГРАФА ПО ЛКМ:")
            try:
                if (hasattr(self.spore_manager, 'graph') and
                        self.spore_manager.graph):
                    real_graph = self.spore_manager.graph
                    
                    # Создаем визуализацию полного реального графа
                    full_graph_path = real_graph.create_debug_visualization(
                        "full_real_graph_lmb")
                    if full_graph_path:
                        print(f"🖼️ Полный реальный граф сохранен: "
                              f"{full_graph_path}")
                        
                        # Выводим полную статистику
                        print("📊 ПОЛНАЯ СТАТИСТИКА ПО ЛКМ:")
                        print(f"   🔴 Всего спор: {len(real_graph.nodes)}")
                        print(f"   🔗 Всего связей: {len(real_graph.edges)}")
                        
                        # Статистика по типам связей
                        link_types = {}
                        for edge_info in real_graph.edges.values():
                            link_type = edge_info.link_type
                            link_types[link_type] = (
                                link_types.get(link_type, 0) + 1)
                        for link_type, count in link_types.items():
                            print(f"   🎨 {link_type}: {count}")
                    else:
                        print("❌ Не удалось создать визуализацию полного графа")
                else:
                    print("❌ Реальный граф недоступен")
                    
            except Exception as e:
                print(f"❌ Ошибка генерации полного графа: {e}")
                import traceback
                traceback.print_exc()

            # 🔍 СОЗДАНИЕ ОТЛАДОЧНОЙ ВИЗУАЛИЗАЦИИ РЕАЛЬНОГО ГРАФА
            print(f"\n🔍 СОЗДАНИЕ ОТЛАДОЧНОЙ ВИЗУАЛИЗАЦИИ РЕАЛЬНОГО ГРАФА:")
            try:
                if hasattr(self.spore_manager, 'graph') and self.spore_manager.graph:
                    real_graph = self.spore_manager.graph
                    
                    # Используем встроенный метод debug из SporeGraph
                    print(f"📋 СТРУКТУРА РЕАЛЬНОГО ГРАФА ПОСЛЕ СОЗДАНИЯ:")
                    real_graph.debug_print()

                    # Создаем простую статистику
                    print(f"\n📊 ПОДРОБНАЯ СТАТИСТИКА РЕАЛЬНОГО ГРАФА:")
                    for node_id, spore_obj in real_graph.nodes.items():
                        if hasattr(spore_obj, 'logic') and hasattr(spore_obj.logic, 'position_2d'):
                            pos_2d = spore_obj.logic.position_2d
                            is_ghost = getattr(spore_obj, 'is_ghost', False)
                            print(f"   • {node_id}: pos=({pos_2d[0]:.4f}, {pos_2d[1]:.4f}), is_ghost={is_ghost}")
                        else:
                            print(f"   • {node_id}: позиция недоступна")

                    print(f"\n📊 СВЯЗИ РЕАЛЬНОГО ГРАФА:")
                    for edge_key, edge_info in real_graph.edges.items():
                        print(f"   {edge_info}")

                    # Сравнение с ожидаемыми значениями
                    expected_spores = 9  # 1 корень + 4 ребенка + 4 объединенных внука  
                    actual_spores = len(created_spores)
                    if actual_spores == expected_spores:
                        print(f"✅ Количество созданных спор корректно: {actual_spores}")
                    else:
                        print(f"⚠️ Неожиданное количество спор: {actual_spores} (ожидалось {expected_spores})")
                        print(f"   Возможная проблема: объединения не учтены при создании")
                    
                else:
                    print("⚠️ Реальный граф недоступен для визуализации")
                    
            except Exception as e:
                print(f"❌ Ошибка создания визуализации реального графа: {e}")
                import traceback
                traceback.print_exc()

            tree_logic = None
            return created_spores

        except Exception as e:
            print(f"❌ Ошибка создания дерева: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_last_created_links(self) -> List:
        """
        Возвращает линки, созданные при последнем вызове create_tree_at_cursor().
        
        Returns:
            Список линков, созданных в последней операции
        """
        # Возвращаем только линки, созданные в последней операции
        # Это будут последние добавленные линки в self.created_links
        
        # Для простоты можем вернуть все текущие created_links,
        # так как обычно создается одно дерево за раз
        return self.created_links.copy()

    def get_last_created_objects(self) -> tuple[List, List]:
        """
        Возвращает споры и линки, созданные при последнем вызове create_tree_at_cursor().
        
        Returns:
            tuple: (список спор, список линков)
        """
        return self.created_spores.copy(), self.created_links.copy()

    def _recalculate_positions_with_new_dt(self, tree_logic, ghost_dt_vector, initial_position):
        """
        🔧 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Пересчитывает позиции всех узлов дерева с новыми dt.
        
        Args:
            tree_logic: SporeTree с новыми dt
            ghost_dt_vector: Вектор из 12 dt (4 детей + 8 внуков)  
            initial_position: Начальная позиция корня дерева
        """
        try:
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
                    child_data['dt'] = new_dt  # ← ВАЖНО: синхронизируем знак/величину dt с превью
                    
                    print(f"      Ребенок {i}: dt={new_dt:+.6f}, control={control:+.6f}, pos → {new_position}")
            
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
                            # Используем pendulum для расчета новой позиции внука от родителя
                            new_position = self.deps.pendulum.step(parent_position, control, new_dt)
                            # Обновляем позицию в данных дерева
                            grandchild_data['position'] = new_position
                            grandchild_data['dt'] = new_dt  # ← ВАЖНО
                            
                            print(f"      Внук {i}: dt={new_dt:+.6f}, control={control:+.6f}, parent_idx={parent_idx}, pos → {new_position}")
            
            print(f"   🔧 ПЕРЕСЧЕТ ПОЗИЦИЙ: Завершен")
            
        except Exception as e:
            print(f"❌ Ошибка при пересчете позиций: {e}")

    def _get_current_dt(self) -> float:
        """Получает текущий dt из конфигурации."""
        return self.deps.config.get('pendulum', {}).get('dt', 0.1)

    def clear_all_created_objects(self) -> None:
        """Очищает все объекты созданные этим менеджером."""
        print(f"🧹 TreeCreationManager: очистка {len(self.created_spores)} спор и {len(self.created_links)} линков")
        
        # 1. Удаляем линки
        for i, link in enumerate(self.created_links):
            try:
                # Дерегистрируем из zoom_manager
                if i < len(self.created_link_keys):
                    self.deps.zoom_manager.unregister_object(self.created_link_keys[i])
                
                # Уничтожаем объект
                destroy(link)
                print(f"   ✓ Удален линк: {self.created_link_keys[i] if i < len(self.created_link_keys) else f'link_{i}'}")
            except Exception as e:
                print(f"   ❌ Ошибка удаления линка {i}: {e}")
        
        # 2. Удаляем споры (НЕ из spore_manager - это сделает сам spore_manager)
        for i, spore in enumerate(self.created_spores):
            try:
                # Только дерегистрируем из zoom_manager
                if i < len(self.created_spore_keys):
                    self.deps.zoom_manager.unregister_object(self.created_spore_keys[i])
                    print(f"   ✓ Дерегистрирована спора: {self.created_spore_keys[i]}")
            except Exception as e:
                print(f"   ❌ Ошибка дерегистрации споры {i}: {e}")
        
        # 3. Очищаем списки отслеживания
        self.created_spores.clear()
        self.created_links.clear()
        self.created_spore_keys.clear()
        self.created_link_keys.clear()
        
        print("   ✓ TreeCreationManager очищен")
