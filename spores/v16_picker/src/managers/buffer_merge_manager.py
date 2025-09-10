"""
BufferMergeManager - компонент для мерджа спор по клавише M
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import os
from ..core.spore_graph import SporeGraph


class BufferMergeManager:
    """
    Менеджер для объединения близких спор из призрачного дерева в
    буферный граф.

    Алгоритм:
    1. Создает буферный граф
    2. Копирует споры по порядку: корень → дети → внуки
    3. Для каждой споры проверяет близость в буфере (1.5e-3)
    4. Добавляет спору ИЛИ создает соответствие к существующей
    5. Сохраняет картинку результата
    """

    def __init__(self, distance_threshold: float = 1.5e-3):
        self.distance_threshold = distance_threshold

        # 🔍 Флаг для включения диагностики связей (по умолчанию выключен)
        self._investigate_links = True  # Включить для диагностики
        self._manual_spore_manager_ref = None  # Будет установлен извне

        # Буферный граф для объединенных спор
        self.buffer_graph = SporeGraph('buffer')

        # Двусторонняя карта соответствий
        self.ghost_to_buffer: Dict[str, str] = {}  # ghost_id -> buffer_id
        # buffer_id -> [ghost_ids]
        self.buffer_to_ghosts: Dict[str, List[str]] = {}

        # Статистика
        self.stats = {
            'total_processed': 0,
            'added_to_buffer': 0,
            'merged_to_existing': 0,
            'processing_order': []
        }

        print(f"🔄 BufferMergeManager создан (трешхолд: {distance_threshold})")

    def merge_ghost_tree(self, tree_logic, save_image: bool = True) -> Dict:
        """
        Основной метод: мерджит призрачное дерево в буферный граф.

        Args:
            tree_logic: SporeTree с готовой структурой
            save_image: сохранять ли картинку результата

        Returns:
            dict: статистика мерджа
        """
        print("\n🔄 НАЧАЛО МЕРДЖА ПРИЗРАЧНОГО ДЕРЕВА")

        # Очищаем предыдущие результаты
        self._reset()

        # Проверяем входные данные
        if not self._validate_tree_logic(tree_logic):
            return self._get_error_result("Невалидное призрачное дерево")

        # 🔍 ДИАГНОСТИКА: Исследуем источники связей
        if hasattr(self, '_investigate_links') and self._investigate_links:
            # Получаем manual_spore_manager для исследования prediction_manager
            manual_spore_manager = getattr(self, '_manual_spore_manager_ref', None)
            self._investigate_link_sources(tree_logic, manual_spore_manager)

        try:
            # 1. Обрабатываем корень
            self._process_root(tree_logic)

            # 2. Обрабатываем детей
            self._process_children(tree_logic)

            # 3. Обрабатываем внуков
            self._process_grandchildren(tree_logic)

            # 4. Сохраняем результат
            if save_image:
                image_path = self._save_buffer_image()
                self.stats['image_path'] = image_path

            # 5. Выводим итоговую статистику
            self._print_final_stats()

            return self._get_success_result()

        except Exception as e:
            print(f"❌ Ошибка мерджа: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(str(e))

    def _reset(self):
        """Очищает состояние для нового мерджа."""
        self.buffer_graph.clear()
        self.ghost_to_buffer.clear()
        self.buffer_to_ghosts.clear()
        self.stats = {
            'total_processed': 0,
            'added_to_buffer': 0,
            'merged_to_existing': 0,
            'processing_order': []
        }

    def _validate_tree_logic(self, tree_logic) -> bool:
        """Проверяет что tree_logic готов к обработке."""
        if not tree_logic:
            print("❌ tree_logic is None")
            return False

        if not hasattr(tree_logic, 'root') or not tree_logic.root:
            print("❌ Корень дерева не найден")
            return False

        if (not hasattr(tree_logic, '_children_created') or
                not tree_logic._children_created):
            print("❌ Дети дерева не созданы")
            return False

        return True

    def _investigate_link_sources(self, tree_logic, manual_spore_manager=None):
        """
        ДИАГНОСТИЧЕСКИЙ МЕТОД: Исследует доступные источники информации о связях.
        """
        print("\n🔍 ИССЛЕДОВАНИЕ ИСТОЧНИКОВ СВЯЗЕЙ:")

        # 1. Исследуем tree_logic
        print("\n📊 1. TREE_LOGIC:")
        print(f"   📍 Корень: {tree_logic.root}")
        
        if hasattr(tree_logic, 'children') and tree_logic.children:
            print(f"   👶 Детей: {len(tree_logic.children)}")
            for i, child in enumerate(tree_logic.children[:2]):  # Показываем первых 2
                print(f"      Ребенок {i}: pos={child.get('position', 'нет')}, "
                      f"dt={child.get('dt', 'нет')}, "
                      f"control={child.get('control', 'нет')}")
        
        if hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren:
            print(f"   👶👶 Внуков: {len(tree_logic.grandchildren)}")
            for i, grandchild in enumerate(tree_logic.grandchildren[:2]):  # Показываем первых 2
                print(f"      Внук {i}: pos={grandchild.get('position', 'нет')}, "
                      f"dt={grandchild.get('dt', 'нет')}, control={grandchild.get('control', 'нет')}, "
                      f"parent_idx={grandchild.get('parent_idx', 'нет')}")
        
        # 2. Исследуем prediction_manager
        if manual_spore_manager and hasattr(manual_spore_manager, 'prediction_manager'):
            prediction_manager = manual_spore_manager.prediction_manager
            print(f"\n📊 2. PREDICTION_MANAGER:")
            
            if hasattr(prediction_manager, 'ghost_graph'):
                ghost_graph = prediction_manager.ghost_graph
                print(f"   👻 Ghost graph: {len(ghost_graph.nodes)} узлов, {len(ghost_graph.edges)} связей")
                
                # Показываем несколько связей
                edge_count = 0
                for edge_key, edge_info in ghost_graph.edges.items():
                    if edge_count < 3:  # Показываем первые 3
                        print(f"      Связь {edge_count}: {edge_key} -> type={edge_info.link_type}")
                        edge_count += 1
            
            if hasattr(prediction_manager, 'prediction_links'):
                links = prediction_manager.prediction_links
                print(f"   🔗 Prediction links: {len(links)} линков")
                
                # Показываем несколько линков
                for i, link in enumerate(links[:3]):  # Показываем первые 3
                    parent_id = getattr(link.parent_spore, 'id', 'нет') if hasattr(link, 'parent_spore') else 'нет'
                    child_id = getattr(link.child_spore, 'id', 'нет') if hasattr(link, 'child_spore') else 'нет'
                    color = getattr(link, 'current_color', 'нет') if hasattr(link, 'current_color') else 'нет'
                    print(f"      Линк {i}: {parent_id} -> {child_id}, цвет={color}")
        
        # 3. Рекомендация
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        
        has_tree_structure = (hasattr(tree_logic, 'children') and tree_logic.children and 
                             hasattr(tree_logic, 'grandchildren') and tree_logic.grandchildren)
        
        has_prediction_links = (manual_spore_manager and 
                               hasattr(manual_spore_manager, 'prediction_manager') and
                               hasattr(manual_spore_manager.prediction_manager, 'prediction_links'))
        
        if has_tree_structure:
            print(f"   ✅ tree_logic содержит четкую структуру parent-child")
            print(f"   💡 Рекомендация: использовать tree_logic как основной источник")
        
        if has_prediction_links:
            print(f"   ✅ prediction_manager содержит готовые связи с цветами/направлениями")  
            print(f"   💡 Рекомендация: использовать prediction_manager для типов связей")
        
        if has_tree_structure and has_prediction_links:
            print(f"   🎯 ЛУЧШИЙ ПЛАН: Структура из tree_logic + цвета/типы из prediction_manager")

    def _process_root(self, tree_logic):
        """Обрабатывает корень дерева."""
        print("\n📍 ОБРАБОТКА КОРНЯ")

        root_data = tree_logic.root
        root_position = np.array(root_data['position'])

        # Корень всегда добавляется в буферный граф (первый элемент)
        buffer_id = "buffer_root"
        ghost_id = "ghost_root"

        # Создаем запись в буферном графе (пока без реальной споры)
        self._add_to_buffer_graph(buffer_id, ghost_id, root_position)

        self.stats['total_processed'] += 1
        self.stats['added_to_buffer'] += 1
        self.stats['processing_order'].append(f"root({ghost_id}→{buffer_id})")

        print(f"   ✅ Корень добавлен: {ghost_id} → {buffer_id}")
        print(f"   📍 Позиция: ({root_position[0]:.4f}, {root_position[1]:.4f})")

    def _process_children(self, tree_logic):
        """Обрабатывает детей дерева."""
        print("\n👶 ОБРАБОТКА ДЕТЕЙ")

        if not hasattr(tree_logic, 'children') or not tree_logic.children:
            print("   ⏭️ Дети отсутствуют")
            return

        for i, child_data in enumerate(tree_logic.children):
            child_position = np.array(child_data['position'])
            ghost_id = f"ghost_child_{i}"

            print(f"\n   🔍 Ребенок {i}: {ghost_id}")
            print(f"      📍 Позиция: ({child_position[0]:.4f}, {child_position[1]:.4f})")

            # Ищем ближайшую спору в буферном графе
            closest_buffer_id, min_distance = self._find_closest_in_buffer(child_position)

            if closest_buffer_id and min_distance < self.distance_threshold:
                # Объединяем с существующей
                self._merge_to_existing(ghost_id, closest_buffer_id, min_distance)
                self.stats['merged_to_existing'] += 1
                self.stats['processing_order'].append(
                    f"child_{i}({ghost_id}→{closest_buffer_id}, "
                    f"d={min_distance:.2e})")
            else:
                # Добавляем новую
                buffer_id = f"buffer_child_{i}"
                self._add_to_buffer_graph(buffer_id, ghost_id, child_position)
                self.stats['added_to_buffer'] += 1
                self.stats['processing_order'].append(f"child_{i}({ghost_id}→{buffer_id})")

            self.stats['total_processed'] += 1

    def _process_grandchildren(self, tree_logic):
        """Обрабатывает внуков дерева."""
        print("\n👶👶 ОБРАБОТКА ВНУКОВ")

        if (not hasattr(tree_logic, '_grandchildren_created') or
                not tree_logic._grandchildren_created or
                not hasattr(tree_logic, 'grandchildren') or
                not tree_logic.grandchildren):
            print("   ⏭️ Внуки отсутствуют")
            return

        for i, grandchild_data in enumerate(tree_logic.grandchildren):
            grandchild_position = np.array(grandchild_data['position'])
            ghost_id = f"ghost_grandchild_{i}"

            print(f"\n   🔍 Внук {i}: {ghost_id}")
            print(f"      📍 Позиция: ({grandchild_position[0]:.4f}, "
                  f"{grandchild_position[1]:.4f})")

            # Ищем ближайшую спору в буферном графе
            closest_buffer_id, min_distance = self._find_closest_in_buffer(grandchild_position)

            if closest_buffer_id and min_distance < self.distance_threshold:
                # Объединяем с существующей
                self._merge_to_existing(ghost_id, closest_buffer_id, min_distance)
                self.stats['merged_to_existing'] += 1
                self.stats['processing_order'].append(
                    f"grandchild_{i}({ghost_id}→{closest_buffer_id}, "
                    f"d={min_distance:.2e})")
            else:
                # Добавляем новую
                buffer_id = f"buffer_grandchild_{i}"
                self._add_to_buffer_graph(buffer_id, ghost_id, grandchild_position)
                self.stats['added_to_buffer'] += 1
                self.stats['processing_order'].append(
                    f"grandchild_{i}({ghost_id}→{buffer_id})")

            self.stats['total_processed'] += 1

    def _find_closest_in_buffer(self, position: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Находит ближайшую спору в буферном графе.

        Returns:
            (buffer_id, distance) или (None, float('inf')) если буфер пуст
        """
        if not self.buffer_positions:
            return None, float('inf')

        min_distance = float('inf')
        closest_buffer_id = None

        for buffer_id, buffer_position in self.buffer_positions.items():
            distance = np.linalg.norm(position - buffer_position)
            if distance < min_distance:
                min_distance = distance
                closest_buffer_id = buffer_id

        return closest_buffer_id, min_distance

    def _add_to_buffer_graph(self, buffer_id: str, ghost_id: str, position: np.ndarray):
        """Добавляет новую спору в буферный граф."""
        # Сохраняем позицию для поиска близости
        if not hasattr(self, 'buffer_positions'):
            self.buffer_positions = {}
        self.buffer_positions[buffer_id] = position.copy()

        # Обновляем карты соответствий
        self.ghost_to_buffer[ghost_id] = buffer_id
        if buffer_id not in self.buffer_to_ghosts:
            self.buffer_to_ghosts[buffer_id] = []
        self.buffer_to_ghosts[buffer_id].append(ghost_id)

        print(f"      ✅ Добавлен в буфер: {ghost_id} → {buffer_id}")

    def _merge_to_existing(self, ghost_id: str, buffer_id: str, distance: float):
        """Объединяет призрачную спору с существующей в буфере."""
        # Обновляем карты соответствий
        self.ghost_to_buffer[ghost_id] = buffer_id
        if buffer_id not in self.buffer_to_ghosts:
            self.buffer_to_ghosts[buffer_id] = []
        self.buffer_to_ghosts[buffer_id].append(ghost_id)

        print(f"      🔗 Объединен с существующим: {ghost_id} → {buffer_id}")
        print(f"      📏 Расстояние: {distance:.2e} < "
              f"{self.distance_threshold:.2e}")

    def _save_buffer_image(self) -> str:
        """Сохраняет картинку буферного графа."""
        try:
            # Создаем директорию если нужно
            buffer_dir = "buffer"
            if not os.path.exists(buffer_dir):
                os.makedirs(buffer_dir)

            save_path = os.path.join(buffer_dir, "buffer_merge_result.png")

            # Создаем график
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # Рисуем споры буферного графа
            for buffer_id, position in getattr(self, 'buffer_positions', {}).items():
                ghost_count = len(self.buffer_to_ghosts.get(buffer_id, []))

                # Размер маркера зависит от количества объединенных спор
                marker_size = 50 + 30 * (ghost_count - 1)

                # Цвет: синий для одиночных, красный для объединенных
                color = 'blue' if ghost_count == 1 else 'red'

                ax.scatter(position[0], position[1], s=marker_size, c=color,
                          alpha=0.7, edgecolors='black', linewidth=1)

                # Подпись
                label = f"{buffer_id}\n({ghost_count} спор)"
                ax.annotate(label, (position[0], position[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left')

            ax.set_title(f"Буферный граф после мерджа\n"
                        f"Обработано: {self.stats['total_processed']}, "
                        f"В буфере: {self.stats['added_to_buffer']}, "
                        f"Объединено: {self.stats['merged_to_existing']}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

            # Легенда
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, label='Одиночная спора'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, label='Объединенная спора')
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n📊 Картинка буферного графа сохранена: {save_path}")
            return save_path

        except Exception as e:
            print(f"❌ Ошибка сохранения картинки: {e}")
            return ""

    def _print_final_stats(self):
        """Выводит итоговую статистику мерджа."""
        print("\n📊 ИТОГОВАЯ СТАТИСТИКА МЕРДЖА:")
        print(f"   🔢 Всего обработано спор: {self.stats['total_processed']}")
        print(f"   ➕ Добавлено в буфер: {self.stats['added_to_buffer']}")
        print(f"   🔗 Объединено с существующими: {self.stats['merged_to_existing']}")
        compression_ratio = (self.stats['merged_to_existing'] /
                                max(self.stats['total_processed'], 1))
        print(f"   📉 Коэффициент сжатия: {compression_ratio:.1%}")

        print("\n🗺️ КАРТА СООТВЕТСТВИЙ:")
        for buffer_id, ghost_list in self.buffer_to_ghosts.items():
            if len(ghost_list) > 1:
                print(f"   🔗 {buffer_id} ← {ghost_list}")
            else:
                print(f"   📍 {buffer_id} ← {ghost_list[0]}")

    def _get_success_result(self) -> Dict:
        """Возвращает результат успешного мерджа."""
        return {
            'success': True,
            'stats': self.stats.copy(),
            'ghost_to_buffer': self.ghost_to_buffer.copy(),
            'buffer_to_ghosts': self.buffer_to_ghosts.copy()
        }

    def _get_error_result(self, error_msg: str) -> Dict:
        """Возвращает результат с ошибкой."""
        return {
            'success': False,
            'error': error_msg,
            'stats': self.stats.copy()
        }

    def get_buffer_spore_count(self) -> int:
        """Возвращает количество спор в буферном графе."""
        return len(getattr(self, 'buffer_positions', {}))

    def get_correspondence_map(self) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """Возвращает карты соответствий."""
        return (self.ghost_to_buffer.copy(),
                self.buffer_to_ghosts.copy())