"""
BufferMergeManager - компонент для мерджа спор по клавише M
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
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
    
    @staticmethod
    def safe_str_replace(value, old, new):
        """Безопасная замена с конверсией в строку."""
        return str(value).replace(old, new)

    def __init__(self, distance_threshold: float = 1.5e-3):
        self.distance_threshold = distance_threshold

        # 🔍 Флаг для включения диагностики связей (по умолчанию выключен)
        self._investigate_links = True  # Включить для диагностики
        self._manual_spore_manager_ref = None  # Будет установлен извне

        # Буферный граф для объединенных спор
        self.buffer_graph = SporeGraph('buffer')

        # 🔗 Буферный список связей для мерджа
        # [{'parent_id': str, 'child_id': str, 'link_type': str}]
        self.buffer_links: List[Dict] = []

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
        
        # 🔍 ДИАГНОСТИКА: Проверяем состояние буфера до очистки
        if hasattr(self, 'buffer_positions') and self.buffer_positions:
            print(f"⚠️ ВНИМАНИЕ: В буфере найдены старые данные ({len(self.buffer_positions)} позиций)")
            print(f"   Будет выполнена полная очистка...")

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

            # 4. 🔗 НОВОЕ: Обрабатываем связи
            self._process_links(tree_logic)

            # 5. Сохраняем результат
            if save_image:
                image_path = self._save_buffer_image()
                self.stats['image_path'] = image_path

            # 6. 💾 Экспортируем буферный граф в JSON
            export_path = self.export_buffer_graph()
            if export_path:
                self.stats['export_path'] = export_path

            # 7. Выводим итоговую статистику
            self._print_final_stats()

            return self._get_success_result()

        except Exception as e:
            print(f"❌ Ошибка мерджа: {e}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(str(e))

    def _reset(self):
        """Очищает состояние для нового мерджа."""
        print(f"🧹 Очистка буферного графа перед новым мерджем...")
        
        # Очищаем основные структуры
        self.buffer_graph.clear()
        self.ghost_to_buffer.clear()
        self.buffer_to_ghosts.clear()
        if hasattr(self, 'buffer_links'):
            self.buffer_links.clear()
        
        # 🔧 ИСПРАВЛЕНИЕ: Очищаем buffer_positions
        if hasattr(self, 'buffer_positions'):
            old_count = len(self.buffer_positions)
            self.buffer_positions.clear()
            if old_count > 0:
                print(f"   🗑️ Удалено {old_count} старых позиций из буфера")
        else:
            self.buffer_positions = {}
        
        # Сбрасываем статистику
        self.stats = {
            'total_processed': 0,
            'added_to_buffer': 0,
            'merged_to_existing': 0,
            'processing_order': [],
            'total_links': 0,
            'merged_links': 0
        }
        
        print(f"   ✅ Буферный граф полностью очищен")

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

    def _process_links(self, tree_logic):
        """Обрабатывает связи между спорами дерева."""
        print(f"\n🔗 ОБРАБОТКА СВЯЗЕЙ")
        
        # 1. Создаем связи корень ↔ дети
        self._create_root_child_links(tree_logic)
        
        # 2. Создаем связи дети ↔ внуки
        self._create_child_grandchild_links(tree_logic)
        
        # 3. Выводим итоговую статистику связей
        self._print_links_stats()

    def _create_root_child_links(self, tree_logic):
        """Создает связи между корнем и детьми."""
        if not hasattr(tree_logic, 'children') or not tree_logic.children:
            return
            
        print(f"\n   🔗 СВЯЗИ КОРЕНЬ ↔ ДЕТИ:")
        root_buffer_id = "buffer_root"
        
        for i, child_data in enumerate(tree_logic.children):
            child_ghost_id = f"ghost_child_{i}"
            child_buffer_id = self.ghost_to_buffer.get(child_ghost_id)
            
            if not child_buffer_id:
                print(f"      ❌ Не найден buffer_id для {child_ghost_id}")
                continue
                
            # Определяем тип связи по control
            control = child_data.get('control', 0)
            link_type = 'buffer_max' if control > 0 else 'buffer_min'
            
            # Определяем направление по dt
            dt = child_data.get('dt', 0)
            if dt > 0:
                # dt > 0: корень → ребенок
                parent_id, child_id = root_buffer_id, child_buffer_id
                direction = "→"
            else:
                # dt < 0: ребенок → корень  
                parent_id, child_id = child_buffer_id, root_buffer_id
                direction = "←"
            
            # Создаем связь
            link = {
                'parent_id': parent_id,
                'child_id': child_id, 
                'link_type': link_type,
                'source_info': f"root-child_{i}(dt={dt:.3f},u={control})"
            }
            self.buffer_links.append(link)
            self.stats['total_links'] += 1
            
            print(f"      ✅ {parent_id} {direction} {child_buffer_id} ({link_type})")

    def _create_child_grandchild_links(self, tree_logic):
        """Создает связи между детьми и внуками.""" 
        if not hasattr(tree_logic, 'grandchildren') or not tree_logic.grandchildren:
            return
            
        print(f"\n   🔗 СВЯЗИ ДЕТИ ↔ ВНУКИ:")
        
        for i, grandchild_data in enumerate(tree_logic.grandchildren):
            grandchild_ghost_id = f"ghost_grandchild_{i}"
            grandchild_buffer_id = self.ghost_to_buffer.get(grandchild_ghost_id)
            
            if not grandchild_buffer_id:
                print(f"      ❌ Не найден buffer_id для {grandchild_ghost_id}")
                continue
                
            # Получаем parent_idx - номер ребенка-родителя
            parent_idx = grandchild_data.get('parent_idx')
            if parent_idx is None:
                print(f"      ❌ Нет parent_idx для внука {i}")
                continue
                
            # Находим buffer_id родителя-ребенка
            parent_ghost_id = f"ghost_child_{parent_idx}"
            parent_buffer_id = self.ghost_to_ghosts.get(parent_ghost_id) if False else self.ghost_to_buffer.get(parent_ghost_id)
            
            if not parent_buffer_id:
                print(f"      ❌ Не найден buffer_id для родителя {parent_ghost_id}")
                continue
            
            # Определяем тип связи по control
            control = grandchild_data.get('control', 0)
            link_type = 'buffer_max' if control > 0 else 'buffer_min'
            
            # Определяем направление по dt
            dt = grandchild_data.get('dt', 0)
            if dt > 0:
                # dt > 0: ребенок → внук
                parent_id, child_id = parent_buffer_id, grandchild_buffer_id
                direction = "→"
            else:
                # dt < 0: внук → ребенок
                parent_id, child_id = grandchild_buffer_id, parent_buffer_id
                direction = "←"
            
            # Проверяем на дублирование связи (может быть при объединении)
            existing_link = self._find_existing_link(parent_id, child_id, link_type)
            if existing_link:
                print(f"      🔗 Связь уже существует: {parent_id} {direction} {child_id} ({link_type})")
                self.stats['merged_links'] += 1
                continue
            
            # Создаем связь
            link = {
                'parent_id': parent_id,
                'child_id': child_id,
                'link_type': link_type,
                'source_info': f"child_{parent_idx}-grandchild_{i}(dt={dt:.3f},u={control})"
            }
            self.buffer_links.append(link)
            self.stats['total_links'] += 1
            
            print(f"      ✅ {parent_buffer_id} {direction} {grandchild_buffer_id} ({link_type})")

    def _find_existing_link(self, parent_id: str, child_id: str, link_type: str) -> bool:
        """Проверяет существует ли уже такая связь."""
        for link in self.buffer_links:
            if (link['parent_id'] == parent_id and 
                link['child_id'] == child_id and 
                link['link_type'] == link_type):
                return True
        return False

    def _print_links_stats(self):
        """Выводит статистику связей."""
        print(f"\n📊 СТАТИСТИКА СВЯЗЕЙ:")
        print(f"   🔗 Всего связей создано: {len(self.buffer_links)}")
        print(f"   🔗 Объединено дублирующихся: {self.stats['merged_links']}")
        
        # Группируем по типам
        link_types = {}
        for link in self.buffer_links:
            link_type = link['link_type']
            link_types[link_type] = link_types.get(link_type, 0) + 1
        
        print(f"   🎨 По типам: {link_types}")
        
        # Показываем несколько примеров связей
        print(f"\n📝 ПРИМЕРЫ СВЯЗЕЙ:")
        for i, link in enumerate(self.buffer_links[:4]):  # Показываем первые 4
            print(f"   {i+1}. {link['parent_id']} → {link['child_id']} ({link['link_type']})")

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
        
        # 🔍 ОТЛАДКА: Проверяем что позиция действительно сохранена
        if buffer_id in self.buffer_positions:
            saved_pos = self.buffer_positions[buffer_id]
            print(f"         📍 Позиция сохранена: ({saved_pos[0]:.4f}, {saved_pos[1]:.4f})")

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
        """Сохраняет картинку буферного графа со связями."""
        try:
            # Создаем директорию если нужно
            buffer_dir = "buffer"
            if not os.path.exists(buffer_dir):
                os.makedirs(buffer_dir)

            save_path = os.path.join(buffer_dir, "buffer_merge_latest.png")

            # Создаем график
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))

            # 1. Рисуем связи первыми (под спорами)
            self._draw_buffer_links(ax)

            # 2. Рисуем споры поверх
            self._draw_buffer_spores(ax)

            # 3. Настройки графика
            ax.set_title(
                f"Буферный граф после мерджа\n"
                f"Споры: {len(getattr(self, 'buffer_positions', {}))}, "
                f"Связи: {len(getattr(self, 'buffer_links', []))}, "
                f"Объединено: {self.stats['merged_to_existing']}"
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

            # 4. Легенда
            self._add_legend(ax)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\n📊 Картинка буферного графа сохранена: {save_path}")
            return save_path

        except Exception as e:
            print(f"❌ Ошибка сохранения картинки: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _draw_buffer_spores(self, ax):
        """Рисует споры буферного графа."""
        for buffer_id, position in getattr(self, 'buffer_positions', {}).items():
            ghost_count = len(self.buffer_to_ghosts.get(buffer_id, []))

            # Размер маркера зависит от количества объединенных спор
            marker_size = 80 + 40 * (ghost_count - 1)

            # Цвет: синий для одиночных, красный для объединенных
            color = 'lightblue' if ghost_count == 1 else 'lightcoral'
            edge_color = 'blue' if ghost_count == 1 else 'red'

            ax.scatter(
                position[0], position[1], s=marker_size, c=color,
                alpha=0.8, edgecolors=edge_color, linewidth=2
            )

            # Подпись
            label = f"{self.safe_str_replace(buffer_id, 'buffer_', '')}\n({ghost_count})"  # 🔧 ИСПРАВЛЕНИЕ: Используем безопасную замену
            ax.annotate(
                label, (position[0], position[1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, ha='left', weight='bold'
            )

    def _draw_buffer_links(self, ax):
        """Рисует связи буферного графа как стрелки."""
        if not hasattr(self, 'buffer_positions'):
            return

        # Цвета для разных типов связей
        link_colors = {
            'buffer_max': 'red',      # u_max - красный
            'buffer_min': 'blue'      # u_min - синий
        }

        for link in getattr(self, 'buffer_links', []):
            parent_id = link['parent_id']
            child_id = link['child_id']
            link_type = link['link_type']

            # Получаем позиции
            parent_pos = self.buffer_positions.get(parent_id)
            child_pos = self.buffer_positions.get(child_id)

            if parent_pos is None or child_pos is None:
                continue

            # Цвет стрелки
            color = link_colors.get(link_type, 'gray')

            # Рисуем стрелку от parent к child
            dx = child_pos[0] - parent_pos[0]
            dy = child_pos[1] - parent_pos[1]

            # Смещаем начало и конец стрелки чтобы они не перекрывали споры
            length = np.sqrt(dx * dx + dy * dy)
            if length > 0:
                offset = 0.015  # Отступ от центра споры
                start_offset = offset / length
                end_offset = offset / length

                start_x = parent_pos[0] + dx * start_offset
                start_y = parent_pos[1] + dy * start_offset
                arrow_dx = dx * (1 - 2 * end_offset)
                arrow_dy = dy * (1 - 2 * end_offset)

                # Настраиваем ширину стрелки в зависимости от длины
                # Минимальная ширина 0.5, максимальная 3.0
                # Ширина = 1/10 от длины, но в разумных пределах
                arrow_width = max(0.5, min(3.0, length * 0.1))
                head_width = max(0.004, min(0.012, length * 0.05))
                head_length = max(0.004, min(0.012, length * 0.05))

                ax.arrow(
                    start_x, start_y, arrow_dx, arrow_dy,
                    head_width=head_width, head_length=head_length,
                    fc=color, ec=color, alpha=0.7, linewidth=arrow_width
                )

    def _add_legend(self, ax):
        """Добавляет легенду к графику."""
        from matplotlib.lines import Line2D

        legend_elements = [
            # Споры
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markeredgecolor='blue', markersize=10, label='Одиночная спора'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                   markeredgecolor='red', markersize=12, label='Объединенная спора'),

            # Связи
            Line2D([0], [0], color='red', linewidth=3, label='buffer_max (u_max)'),
            Line2D([0], [0], color='blue', linewidth=3, label='buffer_min (u_min)')
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def export_buffer_graph(self, filename: str = None) -> str:
        """
        Экспортирует буферный граф в JSON файл.
        
        Args:
            filename: имя файла (если None, генерируется автоматически)
            
        Returns:
            путь к сохраненному файлу
        """
        try:
            # Создаем директорию если нужно
            buffer_dir = "buffer"
            if not os.path.exists(buffer_dir):
                os.makedirs(buffer_dir)

            # Генерируем имя файла если не задано
            if filename is None:
                filename = "buffer_graph_latest.json"

            save_path = os.path.join(buffer_dir, filename)

            # Формируем структуру данных для экспорта
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'algorithm_version': 'BufferMergeManager_v1.0',
                    'distance_threshold': self.distance_threshold
                },
                
                'stats': self.stats.copy(),
                
                'spores': self._export_spores_data(),
                
                'links': self._export_links_data(),
                
                'correspondence_maps': {
                    'ghost_to_buffer': self.ghost_to_buffer.copy(),
                    'buffer_to_ghosts': self.buffer_to_ghosts.copy()
                }
            }
            
            # Сохраняем в JSON
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Буферный граф экспортирован: {save_path}")
            print(f"   📊 Спор: {len(export_data['spores'])}")
            print(f"   🔗 Связей: {len(export_data['links'])}")
            
            return save_path
            
        except Exception as e:
            print(f"❌ Ошибка экспорта буферного графа: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _export_spores_data(self) -> List[Dict]:
        """Формирует данные спор для экспорта."""
        spores_data = []
        
        for buffer_id, position in getattr(self, 'buffer_positions', {}).items():
            ghost_list = self.buffer_to_ghosts.get(buffer_id, [])
            
            spore_data = {
                'buffer_id': buffer_id,
                'position': [float(position[0]), float(position[1])],
                'merged_ghosts': ghost_list.copy(),
                'ghost_count': len(ghost_list),
                'is_merged': len(ghost_list) > 1
            }
            
            spores_data.append(spore_data)
        
        return spores_data

    def _export_links_data(self) -> List[Dict]:
        """Формирует данные связей для экспорта."""
        links_data = []
        
        for link in getattr(self, 'buffer_links', []):
            link_data = {
                'parent_id': link['parent_id'],
                'child_id': link['child_id'],
                'link_type': link['link_type'],
                'source_info': link.get('source_info', 'unknown')
            }
            
            links_data.append(link_data)
        
        return links_data

    def _print_final_stats(self):
        """Выводит итоговую статистику мерджа."""
        print("\n📊 ИТОГОВАЯ СТАТИСТИКА МЕРДЖА:")
        print(f"   🔢 Всего обработано спор: {self.stats['total_processed']}")
        print(f"   ➕ Добавлено в буфер: {self.stats['added_to_buffer']}")
        print(f"   🔗 Объединено с существующими: {self.stats['merged_to_existing']}")
        compression_ratio = (self.stats['merged_to_existing'] /
                                max(self.stats['total_processed'], 1))
        print(f"   📉 Коэффициент сжатия: {compression_ratio:.1%}")
        print(f"   🔗 Связей создано: {len(getattr(self, 'buffer_links', []))}")
        if self.stats.get('merged_links', 0) > 0:
            print(f"   🔗 Связей объединено: {self.stats['merged_links']}")

        # Информация о сохраненных файлах
        if 'image_path' in self.stats:
            print(f"   🖼️ Визуализация: {self.stats['image_path']}")
        if 'export_path' in self.stats:
            print(f"   💾 JSON экспорт: {self.stats['export_path']}")

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

    def materialize_buffer_to_real(self, spore_manager, zoom_manager, color_manager, pendulum, config) -> Dict:
        """
        Материализует буферный граф в реальные споры и связи.
        
        Args:
            spore_manager: SporeManager для создания реальных спор и связей
            zoom_manager: ZoomManager для регистрации объектов
            color_manager: ColorManager для цветов
            pendulum: PendulumSystem для физики
            config: конфиг для настроек спор
            
        Returns:
            dict: результат материализации
        """
        print(f"\n🎨 МАТЕРИАЛИЗАЦИЯ БУФЕРНОГО ГРАФА В РЕАЛЬНЫЙ")
        
        if not self.buffer_positions:
            return self._get_error_result("Буферный граф пуст - нечего материализовать")
            
        try:
            # Статистика материализации
            materialize_stats = {
                'spores_created': 0,
                'links_created': 0,
                'errors': []
            }
            
            # 1. Создаем реальные споры
            real_spores_map = self._create_real_spores(
                spore_manager, zoom_manager, color_manager, pendulum, config, materialize_stats)
            
            # 2. Создаем реальные связи
            self._create_real_links(spore_manager, real_spores_map, materialize_stats, zoom_manager, config)
            
            # 3. Создаем визуализацию реального графа  
            visualization_path = self._create_real_graph_visualization(spore_manager)
            if visualization_path:
                materialize_stats['visualization_path'] = visualization_path
            
            # 4. Выводим статистику
            self._print_materialize_stats(materialize_stats)
            
            # 5. Обновляем трансформации
            zoom_manager.update_transform()
            
            return {
                'success': True,
                'stats': materialize_stats
            }
            
        except Exception as e:
            error_msg = f"Ошибка материализации: {e}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return self._get_error_result(error_msg)

    def _create_real_spores(self, spore_manager, zoom_manager, color_manager, pendulum, config, stats) -> Dict[str, any]:
        """Создает реальные споры из буферного графа."""
        print(f"\n   🌟 СОЗДАНИЕ РЕАЛЬНЫХ СПОР:")
        
        # Импортируем Spore
        from ..core.spore import Spore
        
        real_spores_map = {}  # buffer_id -> real_spore
        spore_config = config.get('spore', {})
        goal_position = spore_config.get('goal_position', [0, 0])
        
        for buffer_id, position in self.buffer_positions.items():
            try:
                # Позиция в 3D (Y=0)
                position_3d = (float(position[0]), 0, float(position[1]))
                
                # Определяем является ли спора целевой
                is_goal = buffer_id == "buffer_root"
                
                # Создаем реальную спору
                real_spore = Spore(
                    pendulum=pendulum,
                    dt=spore_config.get('dt', 0.05),
                    scale=spore_config.get('scale', 0.1),
                    position=position_3d,
                    goal_position=goal_position,
                    is_goal=is_goal,
                    color_manager=color_manager,
                    config=spore_config
                )
                
                # Устанавливаем уникальный ID
                real_spore.id = f"real_{buffer_id}"
                
                # Добавляем в SporeManager
                spore_manager.add_spore_manual(real_spore)
                
                # Регистрируем в ZoomManager
                zoom_key = f"real_{buffer_id}"
                zoom_manager.register_object(real_spore, zoom_key)
                real_spore._zoom_manager_key = zoom_key  # Сохраняем для удаления
                
                # Сохраняем в карте
                real_spores_map[buffer_id] = real_spore
                stats['spores_created'] += 1
                
                # Информация об объединенных спорах
                ghost_count = len(self.buffer_to_ghosts.get(buffer_id, []))
                merge_info = f" (объединяет {ghost_count})" if ghost_count > 1 else ""
                
                print(f"      ✅ {buffer_id} → real_spore{merge_info}")
                
            except Exception as e:
                error_msg = f"Ошибка создания споры {buffer_id}: {e}"
                print(f"      ❌ {error_msg}")
                stats['errors'].append(error_msg)
        
        return real_spores_map

    def _create_real_links(self, spore_manager, real_spores_map, stats, zoom_manager, config):
        """Создает реальные связи из буферных связей."""
        print(f"\n   🔗 СОЗДАНИЕ РЕАЛЬНЫХ СВЯЗЕЙ:")
        
        # 🔍 ОТЛАДКА ТИПОВ ДАННЫХ
        print(f"   🔍 ОТЛАДКА: Проверяем типы данных в buffer_links:")
        for i, link in enumerate(self.buffer_links):
            print(f"      Link {i}: link_type type={type(link['link_type'])}, value={link['link_type']}")
            if i >= 3:  # Показываем только первые 3
                break
        
        # Импортируем Link
        from ..visual.link import Link
        
        # Цвета для разных типов связей  
        link_colors = {
            'buffer_max': 'ghost_max',    # Красный для u_max
            'buffer_min': 'ghost_min'     # Синий для u_min
        }
        
        for link in self.buffer_links:
            try:
                parent_buffer_id = link['parent_id']
                child_buffer_id = link['child_id']
                link_type = str(link['link_type'])  # 🔧 ИСПРАВЛЕНИЕ: Принудительно конвертируем в строку
                
                # Получаем реальные споры
                parent_spore = real_spores_map.get(parent_buffer_id)
                child_spore = real_spores_map.get(child_buffer_id)
                
                if not parent_spore or not child_spore:
                    error_msg = f"Не найдены споры для связи {parent_buffer_id} → {child_buffer_id}"
                    print(f"      ❌ {error_msg}")
                    stats['errors'].append(error_msg)
                    continue
                
                # Определяем цвет связи
                color_key = link_colors.get(link_type, 'link_default')
                
                # Создаем визуальную связь
                visual_link = Link(
                    parent_spore=parent_spore,
                    child_spore=child_spore,
                    zoom_manager=zoom_manager,
                    color_manager=spore_manager.color_manager,
                    config=spore_manager.config
                )
                
                # Устанавливаем цвет после создания
                visual_link.color = spore_manager.color_manager.get_color('link', color_key)
                
                # Устанавливаем ID
                visual_link.id = f"real_link_{parent_buffer_id}_to_{child_buffer_id}"
                
                # Добавляем в SporeManager (правильная последовательность)
                spore_manager.links.append(visual_link)
                
                # Регистрируем в ZoomManager
                link_id = zoom_manager.get_unique_link_id()
                zoom_manager.register_object(visual_link, link_id)
                visual_link._zoom_manager_key = link_id  # Сохраняем для удаления
                
                # Добавляем связь в граф
                spore_manager.graph.add_edge(
                    parent_spore=parent_spore,
                    child_spore=child_spore,
                    link_type=self.safe_str_replace(link_type, 'buffer_', 'real_'),  # buffer_max → real_max
                    link_object=visual_link
                )
                
                stats['links_created'] += 1
                
                print(f"      ✅ {parent_buffer_id} → {child_buffer_id} ({link_type})")
                
            except Exception as e:
                error_msg = f"Ошибка создания связи {link.get('source_info', 'unknown')}: {e}"
                print(f"      ❌ {error_msg}")
                stats['errors'].append(error_msg)

    def _create_real_graph_visualization(self, spore_manager) -> str:
        """Создает визуализацию реального графа."""
        try:
            print(f"\n   🖼️ СОЗДАНИЕ ВИЗУАЛИЗАЦИИ РЕАЛЬНОГО ГРАФА:")
            
            # Создаем директорию если нужно
            buffer_dir = "buffer"
            if not os.path.exists(buffer_dir):
                os.makedirs(buffer_dir)
            
            save_path = os.path.join(buffer_dir, "real_graph_result.png")
            
            # Создаем график
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))
            
            # Собираем данные о реальных спорах
            real_spores = spore_manager.objects
            real_links = spore_manager.links
            
            # 1. Рисуем связи первыми
            self._draw_real_links(ax, real_links)
            
            # 2. Рисуем споры поверх
            self._draw_real_spores(ax, real_spores)
            
            # 3. Настройки графика
            ax.set_title(f"Реальный граф после материализации\n"
                        f"Споры: {len(real_spores)}, "
                        f"Связи: {len(real_links)}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y") 
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            # 4. Легенда для реального графа
            self._add_real_graph_legend(ax)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"      ✅ Визуализация реального графа: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"      ❌ Ошибка визуализации: {e}")
            return ""

    def _draw_real_spores(self, ax, real_spores):
        """Рисует реальные споры."""
        for spore in real_spores:
            if hasattr(spore, 'calc_2d_pos'):
                pos = spore.calc_2d_pos()
                
                # Цвет: зеленый для реальных спор
                color = 'lightgreen'
                edge_color = 'darkgreen'
                
                # Размер зависит от типа
                marker_size = 120 if getattr(spore, 'is_goal', False) else 80
                
                ax.scatter(pos[0], pos[1], s=marker_size, c=color,
                          alpha=0.8, edgecolors=edge_color, linewidth=2)
                
                # Подпись
                spore_id = getattr(spore, 'id', 'unknown')
                label = self.safe_str_replace(spore_id, 'real_buffer_', '')  # 🔧 ИСПРАВЛЕНИЕ: Используем безопасную замену
                ax.annotate(label, (pos[0], pos[1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, ha='left', weight='bold')

    def _draw_real_links(self, ax, real_links):
        """Рисует реальные связи."""
        # Цвета для разных типов связей
        link_colors = {
            'real_max': 'red',      # u_max - красный
            'real_min': 'blue'      # u_min - синий  
        }
        
        for link in real_links:
            try:
                # Получаем позиции спор
                parent_pos = link.parent_spore.calc_2d_pos() if hasattr(link.parent_spore, 'calc_2d_pos') else None
                child_pos = link.child_spore.calc_2d_pos() if hasattr(link.child_spore, 'calc_2d_pos') else None
                
                if parent_pos is None or child_pos is None:
                    continue
                
                # Определяем цвет по типу связи (из реального графа)
                link_type = 'real_max'  # По умолчанию
                if hasattr(link, 'color_key'):
                    if 'min' in str(link.color_key):
                        link_type = 'real_min'
                
                color = link_colors.get(link_type, 'gray')
                
                # Рисуем стрелку
                dx = child_pos[0] - parent_pos[0]
                dy = child_pos[1] - parent_pos[1]
                
                # Смещение от центра спор
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0:
                    offset = 0.02
                    start_offset = offset / length
                    end_offset = offset / length
                    
                    start_x = parent_pos[0] + dx * start_offset
                    start_y = parent_pos[1] + dy * start_offset
                    arrow_dx = dx * (1 - 2*end_offset)
                    arrow_dy = dy * (1 - 2*end_offset)
                    
                    ax.arrow(start_x, start_y, arrow_dx, arrow_dy,
                            head_width=0.01, head_length=0.01,
                            fc=color, ec=color, alpha=0.8, linewidth=2)
                            
            except Exception as e:
                print(f"Ошибка отрисовки связи: {e}")

    def _add_real_graph_legend(self, ax):
        """Добавляет легенду для реального графа."""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            # Споры
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markeredgecolor='darkgreen', markersize=10, label='Реальная спора'),
            
            # Связи
            Line2D([0], [0], color='red', linewidth=3, label='real_max (u_max)'),
            Line2D([0], [0], color='blue', linewidth=3, label='real_min (u_min)')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def _print_materialize_stats(self, stats):
        """Выводит статистику материализации."""
        print(f"\n📊 СТАТИСТИКА МАТЕРИАЛИЗАЦИИ:")
        print(f"   🌟 Создано спор: {stats['spores_created']}")
        print(f"   🔗 Создано связей: {stats['links_created']}")
        
        if stats['errors']:
            print(f"   ❌ Ошибок: {len(stats['errors'])}")
            for error in stats['errors'][:3]:  # Показываем первые 3
                print(f"      • {error}")
        else:
            print(f"   ✅ Ошибок нет")
            
        if 'visualization_path' in stats:
            print(f"   🖼️ Визуализация: {stats['visualization_path']}")

    def has_buffer_data(self) -> bool:
        """Проверяет есть ли данные в буферном графе."""
        return bool(getattr(self, 'buffer_positions', {}))