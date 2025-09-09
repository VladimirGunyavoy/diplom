"""
SporeGraph - Центральное хранилище структуры связей между спорами

Этот класс знает:
- Какая спора с какой соединена
- Направление связи (parent_spore -> child_spore)
- Тип связи (цвет: ghost_max, ghost_min, default)
- Граф может быть реальным или призрачным
"""

from typing import Dict, Optional, Set, List, Tuple, Any
from ..core.spore import Spore
from ..visual.link import Link


class EdgeInfo:
    """Информация о ребре графа между двумя спорами"""

    def __init__(self,
                 parent_spore: Spore,
                 child_spore: Spore,
                 link_type: str = 'default',
                 link_object: Optional[Link] = None):
        self.parent_spore = parent_spore
        self.child_spore = child_spore
        self.link_type = link_type
        self.link_object = link_object

    def get_direction_tuple(self) -> Tuple[str, str]:
        """Возвращает кортеж (parent_id, child_id) для идентификации ребра"""
        # 🔧 ИСПРАВЛЕНИЕ: Получаем правильные ID
        parent_id = self._get_spore_id(self.parent_spore)
        child_id = self._get_spore_id(self.child_spore)
        return (parent_id, child_id)
    
    def _get_spore_id(self, spore: Spore) -> str:
        """Получает правильный ID споры, исправляя bound method"""
        spore_id = spore.id
        if hasattr(spore_id, '__call__'):  # Если это bound method
            if hasattr(spore, 'is_ghost') and spore.is_ghost:
                spore_id = f"tree_ghost_root"
            else:
                spore_id = f"spore_{id(spore)}"
        return str(spore_id)

    def __repr__(self):
        parent_id = self._get_spore_id(self.parent_spore)
        child_id = self._get_spore_id(self.child_spore)
        return (f"EdgeInfo({parent_id} -> {child_id}, "
                f"type={self.link_type})")


class SporeGraph:
    """
    Центральный граф связей между спорами.

    Хранит структуру связей отдельно от визуальных объектов Link.
    Позволяет легко копировать структуру между реальным и призрачным графом.
    """

    def __init__(self, graph_type: str = 'real'):
        """
        Args:
            graph_type: 'real' для реального графа, 'ghost' для призрачного
        """
        self.graph_type = graph_type

        # Основное хранилище: от (parent_id, child_id) к EdgeInfo
        self.edges: Dict[Tuple[str, str], EdgeInfo] = {}

        # Индексы для быстрого поиска
        self.nodes: Dict[str, Spore] = {}
        self.outgoing: Dict[str, Set[str]] = {}
        self.incoming: Dict[str, Set[str]] = {}

    def add_spore(self, spore: Spore) -> None:
        """Добавляет спору в граф"""
        if not hasattr(spore, 'id') or spore.id is None:
            raise ValueError(f"Spore должна иметь уникальный id: {spore}")

        # 🔧 ИСПРАВЛЕНИЕ: Проверяем и исправляем bound method ID
        spore_id = spore.id
        if hasattr(spore_id, '__call__'):  # Если это bound method
            # Генерируем правильный ID для призрачных спор
            if hasattr(spore, 'is_ghost') and spore.is_ghost:
                spore_id = f"tree_ghost_root"
            else:
                spore_id = f"spore_{id(spore)}"  # Используем id объекта как fallback

        self.nodes[spore_id] = spore
        if spore_id not in self.outgoing:
            self.outgoing[spore_id] = set()
        if spore_id not in self.incoming:
            self.incoming[spore_id] = set()

    def add_edge(self,
                 parent_spore: Spore,
                 child_spore: Spore,
                 link_type: str = 'default',
                 link_object: Optional[Link] = None) -> EdgeInfo:
        """
        Добавляет ребро в граф

        Args:
            parent_spore: Спора-родитель (начало стрелки)
            child_spore: Спора-ребенок (конец стрелки)
            link_type: Тип связи (ghost_max, ghost_min, default)
            link_object: Опциональная ссылка на визуальный Link

        Returns:
            EdgeInfo: Информация о добавленном ребре
        """
        # Добавляем споры если их нет
        self.add_spore(parent_spore)
        self.add_spore(child_spore)

        edge_info = EdgeInfo(parent_spore, child_spore, link_type, link_object)
        edge_key = edge_info.get_direction_tuple()

        # Если ребро уже существует, обновляем его
        if edge_key in self.edges:
            print(f"⚠️ SporeGraph: Обновляем существующее ребро {edge_key}")

        self.edges[edge_key] = edge_info

        # Обновляем индексы
        parent_id = self._get_spore_id(parent_spore)
        child_id = self._get_spore_id(child_spore)
        self.outgoing[parent_id].add(child_id)
        self.incoming[child_id].add(parent_id)

        return edge_info
    
    def _get_spore_id(self, spore: Spore) -> str:
        """Получает правильный ID споры, исправляя bound method"""
        spore_id = spore.id
        if hasattr(spore_id, '__call__'):  # Если это bound method
            if hasattr(spore, 'is_ghost') and spore.is_ghost:
                spore_id = f"tree_ghost_root"
            else:
                spore_id = f"spore_{id(spore)}"
        return str(spore_id)

    def remove_edge(self, parent_id: str, child_id: str) -> bool:
        """
        Удаляет ребро из графа

        Returns:
            bool: True если ребро было удалено, False если не найдено
        """
        edge_key = (parent_id, child_id)
        if edge_key not in self.edges:
            return False

        del self.edges[edge_key]

        # Обновляем индексы
        if parent_id in self.outgoing:
            self.outgoing[parent_id].discard(child_id)
        if child_id in self.incoming:
            self.incoming[child_id].discard(parent_id)

        return True

    def get_children(self, parent_id: str) -> List[Spore]:
        """Возвращает всех детей данной споры"""
        child_ids = self.outgoing.get(parent_id, set())
        return [self.nodes[child_id] for child_id in child_ids
                if child_id in self.nodes]

    def get_parents(self, child_id: str) -> List[Spore]:
        """Возвращает всех родителей данной споры"""
        parent_ids = self.incoming.get(child_id, set())
        return [self.nodes[parent_id] for parent_id in parent_ids
                if parent_id in self.nodes]

    def get_edge_info(self, parent_id: str,
                      child_id: str) -> Optional[EdgeInfo]:
        """Возвращает информацию о ребре"""
        return self.edges.get((parent_id, child_id))

    def copy_structure_from(self, other_graph: 'SporeGraph',
                           spore_manager=None) -> None:
        """
        Копирует структуру связей из другого графа,
        создавая связи между реальными спорами.

        Args:
            other_graph: Граф-источник (обычно призрачный)
            spore_manager: SporeManager для создания визуальных Link
        """
        print(f"🔄 Копируем структуру из {other_graph.graph_type} "
              f"графа в {self.graph_type}")
        print(f"   📊 Источник: {len(other_graph.edges)} ребер, "
              f"{len(other_graph.nodes)} узлов")

        if not spore_manager or self.graph_type != 'real':
            print("   ⚠️ Копирование возможно только в реальный "
                  "граф с SporeManager")
            return

        created_links = 0
        skipped_links = 0

        for edge_key, edge_info in other_graph.edges.items():
            # Ищем реальные споры, которые соответствуют призрачным
            real_parent = find_real_spore_for_ghost(
                edge_info.parent_spore, spore_manager)
            real_child = find_real_spore_for_ghost(
                edge_info.child_spore, spore_manager)

            if real_parent and real_child:
                # Проверяем что связь еще не существует
                edge_exists = self.get_edge_info(
                    real_parent.id, real_child.id) is not None

                if not edge_exists:
                    try:
                        # Импортируем Link здесь чтобы избежать
                        # циклических импортов
                        from ..visual.link import Link

                        # Создаем визуальный Link между РЕАЛЬНЫМИ спорами
                        visual_link = Link(
                            parent_spore=real_parent,
                            child_spore=real_child,
                            color_manager=spore_manager.color_manager,
                            zoom_manager=spore_manager.zoom_manager,
                            config=spore_manager.config
                        )

                        # Все скопированные связи получают обычный цвет
                        visual_link.color = spore_manager.color_manager.get_color(
                            'link', 'default')

                        # Добавляем связь в граф (между реальными спорами)
                        self.add_edge(
                            parent_spore=real_parent,
                            child_spore=real_child,
                            link_type='default',  # Только обычные связи
                            link_object=visual_link
                        )

                        # Добавляем в SporeManager
                        spore_manager.links.append(visual_link)

                        # Регистрируем в ZoomManager
                        link_id = spore_manager.zoom_manager.get_unique_link_id()
                        spore_manager.zoom_manager.register_object(
                            visual_link, link_id)
                        visual_link._zoom_manager_key = link_id

                        created_links += 1

                    except Exception as e:
                        print(f"   ❌ Ошибка создания Link между "
                              f"{real_parent.id} -> {real_child.id}: {e}")
                        skipped_links += 1
                else:
                    skipped_links += 1
            else:
                print("   ⚠️ Не найдены реальные споры для "
                      "призрачной связи")
                skipped_links += 1

        print(f"   ✅ Скопировано в граф: {len(self.edges)} ребер, "
              f"{len(self.nodes)} узлов")
        print(f"   🔗 Создано визуальных линков: {created_links}")
        print(f"   ⏭️ Пропущено связей: {skipped_links}")

        # Обновляем трансформации всех объектов
        if spore_manager and hasattr(spore_manager, 'zoom_manager'):
            spore_manager.zoom_manager.update_transform()

    def clear(self) -> None:
        """Очищает весь граф"""
        self.edges.clear()
        self.nodes.clear()
        self.outgoing.clear()
        self.incoming.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику графа"""
        link_types = {}
        for edge_info in self.edges.values():
            link_types[edge_info.link_type] = (
                link_types.get(edge_info.link_type, 0) + 1)

        return {
            'graph_type': self.graph_type,
            'nodes_count': len(self.nodes),
            'edges_count': len(self.edges),
            'link_types': link_types
        }

    def debug_print(self) -> None:
        """Выводит отладочную информацию о графе"""
        stats = self.get_stats()
        print(f"📊 {self.graph_type.upper()} ГРАФ:")
        print(f"   🔴 Узлов (спор): {stats['nodes_count']}")
        print(f"   🔗 Ребер (связей): {stats['edges_count']}")
        print(f"   🎨 Типы связей: {stats['link_types']}")

    def create_debug_visualization(self, filename_prefix="graph_debug"):
        """
        Создает отладочную визуализацию графа с matplotlib.
        
        Args:
            filename_prefix: Префикс имени файла
            
        Returns:
            str: Путь к созданному файлу
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            
            # Создаем папку buffer если не существует
            buffer_dir = "buffer"
            os.makedirs(buffer_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Собираем информацию о узлах
            nodes_info = []
            for spore_id, spore in self.nodes.items():
                try:
                    if hasattr(spore, 'calc_2d_pos'):
                        pos_2d = spore.calc_2d_pos()
                        is_ghost = getattr(spore, 'is_ghost', False)
                        nodes_info.append({
                            'id': spore_id,
                            'pos': pos_2d,
                            'is_ghost': is_ghost,
                            'spore': spore
                        })
                    else:
                        print(f"   ⚠️ Узел {spore_id} не имеет calc_2d_pos")
                except Exception as e:
                    print(f"   ❌ Ошибка обработки узла {spore_id}: {e}")
            
            print(f"🔍 АНАЛИЗ {self.graph_type.upper()} ГРАФА:")
            print(f"   📊 Узлов в графе: {len(self.nodes)}")
            print(f"   📊 Узлов с позициями: {len(nodes_info)}")
            print(f"   📊 Ребер в графе: {len(self.edges)}")
            
            # Рисуем узлы
            for node in nodes_info:
                pos = node['pos']
                color = 'red' if node['is_ghost'] else 'blue'
                alpha = 0.7 if node['is_ghost'] else 1.0
                
                ax.scatter(pos[0], pos[1], c=color, s=100, alpha=alpha)
                ax.annotate(f"{node['id']}", (pos[0], pos[1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Рисуем ребра
            edge_count_by_type = {}
            for edge_key, edge_info in self.edges.items():
                try:
                    parent_pos = edge_info.parent_spore.calc_2d_pos()
                    child_pos = edge_info.child_spore.calc_2d_pos()
                    
                    # Цвет по типу связи (БЕЗ active!)
                    if edge_info.link_type == 'ghost_max':
                        color = 'green'
                    elif edge_info.link_type == 'ghost_min':
                        color = 'orange'
                    else:
                        color = 'blue'  # Все остальные = обычные связи
                    
                    # Рисуем стрелку
                    ax.annotate('', xy=child_pos, xytext=parent_pos,
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
                    
                    # Статистика типов
                    edge_count_by_type[edge_info.link_type] = edge_count_by_type.get(edge_info.link_type, 0) + 1
                    
                except Exception as e:
                    print(f"   ❌ Ошибка рисования ребра {edge_key}: {e}")
            
            # Настройка графика
            ax.set_title(f'{self.graph_type.upper()} ГРАФ - Отладочная визуализация')
            ax.set_xlabel('θ (угол, рад)')
            ax.set_ylabel('θ̇ (скорость, рад/с)')
            ax.grid(True, alpha=0.3)
            
            # Легенда (БЕЗ active!)
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Реальные узлы'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, alpha=0.7, label='Призрачные узлы'),
                plt.Line2D([0], [0], color='green', linewidth=2, label='ghost_max связи'),
                plt.Line2D([0], [0], color='orange', linewidth=2, label='ghost_min связи'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='обычные связи')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Сохраняем файл с фиксированным именем
            filename = f"{filename_prefix}_{self.graph_type}.png"
            filepath = os.path.join(buffer_dir, filename)
            
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 СТАТИСТИКА ТИПОВ СВЯЗЕЙ:")
            for link_type, count in edge_count_by_type.items():
                print(f"   🎨 {link_type}: {count}")
            
            print(f"💾 График {self.graph_type} графа сохранен: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Ошибка создания визуализации: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_graph_structure(self):
        """
        Выводит структуру графа в консоль в читаемом виде.
        """
        print(f"\n" + "="*60)
        print(f"📊 СТРУКТУРА {self.graph_type.upper()} ГРАФА")
        print("="*60)
        
        # Выводим узлы
        print(f"🔴 УЗЛЫ ({len(self.nodes)}):")
        for spore_id, spore in self.nodes.items():
            try:
                if hasattr(spore, 'calc_2d_pos'):
                    pos = spore.calc_2d_pos()
                    is_ghost = getattr(spore, 'is_ghost', 'N/A')
                    print(f"   • {spore_id}: pos=({pos[0]:.4f}, {pos[1]:.4f}), is_ghost={is_ghost}")
                else:
                    print(f"   • {spore_id}: НЕТ ПОЗИЦИИ")
            except Exception as e:
                print(f"   • {spore_id}: ОШИБКА - {e}")
        
        # Выводим связи
        print(f"\n🔗 СВЯЗИ ({len(self.edges)}):")
        if not self.edges:
            print("   (нет связей)")
        else:
            for i, (edge_key, edge_info) in enumerate(self.edges.items(), 1):
                try:
                    parent_id = edge_key[0]
                    child_id = edge_key[1]
                    link_type = edge_info.link_type
                    
                    # Позиции
                    if hasattr(edge_info.parent_spore, 'calc_2d_pos'):
                        parent_pos = edge_info.parent_spore.calc_2d_pos()
                        parent_pos_str = f"({parent_pos[0]:.4f}, {parent_pos[1]:.4f})"
                    else:
                        parent_pos_str = "(нет позиции)"
                        
                    if hasattr(edge_info.child_spore, 'calc_2d_pos'):
                        child_pos = edge_info.child_spore.calc_2d_pos()
                        child_pos_str = f"({child_pos[0]:.4f}, {child_pos[1]:.4f})"
                    else:
                        child_pos_str = "(нет позиции)"
                    
                    print(f"   {i:2d}. {parent_id} → {child_id} [тип: {link_type}]")
                    print(f"       от: {parent_pos_str} к: {child_pos_str}")
                    
                except Exception as e:
                    print(f"   {i:2d}. ОШИБКА СВЯЗИ {edge_key}: {e}")
        
        print("="*60)


def find_real_spore_for_ghost(ghost_spore, spore_manager, tolerance=1e-6):
    """
    Находит реальную спору, которая соответствует призрачной по позиции.

    Args:
        ghost_spore: Призрачная спора
        spore_manager: SporeManager с реальными спорами
        tolerance: Допустимая погрешность в позиции

    Returns:
        Реальная спора или None
    """
    if not ghost_spore or not hasattr(ghost_spore, 'calc_2d_pos'):
        return None

    try:
        ghost_pos = ghost_spore.calc_2d_pos()

        # Ищем среди реальных спор
        for real_spore in spore_manager.objects:
            if (hasattr(real_spore, 'calc_2d_pos') and
                    not getattr(real_spore, 'is_ghost', False)):
                real_pos = real_spore.calc_2d_pos()

                # Вычисляем расстояние между позициями
                distance = ((ghost_pos[0] - real_pos[0])**2 +
                           (ghost_pos[1] - real_pos[1])**2)**0.5

                if distance < tolerance:
                    return real_spore

        return None

    except Exception as e:
        print(f"   ❌ Ошибка поиска реальной споры: {e}")
        return None
