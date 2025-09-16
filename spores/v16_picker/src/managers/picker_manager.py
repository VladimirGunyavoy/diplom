"""
Picker Manager - Отслеживание близких спор к точке взгляда
==========================================================

Класс для постоянного отслеживания спор, которые находятся
вблизи текущей точки взгляда камеры.

Функциональность:
- Подписка на изменения look_point от ZoomManager
- Поиск спор в реальном графе на расстоянии < 0.05
- Обновление списка близких спор
- Логирование изменений в консоль
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..managers.zoom_manager import ZoomManager
from ..managers.spore_manager import SporeManager
from ..core.spore import Spore


class PickerManager:
    """
    Менеджер для отслеживания спор, близких к точке взгляда камеры.
    
    Ответственности:
    - Подписка на изменения look_point от ZoomManager
    - Поиск близких спор в реальном графе SporeManager
    - Обновление списка близких спор
    - Логирование изменений
    """
    
    def __init__(self,
                 zoom_manager: ZoomManager,
                 spore_manager: SporeManager,
                 distance_threshold: float = 0.05,
                 verbose_output: bool = True):
        """
        Инициализация PickerManager.

        Args:
            zoom_manager: Менеджер зума для получения look_point
            spore_manager: Менеджер спор для доступа к реальному графу
            distance_threshold: Пороговое расстояние для определения близости
            verbose_output: Выводить ли информацию при каждом изменении
        """
        self.zoom_manager: ZoomManager = zoom_manager
        self.spore_manager: SporeManager = spore_manager
        self.distance_threshold: float = distance_threshold
        self.verbose_output: bool = verbose_output
        
        # Список близких спор
        self.close_spores: List[Dict[str, Any]] = []
        
        # Предыдущие координаты look_point для проверки изменений
        self.last_look_point: Optional[Tuple[float, float]] = None
        
        # Подписка на изменения look_point
        self._subscribe_to_look_point_changes()
        
        print(f"🎯 PickerManager инициализирован "
              f"(threshold: {distance_threshold}, verbose: {verbose_output})")

    def _subscribe_to_look_point_changes(self) -> None:
        """Подписывается на изменения look_point в ZoomManager."""
        if hasattr(self.zoom_manager, 'subscribe_look_point_change'):
            self.zoom_manager.subscribe_look_point_change(
                self._on_look_point_changed)
            print("🎯 PickerManager подписан на изменения look_point")
        else:
            print("⚠️ ZoomManager не поддерживает подписку на look_point")

    def _get_corrected_look_point(self) -> Tuple[float, float]:
        """
        Возвращает исправленный look point с учетом зума.
        Использует ту же формулу коррекции, что и в UI manager.
        
        Returns:
            Кортеж (x, z) исправленных координат
        """
        # Получаем сырую точку взгляда
        raw_x, raw_z = self.zoom_manager.identify_invariant_point()
        
        # Получаем позицию origin_cube с проверкой существования
        try:
            origin_x = (self.zoom_manager.scene_setup.frame
                        .origin_cube.position.x)
            origin_z = (self.zoom_manager.scene_setup.frame
                        .origin_cube.position.z)
        except Exception:
            origin_x, origin_z = 0.0, 0.0
        
        # Применяем формулу коррекции
        scale = self.zoom_manager.a_transformation
        corrected_x = (raw_x - origin_x) / scale
        corrected_z = (raw_z - origin_z) / scale
        
        return corrected_x, corrected_z

    def _correct_look_point(self, raw_x: float, raw_z: float) -> Tuple[float, float]:
        """
        Исправляет сырые координаты look_point с учетом зума.
        НЕ вызывает identify_invariant_point() чтобы избежать рекурсии.
        
        Args:
            raw_x: Сырая X координата
            raw_z: Сырая Z координата
            
        Returns:
            Кортеж (x, z) исправленных координат
        """
        # Получаем позицию origin_cube с проверкой существования
        try:
            origin_x = (self.zoom_manager.scene_setup.frame
                        .origin_cube.position.x)
            origin_z = (self.zoom_manager.scene_setup.frame
                        .origin_cube.position.z)
        except Exception:
            origin_x, origin_z = 0.0, 0.0
        
        # Применяем формулу коррекции
        scale = self.zoom_manager.a_transformation
        corrected_x = (raw_x - origin_x) / scale
        corrected_z = (raw_z - origin_z) / scale
        
        return corrected_x, corrected_z

    def _on_look_point_changed(self, look_point_x: float,
                               look_point_z: float) -> None:
        """
        Колбэк, вызываемый при изменении look_point.

        Args:
            look_point_x: X координата точки взгляда (сырая)
            look_point_z: Z координата точки взгляда (сырая)
        """
        # Исправляем координаты прямо здесь, не вызывая identify_invariant_point
        corrected_x, corrected_z = self._correct_look_point(
            look_point_x, look_point_z)
        
        # Проверяем, действительно ли изменился look_point (по исправленным)
        current_look_point = (corrected_x, corrected_z)
        
        # Если это первый вызов или координаты изменились
        if (self.last_look_point is None or
                abs(self.last_look_point[0] - corrected_x) > 1e-6 or
                abs(self.last_look_point[1] - corrected_z) > 1e-6):
            
            # Обновляем предыдущие координаты
            self.last_look_point = current_look_point
            
            # Вызываем обновление только при реальном изменении
            self._update_close_spores(corrected_x, corrected_z)

    def _update_close_spores(self, look_point_x: float,
                             look_point_z: float) -> None:
        """
        Проверяет близкие споры и обновляет список.

        Args:
            look_point_x: X координата точки взгляда
            look_point_z: Z координата точки взгляда
        """
        new_close_spores = []
        far_spores = []
        
        # Проходим по всем спорам в реальном графе
        for spore_id, spore in self.spore_manager.graph.nodes.items():
            try:
                # Получаем реальные координаты споры
                spore_pos = spore.calc_2d_pos()
                
                # Вычисляем расстояние до точки взгляда
                distance = np.sqrt(
                    (spore_pos[0] - look_point_x)**2 +
                    (spore_pos[1] - look_point_z)**2
                )

                spore_info = {
                    'id': spore_id,
                    'position': spore_pos,
                    'distance': distance,
                    'spore': spore
                }

                # Если спора близко - добавляем в список близких
                if distance < self.distance_threshold:
                    new_close_spores.append(spore_info)
                else:
                    far_spores.append(spore_info)

            except Exception:
                # Пропускаем споры с ошибками
                continue
        
        # ВЫВОДИМ ТОЛЬКО САМУЮ БЛИЗКУЮ СПОРУ
        if self.verbose_output:
            print(f"\n🎯 LOOK_POINT: ({look_point_x:.4f}, {look_point_z:.4f})")
            
            # Объединяем все споры и сортируем по дистанции
            all_spores = new_close_spores + far_spores
            all_spores.sort(key=lambda spore: spore['distance'])
            
            print(f"   📊 Всего спор в графе: {len(all_spores)}")
            
            # Показываем только самую близкую спору
            if all_spores:
                closest_spore = all_spores[0]
                pos = closest_spore['position']
                dist = closest_spore['distance']
                is_close = dist < self.distance_threshold
                marker = "📍" if is_close else "📏"
                
                # Получаем визуальный ID для самой близкой споры
                visual_id = self._get_visual_spore_id(closest_spore['spore'])
                
                print(f"   🎯 САМАЯ БЛИЗКАЯ СПОРА: {marker} {visual_id}: "
                      f"({pos[0]:.4f}, {pos[1]:.4f}), dist={dist:.4f}")
                
                # Выводим соседей самой близкой споры
                self._print_closest_spore_neighbors(closest_spore)
            else:
                print("   📭 Спор в графе нет")

        # Обновляем список близких спор
        self.close_spores = new_close_spores
    
    def get_closest_spore(self) -> Optional[Dict[str, Any]]:
        """
        Возвращает самую близкую спору к текущему look_point.
        
        Returns:
            Словарь с информацией о самой близкой споре или None если спор нет
        """
        corrected_x, corrected_z = self._get_corrected_look_point()
        
        closest_spore = None
        min_distance = float('inf')
        
        for spore_id, spore in self.spore_manager.graph.nodes.items():
            try:
                spore_pos = spore.calc_2d_pos()
                distance = np.sqrt(
                    (spore_pos[0] - corrected_x)**2 +
                    (spore_pos[1] - corrected_z)**2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_spore = {
                        'id': spore_id,
                        'position': spore_pos,
                        'distance': distance,
                        'spore': spore
                    }
                    
            except Exception:
                continue
        
        return closest_spore

    def _get_visual_spore_id(self, spore: Spore) -> str:
        """
        Получает визуальный ID споры (как на картинке) - индекс в списке real_spores + 1.
        
        Args:
            spore: Объект споры
            
        Returns:
            Строковый ID для отображения
        """
        try:
            # Используем ТОЧНО тот же алгоритм, что и на картинке
            # Получаем весь список объектов (как на картинке)
            real_spores = self.spore_manager.objects
            
            # Находим индекс споры в списке (как на картинке)
            spore_index = next((i for i, s in enumerate(real_spores) if s is spore), -1)
            
            if spore_index >= 0:
                return str(spore_index + 1)  # Нумерация от 1 (как на картинке)
            else:
                return "?"
                
        except Exception:
            return "?"

    def _print_closest_spore_neighbors(self, closest_spore: Dict[str, Any]) -> None:
        """
        Выводит информацию о соседях самой близкой споры по графу.
        
        Args:
            closest_spore: Словарь с информацией о самой близкой споре
        """
        if not closest_spore:
            return
            
        spore = closest_spore['spore']
        
        # Получаем правильный ID для графа
        graph_spore_id = self.spore_manager.graph._get_spore_id(spore)
        
        # Получаем визуальный ID (как на картинке) - индекс в списке + 1
        visual_id = self._get_visual_spore_id(spore)
        
        print(f"\n🔗 СОСЕДИ СПОРЫ {visual_id}:")
        
        # Получаем соседей на расстоянии 1 (прямые связи)
        neighbors_1 = self._get_neighbors_at_distance(graph_spore_id, 1)
        if neighbors_1:
            print("   📍 Маршруты длиной 1:")
            for neighbor_info in neighbors_1:
                self._print_neighbor_info(neighbor_info, 1)
        
        # Получаем соседей на расстоянии 2 (через промежуточную спору)
        neighbors_2 = self._get_neighbors_at_distance(graph_spore_id, 2)
        if neighbors_2:
            print("   📍 Маршруты длиной 2:")
            for neighbor_info in neighbors_2:
                self._print_neighbor_info(neighbor_info, 2)
        
        if not neighbors_1 and not neighbors_2:
            print("   📭 Нет соседей в графе")

    def _get_neighbors_at_distance(self, spore_id: str, distance: int) -> List[Dict[str, Any]]:
        """
        Получает соседей споры на заданном расстоянии по графу.
        
        Args:
            spore_id: ID споры
            distance: Расстояние (1 или 2)
            
        Returns:
            Список информации о соседях
        """
        if distance == 1:
            return self._get_direct_neighbors(spore_id)
        elif distance == 2:
            return self._get_neighbors_at_distance_2(spore_id)
        else:
            return []

    def _get_direct_neighbors(self, spore_id: str) -> List[Dict[str, Any]]:
        """
        Получает прямых соседей споры (расстояние 1) с учетом направления времени.
        
        Логика:
        - Если связь A → B с dt > 0, то из A можно добраться в B с прямым временем
        - Если связь A → B с dt < 0, то из B можно добраться в A с обратным временем
        - Для поиска соседей споры X учитываем обе возможности
        """
        neighbors = []
        
        # Получаем детей (исходящие связи) - можем добраться с прямым временем
        children = self.spore_manager.graph.get_children(spore_id)
        
        for child in children:
            child_id = self.spore_manager.graph._get_spore_id(child)
            edge_info = self.spore_manager.graph.get_edge_info(
                spore_id, child_id)
            
            neighbor_info = {
                'target_spore': child,
                'target_id': child_id,
                'path': [spore_id, child_id],
                'edges': [edge_info] if edge_info else [],
                'time_direction': 'forward',  # Прямое время
                'can_reach': True
            }
            neighbors.append(neighbor_info)
        
        # Получаем родителей (входящие связи) - можем добраться с обратным временем
        parents = self.spore_manager.graph.get_parents(spore_id)
        
        for parent in parents:
            parent_id = self.spore_manager.graph._get_spore_id(parent)
            edge_info = self.spore_manager.graph.get_edge_info(
                parent_id, spore_id)
            
            neighbor_info = {
                'target_spore': parent,
                'target_id': parent_id,
                'path': [parent_id, spore_id],
                'edges': [edge_info] if edge_info else [],
                'time_direction': 'backward',  # Обратное время
                'can_reach': True
            }
            neighbors.append(neighbor_info)
        
        return neighbors

    def _get_neighbors_at_distance_2(self, spore_id: str) -> List[Dict[str, Any]]:
        """
        Получает соседей споры на расстоянии 2 с учетом направления времени.
        """
        neighbors = []
        visited = {spore_id}  # Избегаем циклов
        
        # Получаем прямых соседей
        direct_neighbors = self._get_direct_neighbors(spore_id)
        
        for direct_neighbor in direct_neighbors:
            intermediate_id = direct_neighbor['target_id']
            if intermediate_id in visited:
                continue
                
            visited.add(intermediate_id)
            
            # Получаем соседей промежуточной споры
            intermediate_neighbors = self._get_direct_neighbors(intermediate_id)
            
            for neighbor in intermediate_neighbors:
                target_id = neighbor['target_id']
                if target_id in visited:
                    continue
                    
                # Создаем путь длиной 2
                path = [spore_id, intermediate_id, target_id]
                edges = direct_neighbor['edges'] + neighbor['edges']
                
                # Определяем общее направление времени для маршрута длиной 2
                # Если оба шага в одном направлении - сохраняем направление
                # Если в разных - считаем смешанным
                first_direction = direct_neighbor.get('time_direction', 'unknown')
                second_direction = neighbor.get('time_direction', 'unknown')
                
                if first_direction == second_direction:
                    combined_direction = first_direction
                else:
                    combined_direction = 'mixed'
                
                neighbor_info = {
                    'target_spore': neighbor['target_spore'],
                    'target_id': target_id,
                    'path': path,
                    'edges': edges,
                    'intermediate_spore': direct_neighbor['target_spore'],
                    'intermediate_id': intermediate_id,
                    'time_direction': combined_direction,
                    'can_reach': True
                }
                neighbors.append(neighbor_info)
        
        return neighbors

    def _print_neighbor_info(self, neighbor_info: Dict[str, Any], distance: int) -> None:
        """
        Выводит информацию о соседе с учетом направления времени.
        
        Args:
            neighbor_info: Информация о соседе
            distance: Расстояние до соседа
        """
        target_spore = neighbor_info['target_spore']
        
        # Получаем визуальный ID для соседа
        visual_target_id = self._get_visual_spore_id(target_spore)
        
        try:
            target_pos = target_spore.calc_2d_pos()
            pos_str = f"({target_pos[0]:.4f}, {target_pos[1]:.4f})"
        except Exception:
            pos_str = "(нет позиции)"
        
        # Получаем направление времени
        time_direction = neighbor_info.get('time_direction', 'unknown')
        time_arrow = ("⏩" if time_direction == 'forward'
                      else "⏪" if time_direction == 'backward'
                      else "❓")
        
        if distance == 1:
            # Маршрут длиной 1: Спора -> Линк -> Спора
            print(f"      🎯 Спора: {visual_target_id} {pos_str} {time_arrow}")
            
            # Выводим информацию о связи
            for i, edge_info in enumerate(neighbor_info['edges']):
                if edge_info and edge_info.link_object:
                    try:
                        # Получаем информацию о времени и управлении
                        parent_spore = edge_info.parent_spore
                        
                        # Получаем dt из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'dt_value'):
                            dt = edge_info.link_object.dt_value
                            dt_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            dt = getattr(parent_spore.logic, 'dt', 'N/A')
                            dt_source = "spore"
                        else:
                            dt = 'N/A'
                            dt_source = "none"
                        
                        # Получаем управление из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'control_value'):
                            control = edge_info.link_object.control_value
                            control_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            control = getattr(parent_spore.logic, 'optimal_control', 'N/A')
                            control_source = "spore"
                        else:
                            control = 'N/A'
                            control_source = "none"
                        
                        # Обрабатываем массив управления
                        if isinstance(control, np.ndarray):
                            control = control[0] if len(control) > 0 else 'N/A'
                        
                        # Форматируем время со знаком
                        if dt != 'N/A':
                            dt_str = f"+{dt}" if dt >= 0 else str(dt)
                        else:
                            dt_str = 'N/A'
                        
                        # Показываем направление времени
                        direction_text = ("прямое время" if time_direction == 'forward'
                                          else "обратное время" if time_direction == 'backward'
                                          else "неизвестно")
                            
                        print(f"         🔗 Линк: управление={control}, время={dt_str} "
                              f"({direction_text}, источник dt: {dt_source}, источник control: {control_source})")
                    except Exception as e:
                        print(f"         🔗 Линк: ошибка получения данных - {e}")
        
        elif distance == 2:
            # Маршрут длиной 2: Линк -> Линк -> Спора
            time_direction = neighbor_info.get('time_direction', 'unknown')
            time_arrow = ("⏩" if time_direction == 'forward'
                          else "⏪" if time_direction == 'backward'
                          else "🔄" if time_direction == 'mixed'
                          else "❓")
            
            print(f"      🎯 Спора: {visual_target_id} {pos_str} {time_arrow}")
            
            # Показываем направление времени для маршрута длиной 2
            if time_direction == 'mixed':
                direction_text = "смешанное время"
            elif time_direction == 'forward':
                direction_text = "прямое время"
            elif time_direction == 'backward':
                direction_text = "обратное время"
            else:
                direction_text = "неизвестно"
            
            print(f"         📍 Маршрут длиной 2 ({direction_text})")
            
            # Выводим информацию о первой связи
            if len(neighbor_info['edges']) > 0:
                edge_info = neighbor_info['edges'][0]
                if edge_info and edge_info.link_object:
                    try:
                        # Получаем информацию о времени и управлении
                        parent_spore = edge_info.parent_spore
                        
                        # Получаем dt из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'dt_value'):
                            dt = edge_info.link_object.dt_value
                            dt_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            dt = getattr(parent_spore.logic, 'dt', 'N/A')
                            dt_source = "spore"
                        else:
                            dt = 'N/A'
                            dt_source = "none"
                        
                        # Получаем управление из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'control_value'):
                            control = edge_info.link_object.control_value
                            control_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            control = getattr(parent_spore.logic, 'optimal_control', 'N/A')
                            control_source = "spore"
                        else:
                            control = 'N/A'
                            control_source = "none"
                        
                        # Обрабатываем массив управления
                        if isinstance(control, np.ndarray):
                            control = control[0] if len(control) > 0 else 'N/A'
                        
                        # Форматируем время со знаком
                        if dt != 'N/A':
                            dt_str = f"+{dt}" if dt >= 0 else str(dt)
                        else:
                            dt_str = 'N/A'
                            
                        print(f"         🔗 Линк 1: управление={control}, время={dt_str} "
                              f"(источник dt: {dt_source}, источник control: {control_source})")
                    except Exception as e:
                        print(f"         🔗 Линк 1: ошибка получения данных - {e}")
            
            # Выводим информацию о второй связи
            if len(neighbor_info['edges']) > 1:
                edge_info = neighbor_info['edges'][1]
                if edge_info and edge_info.link_object:
                    try:
                        # Получаем информацию о времени и управлении
                        parent_spore = edge_info.parent_spore
                        
                        # Получаем dt из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'dt_value'):
                            dt = edge_info.link_object.dt_value
                            dt_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            dt = getattr(parent_spore.logic, 'dt', 'N/A')
                            dt_source = "spore"
                        else:
                            dt = 'N/A'
                            dt_source = "none"
                        
                        # Получаем управление из линка (если есть) или из родительской споры
                        if hasattr(edge_info.link_object, 'control_value'):
                            control = edge_info.link_object.control_value
                            control_source = "link"
                        elif hasattr(parent_spore, 'logic'):
                            control = getattr(parent_spore.logic, 'optimal_control', 'N/A')
                            control_source = "spore"
                        else:
                            control = 'N/A'
                            control_source = "none"
                        
                        # Обрабатываем массив управления
                        if isinstance(control, np.ndarray):
                            control = control[0] if len(control) > 0 else 'N/A'
                        
                        # Форматируем время со знаком
                        if dt != 'N/A':
                            dt_str = f"+{dt}" if dt >= 0 else str(dt)
                        else:
                            dt_str = 'N/A'
                            
                        print(f"         🔗 Линк 2: управление={control}, время={dt_str} "
                              f"(источник dt: {dt_source}, источник control: {control_source})")
                    except Exception as e:
                        print(f"         🔗 Линк 2: ошибка получения данных - {e}")

    def get_close_spores(self) -> List[Dict[str, Any]]:
        """
        Возвращает текущий список близких спор.
        
        Returns:
            Список словарей с информацией о близких спорах
        """
        return self.close_spores.copy()
    
    def get_close_spores_count(self) -> int:
        """
        Возвращает количество близких спор.
        
        Returns:
            Количество спор в радиусе
        """
        return len(self.close_spores)
    
    def set_distance_threshold(self, threshold: float) -> None:
        """
        Устанавливает новый порог расстояния.

        Args:
            threshold: Новый порог расстояния
        """
        self.distance_threshold = threshold
        print(f"🎯 Порог расстояния изменен на {threshold}")

        # Принудительно обновляем список с новым порогом
        corrected_x, corrected_z = self._get_corrected_look_point()
        self._update_close_spores(corrected_x, corrected_z)

    def force_update(self) -> None:
        """Принудительно обновляет список близких спор."""
        corrected_x, corrected_z = self._get_corrected_look_point()
        self._update_close_spores(corrected_x, corrected_z)
        print("🎯 Принудительное обновление близких спор выполнено")

    def force_update_without_check(self) -> None:
        """Принудительно обновляет список близких спор без проверки."""
        corrected_x, corrected_z = self._get_corrected_look_point()
        
        # Обновляем предыдущие координаты
        self.last_look_point = (corrected_x, corrected_z)
        
        # Вызываем обновление
        self._update_close_spores(corrected_x, corrected_z)
        print("🎯 Принудительное обновление без проверки изменений выполнено")

    def get_last_look_point(self) -> Optional[Tuple[float, float]]:
        """
        Возвращает последние координаты look_point.
        
        Returns:
            Кортеж (x, z) последних координат или None если еще не было
        """
        return self.last_look_point

    def has_look_point_changed(self, new_x: float, new_z: float) -> bool:
        """
        Проверяет, изменился ли look_point по сравнению с последним известным.
        
        Args:
            new_x: Новая X координата
            new_z: Новая Z координата
            
        Returns:
            True если координаты изменились, False если остались теми же
        """
        if self.last_look_point is None:
            return True
            
        return (abs(self.last_look_point[0] - new_x) > 1e-6 or
                abs(self.last_look_point[1] - new_z) > 1e-6)

    def get_all_spores_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Возвращает информацию о всех спорах, разделенных на близкие и не близкие.

        Returns:
            Словарь с ключами 'close' и 'far', содержащий списки спор
        """
        corrected_x, corrected_z = self._get_corrected_look_point()
        
        close_spores = []
        far_spores = []
        
        for spore_id, spore in self.spore_manager.graph.nodes.items():
            try:
                spore_pos = spore.calc_2d_pos()
                distance = np.sqrt(
                    (spore_pos[0] - corrected_x)**2 +
                    (spore_pos[1] - corrected_z)**2
                )
                
                spore_info = {
                    'id': spore_id,
                    'position': spore_pos,
                    'distance': distance,
                    'spore': spore
                }
                
                if distance < self.distance_threshold:
                    close_spores.append(spore_info)
                else:
                    far_spores.append(spore_info)
                    
            except Exception:
                continue
        
        return {
            'close': close_spores,
            'far': far_spores
        }

    def print_all_spores_summary(self) -> None:
        """Выводит сводку по всем спорам, отсортированным по дистанции."""
        spores_info = self.get_all_spores_info()
        close_spores = spores_info['close']
        far_spores = spores_info['far']
        
        # Объединяем и сортируем все споры
        all_spores = close_spores + far_spores
        all_spores.sort(key=lambda spore: spore['distance'])
        
        print("\n🎯 СВОДКА ПО СПОРАМ:")
        print(f"   📍 Близкие споры (< {self.distance_threshold}): "
              f"{len(close_spores)}")
        print(f"   📏 Не близкие споры (≥ {self.distance_threshold}): "
              f"{len(far_spores)}")
        print(f"   📊 Всего спор в графе: {len(all_spores)}")
        
        if all_spores:
            print("\n   📍 ВСЕ СПОРЫ (отсортированы по дистанции):")
            for i, spore_info in enumerate(all_spores, 1):
                pos = spore_info['position']
                dist = spore_info['distance']
                is_close = dist < self.distance_threshold
                marker = "📍" if is_close else "📏"
                print(f"      {i:2d}. {marker} {spore_info['id']}: "
                      f"({pos[0]:.4f}, {pos[1]:.4f}), dist={dist:.4f}")
        else:
            print("\n   📭 Спор в графе нет")

    def set_verbose_output(self, verbose: bool) -> None:
        """
        Включает или отключает подробный вывод при изменении look_point.

        Args:
            verbose: True для включения подробного вывода, False для отключения
        """
        self.verbose_output = verbose
        status = 'включен' if verbose else 'отключен'
        print(f"🎯 Подробный вывод {status}")

    def toggle_verbose_output(self) -> None:
        """Переключает режим подробного вывода."""
        self.verbose_output = not self.verbose_output
        status = 'включен' if self.verbose_output else 'отключен'
        print(f"🎯 Подробный вывод {status}")
