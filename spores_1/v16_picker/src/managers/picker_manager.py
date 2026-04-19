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
import os
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
        self._neighbor_cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

        # Предыдущие координаты look_point для проверки изменений
        self.last_look_point: Optional[Tuple[float, float]] = None
        
        # 🆕 Кеширование JSON данных
        self._cached_graph_data: Dict[str, Any] = {}
        self._last_json_modified_time: float = 0
        self._json_path: str = os.path.join("buffer", "real_graph_latest.json")
        
        print("🎯 PickerManager с поддержкой JSON инициализирован")
        
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
            
            # 🆕 Принудительная проверка обновлений JSON перед анализом
            self._force_json_reload_if_needed()

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
        neighbor_cache: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        for spore_info in new_close_spores:
            spore_neighbors: Dict[int, List[Dict[str, Any]]] = {}
            try:
                spore_neighbors = self._collect_neighbors_snapshot(spore_info['id'])
            except Exception as error:
                if self.verbose_output:
                    print(f"[PickerManager] failed to collect neighbors for {spore_info['id']}: {error}")
            neighbor_cache[spore_info['id']] = spore_neighbors
            spore_info['neighbors'] = spore_neighbors

        self._neighbor_cache = neighbor_cache

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
                
                # Выводим соседей самой близкой споры (по JSON)
                self._analyze_spore_neighbors(closest_spore)
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
                # Fallback: используем spore_id если доступен
                if hasattr(spore, 'spore_id'):
                    return str(spore.spore_id)
                elif hasattr(spore, 'get_spore_id'):
                    return str(spore.get_spore_id())
                else:
                    return "?"
                
        except Exception:
            # Fallback: используем spore_id если доступен
            if hasattr(spore, 'spore_id'):
                return str(spore.spore_id)
            elif hasattr(spore, 'get_spore_id'):
                return str(spore.get_spore_id())
            else:
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
        
        # 🔍 ОТЛАДОЧНАЯ ВЕРИФИКАЦИЯ ИСПРАВЛЕНИЙ
        print(f"🔧 ВЕРИФИКАЦИЯ ИСПРАВЛЕНИЙ:")

        # Проверяем материализованные связи
        materialized_links = [link for link in self.spore_manager.links if hasattr(link, 'control_value')]
        buffer_links = [link for link in self.spore_manager.links if hasattr(link, 'dt_value')]

        print(f"   📊 Связей с control_value: {len(materialized_links)}")
        print(f"   📊 Связей с dt_value: {len(buffer_links)}")

        # Показываем статистику по знакам управления
        if materialized_links:
            positive_controls = [link for link in materialized_links if link.control_value > 0]
            negative_controls = [link for link in materialized_links if link.control_value < 0]
            zero_controls = [link for link in materialized_links if link.control_value == 0]
            
            print(f"   ✅ Связей с управлением +: {len(positive_controls)}")
            print(f"   ✅ Связей с управлением -: {len(negative_controls)}") 
            print(f"   ✅ Связей с управлением 0: {len(zero_controls)}")
            
            # Показываем несколько примеров
            print(f"   📝 ПРИМЕРЫ ИСПРАВЛЕННЫХ СВЯЗЕЙ:")
            for i, link in enumerate(materialized_links[:3]):  # Первые 3
                dt_val = getattr(link, 'dt_value', 'N/A')
                control_val = link.control_value
                print(f"      {i+1}. Управление: {control_val:+.1f}, dt: {dt_val}")
        else:
            print(f"   ⚠️ НЕТ СВЯЗЕЙ С control_value - исправление не сработало!")
        
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
        """Return direct neighbors in both directions with per-step metadata."""
        neighbors: List[Dict[str, Any]] = []

        # Outgoing edges from this spore (forward direction)
        children = self.spore_manager.graph.get_children(spore_id)
        for child in children:
            child_id = self.spore_manager.graph._get_spore_id(child)
            edge_info = self.spore_manager.graph.get_edge_info(spore_id, child_id)

            dt_value = self._extract_dt(edge_info, 'forward')
            neighbor_info = {
                'target_spore': child,
                'target_id': child_id,
                'path': [spore_id, child_id],
                'edges': [edge_info] if edge_info else [],
                'time_direction': 'forward',
                'dt': dt_value,
                'dt_sequence': [dt_value],
                'step_time_directions': ['forward'],
                'can_reach': True
            }
            neighbors.append(neighbor_info)

        # Incoming edges to this spore (backward direction)
        parents = self.spore_manager.graph.get_parents(spore_id)
        for parent in parents:
            parent_id = self.spore_manager.graph._get_spore_id(parent)
            edge_info = self.spore_manager.graph.get_edge_info(parent_id, spore_id)

            dt_value = self._extract_dt(edge_info, 'backward')
            neighbor_info = {
                'target_spore': parent,
                'target_id': parent_id,
                'path': [spore_id, parent_id],
                'edges': [edge_info] if edge_info else [],
                'time_direction': 'backward',
                'dt': dt_value,
                'dt_sequence': [dt_value],
                'step_time_directions': ['backward'],
                'can_reach': True
            }
            neighbors.append(neighbor_info)

        return neighbors

    def _collect_neighbors_snapshot(
            self,
            spore_id: str,
            max_distance: int = 2
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Collect neighbors around a spore for quick access by distance.
        """
        snapshot: Dict[int, List[Dict[str, Any]]] = {}

        if not self.spore_manager or not getattr(self.spore_manager, 'graph', None):
            return snapshot

        for distance in range(1, max_distance + 1):
            raw_neighbors = self._get_neighbors_at_distance(spore_id, distance)
            if not raw_neighbors:
                continue

            serialized_neighbors = []
            seen_signatures = set()

            for neighbor in raw_neighbors:
                serialized = self._serialize_neighbor_info(distance, neighbor)
                signature = (
                    serialized.get('visual_id'),
                    tuple(serialized.get('path', [])),
                    tuple(serialized.get('dt_values', [])),
                    tuple(serialized.get('control_values', [])),
                    tuple(serialized.get('step_time_directions', []))
                )
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                serialized_neighbors.append(serialized)

            if serialized_neighbors:
                snapshot[distance] = serialized_neighbors

        return snapshot

    def _serialize_neighbor_info(
            self,
            distance: int,
            neighbor_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert raw neighbor info into a JSON-friendly structure.
        """
        target_id = neighbor_info.get('target_id')
        target_spore = neighbor_info.get('target_spore')
        position: Optional[Tuple[float, float]] = None

        if target_spore is not None:
            visual_id = self._get_visual_spore_id(target_spore)
            try:
                pos = target_spore.calc_2d_pos()
                position = (float(pos[0]), float(pos[1]))
            except Exception:
                position = None
        else:
            visual_id = str(target_id) if target_id is not None else None

        raw_path = list(neighbor_info.get('path', []))

        serialized: Dict[str, Any] = {
            'distance': distance,
            'target_id': target_id,
            'visual_id': visual_id,
            'position': position,
            'raw_path': raw_path,
            'path': self._convert_path_to_visual_ids(raw_path),
            'time_direction': neighbor_info.get('time_direction', 'unknown'),
            'can_reach': bool(neighbor_info.get('can_reach', False)),
            'dt_values': [],
            'control_values': []
        }

        if distance == 2:
            serialized['intermediate_id'] = neighbor_info.get('intermediate_id')
            intermediate_spore = neighbor_info.get('intermediate_spore')

            if intermediate_spore is not None:
                serialized['intermediate_visual_id'] = self._get_visual_spore_id(intermediate_spore)
                try:
                    pos = intermediate_spore.calc_2d_pos()
                    serialized['intermediate_position'] = (float(pos[0]), float(pos[1]))
                except Exception:
                    serialized['intermediate_position'] = None
            else:
                serialized['intermediate_visual_id'] = (
                    str(serialized['intermediate_id'])
                    if serialized.get('intermediate_id') is not None
                    else None
                )
                serialized['intermediate_position'] = None

        step_time_dirs = list(neighbor_info.get('step_time_directions') or [])
        if not step_time_dirs:
            step_dir = neighbor_info.get('time_direction')
            if step_dir:
                step_time_dirs = [step_dir]

        step_dt_sequence = neighbor_info.get('dt_sequence')
        if step_dt_sequence is not None:
            step_dt_sequence = list(step_dt_sequence)
        else:
            dt_single = neighbor_info.get('dt')
            step_dt_sequence = [dt_single] if dt_single is not None else []

        num_steps = max(len(raw_path) - 1, len(step_time_dirs), len(step_dt_sequence))

        while len(step_time_dirs) < num_steps:
            step_time_dirs.append('unknown')
        while len(step_dt_sequence) < num_steps:
            step_dt_sequence.append(None)

        edges = neighbor_info.get('edges') or []
        dt_values: List[Optional[float]] = []
        control_values: List[Optional[float]] = []

        for idx in range(num_steps):
            edge = edges[idx] if idx < len(edges) else None
            step_dir = step_time_dirs[idx]
            dt_override = step_dt_sequence[idx]

            control_value = None
            dt_candidate = None
            if edge and getattr(edge, 'link_object', None):
                link = edge.link_object
                dt_candidate = self._to_serializable_number(getattr(link, 'dt_value', None))
                control_value = getattr(link, 'control_value', None)

            if dt_candidate is None:
                dt_candidate = self._to_serializable_number(dt_override)

            dt_value = self._adjust_dt_sign(dt_candidate, step_dir)

            dt_values.append(self._to_serializable_number(dt_value))
            control_values.append(self._to_serializable_number(control_value))

        serialized['dt_values'] = dt_values
        serialized['control_values'] = control_values
        serialized['step_time_directions'] = step_time_dirs[:num_steps]
        serialized['step_dt_values'] = dt_values.copy()

        return serialized

    def _to_serializable_number(self, value: Any) -> Optional[float]:
        """
        Convert numpy-based values to plain Python numbers when possible.
        """
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.flatten()[0]

        if isinstance(value, (np.floating, np.integer)):
            return float(value)

        if isinstance(value, (float, int)):
            return float(value)

        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    
    def _combine_time_directions(self, first: str, second: str) -> str:
        """Combine directions for two-step routes."""
        if first == 'unknown' or second == 'unknown':
            return 'unknown'
        if first == second:
            return first
        return 'mixed'

    def _extract_dt(self, edge_info: Any, direction: str) -> Optional[float]:
        """Extract dt value from edge and align sign with expected direction."""
        if not edge_info or not getattr(edge_info, 'link_object', None):
            return None

        raw_dt = getattr(edge_info.link_object, 'dt_value', None)
        dt_serializable = self._to_serializable_number(raw_dt)
        return self._adjust_dt_sign(dt_serializable, direction)

    def _adjust_dt_sign(self, dt_value: Any, direction: str) -> Optional[float]:
        """Adjust dt sign so that forward steps are positive and backward negative."""
        if dt_value is None:
            return None
        try:
            dt_float = float(dt_value)
        except (TypeError, ValueError):
            return dt_value

        if direction == 'forward':
            return abs(dt_float)
        if direction == 'backward':
            return -abs(dt_float)
        return dt_float

    def _format_step_description(self, dt: Optional[float], direction: str) -> str:
        """Format a single step description with dt and direction label."""
        dt_str = self._format_signed_number(dt)
        direction_str = self._format_time_direction(direction)
        return f"{dt_str} ({direction_str})"

    def _convert_path_to_visual_ids(self, path: List[Any]) -> List[str]:
        """Convert internal graph node ids to visual ids used in UI."""
        if not path:
            return []

        visual_path: List[str] = []
        nodes_map = getattr(self.spore_manager.graph, 'nodes', {})

        for node_id in path:
            spore_obj = nodes_map.get(node_id)
            if spore_obj is not None:
                visual_path.append(self._get_visual_spore_id(spore_obj))
            else:
                visual_path.append(str(node_id))

        return visual_path

    def _format_signed_number(self, value: Optional[float], precision: int = 3) -> str:
        """Render numbers with an explicit sign for console output."""
        if value is None:
            return '—'
        try:
            return f"{float(value):+.{precision}f}"
        except (TypeError, ValueError):
            return str(value)
    
    def _format_time_direction(self, direction: str) -> str:
        """
        Return human-readable text for time direction.
        """
        mapping = {
            'forward': "прямое время ⏩",
            'backward': "обратное время ⏪",
            'mixed': "смешанное время 🔁",
            'unknown': "время неизвестно ❓"
        }
        return mapping.get(direction, mapping['unknown'])

    def _print_neighbor_cache_summary(
            self,
            visual_id: str,
            neighbors_snapshot: Optional[Dict[int, List[Dict[str, Any]]]]
    ) -> None:
        """Print cached neighbors grouped by graph distance."""
        print(f"\n🧭 СОСЕДИ СПОРЫ {visual_id}:")

        if not neighbors_snapshot:
            print('   • Соседи не найдены.')
            return

        for distance in sorted(neighbors_snapshot.keys()):
            neighbors = neighbors_snapshot.get(distance, [])
            header = 'На расстоянии 1' if distance == 1 else f'На расстоянии {distance}'
            print(f"   • {header} ({len(neighbors)}):")

            for neighbor in neighbors:
                target_visual = neighbor.get('visual_id') or neighbor.get('target_id') or '?'
                position = neighbor.get('position')
                if isinstance(position, (tuple, list)) and len(position) == 2:
                    pos_str = f"({position[0]:.4f}, {position[1]:.4f})"
                else:
                    pos_str = '(нет позиции)'

                step_dt_values = neighbor.get('step_dt_values', neighbor.get('dt_values', []))
                step_dirs = neighbor.get('step_time_directions', [])

                dt_str = ', '.join(
                    self._format_signed_number(val) if val is not None else '—'
                    for val in step_dt_values
                ) or '—'

                control_values = neighbor.get('control_values') or []
                control_str = ', '.join(
                    self._format_signed_number(val, precision=2) if val is not None else 'None'
                    for val in control_values
                ) or '—'

                path = neighbor.get('path') or []
                path_str = '→'.join(str(node) for node in path) if path else '—'

                step_parts = []
                for idx, dt_val in enumerate(step_dt_values):
                    direction_for_step = step_dirs[idx] if idx < len(step_dirs) else 'unknown'
                    step_parts.append(self._format_step_description(dt_val, direction_for_step))
                steps_str = '; '.join(step_parts) if step_parts else '—'

                time_direction = neighbor.get('time_direction', 'unknown')
                time_summary = self._format_time_direction(time_direction)

                extra = ''
                if distance == 2:
                    intermediate = neighbor.get('intermediate_visual_id') or neighbor.get('intermediate_id')
                    if intermediate:
                        extra = f', через {intermediate}'

                print(
                    f"      🎯 Спора {target_visual} {pos_str} | dt: {dt_str} | "
                    f"u: {control_str} | путь: {path_str} | шаги: {steps_str} | итог: {time_summary}{extra}"
                )

        return neighbors

    def _get_neighbors_at_distance_2(self, spore_id: str) -> List[Dict[str, Any]]:
        """???????? ??????? ????? ?? ?????????? 2 ? ?????? ??????????? ???????."""
        neighbors: List[Dict[str, Any]] = []

        direct_neighbors = self._get_direct_neighbors(spore_id)

        for direct_neighbor in direct_neighbors:
            intermediate_id = direct_neighbor['target_id']
            if intermediate_id == spore_id:
                continue

            intermediate_neighbors = self._get_direct_neighbors(intermediate_id)

            for neighbor in intermediate_neighbors:
                target_id = neighbor['target_id']
                if target_id in {spore_id, intermediate_id}:
                    continue

                path = [spore_id, intermediate_id, target_id]
                edges = (direct_neighbor.get('edges') or []) + (neighbor.get('edges') or [])

                first_direction = direct_neighbor.get('time_direction', 'unknown')
                second_direction = neighbor.get('time_direction', 'unknown')
                combined_direction = self._combine_time_directions(first_direction, second_direction)

                step_dirs = list(direct_neighbor.get('step_time_directions') or [first_direction])
                step_dirs += list(neighbor.get('step_time_directions') or [second_direction])

                direct_dt_seq = direct_neighbor.get('dt_sequence') or [direct_neighbor.get('dt')]
                neighbor_dt_seq = neighbor.get('dt_sequence') or [neighbor.get('dt')]
                combined_dt_seq = list(direct_dt_seq) + list(neighbor_dt_seq)

                neighbor_info = {
                    'target_spore': neighbor.get('target_spore'),
                    'target_id': target_id,
                    'path': path,
                    'edges': edges,
                    'intermediate_spore': direct_neighbor.get('target_spore'),
                    'intermediate_id': intermediate_id,
                    'time_direction': combined_direction,
                    'step_time_directions': step_dirs,
                    'dt_sequence': combined_dt_seq,
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
            
            # Анализируем временные направления каждой связи
            route_analysis = []
            if len(neighbor_info['edges']) >= 1:
                dt1 = getattr(neighbor_info['edges'][0].link_object, 'dt_value', 0)
                route_analysis.append(f"dt1={dt1:+.3f}")

            if len(neighbor_info['edges']) >= 2:
                dt2 = getattr(neighbor_info['edges'][1].link_object, 'dt_value', 0)
                route_analysis.append(f"dt2={dt2:+.3f}")

            # Определяем направление маршрута по паттерну временных шагов
            first_time_forward = (len(neighbor_info['edges']) > 0 and
                                  getattr(neighbor_info['edges'][0].link_object, 'dt_value', 0) >= 0)
            second_time_forward = (len(neighbor_info['edges']) > 1 and
                                   getattr(neighbor_info['edges'][1].link_object, 'dt_value', 0) >= 0)

            if first_time_forward and second_time_forward:
                route_direction = "⏩"
                route_description = f"прямое время ({', '.join(route_analysis)})"
            elif not first_time_forward and not second_time_forward:
                route_direction = "⏪"
                route_description = f"обратное время ({', '.join(route_analysis)})"
            else:
                route_direction = "🔄"
                route_description = f"смешанное время ({', '.join(route_analysis)})"

            print(f"         📍 Маршрут длиной 2 {route_direction} ({route_description})")
            
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

                        # Определяем направление времени и управления
                        if dt != 'N/A' and control != 'N/A':
                            # Форматируем управление со знаком
                            control_str = f"+{control}" if control > 0 else str(control)

                            # 🔧 ИСПРАВЛЕНО: Правильная интерпретация стрелок
                            # Стрелка A→B с dt>0: из A в B прямое время
                            # Стрелка A→B с dt<0: из A в B обратное время
                            if dt > 0:
                                time_direction = "прямое время"
                                dt_str = f"+{dt}"
                            elif dt < 0:
                                time_direction = "обратное время"
                                dt_str = str(dt)
                            else:
                                time_direction = "нулевое время"
                                dt_str = "0.000"

                            print(f"         🔗 Линк 1: управление={control_str}, время={dt_str} "
                                  f"({time_direction}, источник dt: {dt_source}, источник control: {control_source})")
                        else:
                            # Резервная логика для случаев без данных
                            dt_str = f"+{dt}" if dt != 'N/A' and dt >= 0 else str(dt)
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

                        # Определяем направление времени и управления
                        if dt != 'N/A' and control != 'N/A':
                            # Форматируем управление со знаком
                            control_str = f"+{control}" if control > 0 else str(control)

                            # 🔧 ИСПРАВЛЕНО: Правильная интерпретация стрелок
                            # Стрелка A→B с dt>0: из A в B прямое время
                            # Стрелка A→B с dt<0: из A в B обратное время
                            if dt > 0:
                                time_direction = "прямое время"
                                dt_str = f"+{dt}"
                            elif dt < 0:
                                time_direction = "обратное время"
                                dt_str = str(dt)
                            else:
                                time_direction = "нулевое время"
                                dt_str = "0.000"

                            print(f"         🔗 Линк 2: управление={control_str}, время={dt_str} "
                                  f"({time_direction}, источник dt: {dt_source}, источник control: {control_source})")
                        else:
                            # Резервная логика для случаев без данных
                            dt_str = f"+{dt}" if dt != 'N/A' and dt >= 0 else str(dt)
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

    def _force_json_reload_if_needed(self):
        """Принудительно проверяет и перезагружает JSON если нужно."""
        try:
            if os.path.exists(self._json_path):
                current_size = os.path.getsize(self._json_path)
                current_mtime = os.path.getmtime(self._json_path)
                
                # Проверяем не только время, но и размер (на случай быстрых обновлений)
                if current_mtime != self._last_json_modified_time:
                    # print(f"🔄 Обнаружено обновление JSON, перезагружаем...")
                    self._cached_graph_data.clear()
        except Exception as e:
            print(f"⚠️ Ошибка проверки JSON: {e}")

    def _load_real_graph_json(self) -> dict:
        """Загружает данные реального графа из JSON с кешированием."""
        try:
            import json
            
            if not os.path.exists(self._json_path):
                if self._cached_graph_data:
                    print(f"⚠️ JSON файл исчез, используем кеш")
                    return self._cached_graph_data
                print(f"⚠️ JSON файл реального графа не найден: {self._json_path}")
                return {}
            
            # Проверяем время модификации файла
            current_modified_time = os.path.getmtime(self._json_path)
            
            if (current_modified_time != self._last_json_modified_time or 
                not self._cached_graph_data):
                
                # Загружаем новые данные
                with open(self._json_path, 'r', encoding='utf-8') as f:
                    self._cached_graph_data = json.load(f)
                
                self._last_json_modified_time = current_modified_time

                # spore_count = len(self._cached_graph_data.get('spores', []))
                # link_count = len(self._cached_graph_data.get('links', []))
                # print(f"🔄 JSON обновлен: {spore_count} спор, {link_count} связей")
            
            return self._cached_graph_data
            
        except Exception as e:
            print(f"❌ Ошибка загрузки JSON: {e}")
            return self._cached_graph_data if self._cached_graph_data else {}

    def _analyze_spore_neighbors(self, spore_info: Dict[str, Any]) -> None:
        """Анализирует соседей споры на основе данных из JSON."""
        
        # Загружаем данные из JSON
        graph_data = self._load_real_graph_json()
        if not graph_data:
            print("❌ Нет данных графа для анализа")
            return
        
        target_spore = spore_info['spore']
        target_visual_id = self._get_visual_spore_id(target_spore)

        graph_spore_id = self.spore_manager.graph._get_spore_id(target_spore)

        neighbors_snapshot = spore_info.get('neighbors')
        if not neighbors_snapshot:
            neighbors_snapshot = self._collect_neighbors_snapshot(graph_spore_id)
            if neighbors_snapshot is not None:
                spore_info['neighbors'] = neighbors_snapshot

        self._print_neighbor_cache_summary(target_visual_id, neighbors_snapshot)

        json_spore_id = getattr(target_spore, 'spore_id', None)
        target_spore_data = None

        if json_spore_id is not None:
            for spore_data in graph_data.get('spores', []):
                if spore_data.get('spore_id') == json_spore_id:
                    target_spore_data = spore_data
                    break

        if not target_spore_data:
            idx_guess = None
            if json_spore_id is None:
                try:
                    idx_guess = int(target_visual_id) - 1
                except ValueError:
                    idx_guess = None
            if idx_guess is not None:
                spores_list = graph_data.get('spores', [])
                if 0 <= idx_guess < len(spores_list):
                    target_spore_data = spores_list[idx_guess]

        if not target_spore_data:
            msg = f"❌ Спора {target_visual_id} не найдена в JSON данных"
            if json_spore_id is not None:
                msg += f" (ожидаемый spore_id={json_spore_id})"
            print(msg)
            return

        print(f"\n🔗 СОСЕДИ СПОРЫ {target_visual_id} (из JSON):")
        
        # 🔧 ОТЛАДОЧНАЯ ИНФОРМАЦИЯ О JSON
        metadata = graph_data.get('metadata', {})
        export_time = metadata.get('export_time', 'неизвестно')
        print(f"📋 JSON экспорт от: {export_time}")
        
        # Анализируем исходящие связи (out_links) - куда можем попасть
        out_links = target_spore_data.get('out_links', [])
        if out_links:
            print(f"   📍 ИСХОДЯЩИЕ СВЯЗИ (куда можем попасть):")
            for i, link_data in enumerate(out_links):
                # 🔧 ИСПРАВЛЕНИЕ: out_links теперь содержит словари с полной информацией
                to_spore_id = link_data.get('to_spore_id')
                
                # Используем данные прямо из link_data вместо поиска в links
                control = link_data.get('control', 0)
                dt = link_data.get('dt', 0)
                dt_sign = link_data.get('dt_sign', 1)
                
                if to_spore_id:
                    # 🔧 ИСПРАВЛЕНИЕ: to_spore_id уже является визуальным номером
                    target_visual_id_out = to_spore_id
                    
                    # 🔧 ИСПРАВЛЕНИЕ: Исходящая связь - всегда прямое время (положительное)
                    time_direction = "прямое время"
                    time_symbol = "⏩"
                    dt_str = f"+{dt}"  # Всегда положительное время
                    
                    control_str = f"+{control}" if control > 0 else str(control)
                    link_type = "max" if control > 0 else "min"  # Простое определение типа
                    
                    print(f"      🎯 Спора {target_visual_id_out}: {link_type}, управление={control_str}, время={dt_str} ({time_direction}) {time_symbol}")
                else:
                    print(f"      🎯 Спора 0: (детали связи не найдены)")
        
        # Анализируем входящие связи (in_links) - откуда можем прийти
        in_links = target_spore_data.get('in_links', [])
        if in_links:
            print(f"   📍 ВХОДЯЩИЕ СВЯЗИ (откуда можем прийти):")
            for i, link_data in enumerate(in_links):
                # 🔧 ИСПРАВЛЕНИЕ: in_links теперь содержит словари с полной информацией
                from_spore_id = link_data.get('from_spore_id')
                
                # Используем данные прямо из link_data вместо поиска в links
                control = link_data.get('control', 0)
                dt = link_data.get('dt', 0)
                dt_sign = link_data.get('dt_sign', 1)
                
                if from_spore_id:
                    # 🔧 ИСПРАВЛЕНИЕ: from_spore_id уже является визуальным номером
                    source_visual_id = from_spore_id
                    
                    # 🔧 ИСПРАВЛЕНИЕ: Входящая связь - всегда обратное время (отрицательное)
                    time_direction = "обратное время"
                    time_symbol = "⏪"
                    dt_str = f"-{dt}"  # Всегда отрицательное время
                    
                    control_str = f"+{control}" if control > 0 else str(control)
                    link_type = "max" if control > 0 else "min"  # Простое определение типа
                    
                    print(f"      🎯 Спора {source_visual_id}: {link_type}, управление={control_str}, время={dt_str} ({time_direction}) {time_symbol}")
                else:
                    print(f"      🎯 Спора 0: (детали связи не найдены)")

    def _find_link_details(self, parent_spore_id: str, child_spore_id: str, graph_data: dict) -> dict:
        """Находит детальную информацию о связи в массиве links."""
        for link_data in graph_data.get('links', []):
            if (link_data.get('parent_spore_id') == parent_spore_id and 
                link_data.get('child_spore_id') == child_spore_id):
                return link_data
        return {}

    def _find_visual_id_by_spore_id(self, spore_id: str, graph_data: dict) -> int:
        """Находит визуальный ID споры по её spore_id в JSON данных."""
        for spore_data in graph_data.get('spores', []):
            if spore_data['spore_id'] == spore_id:
                return spore_data['index'] + 1
        return 0
