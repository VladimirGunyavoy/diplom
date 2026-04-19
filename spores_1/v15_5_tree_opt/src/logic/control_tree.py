"""
Модуль для построения траекторий управления со схождением.
Файл: spores/v14_back/src/logic/control_tree.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    """Траектория из двух шагов управления."""
    id: int
    sequence: List[Tuple[float, float]]  # [(u1, dt1), (u2, dt2)]
    points: List[np.ndarray]  # [начальная, промежуточная, конечная]
    convergence_group: int  # Номер группы схождения (0-3)
    
    def __repr__(self):
        seq_str = " → ".join([f"({u:+.0f}u, {dt:+.0f}T)" for u, dt in self.sequence])
        return f"Traj{self.id}: {seq_str}"


class ControlTreeBuilder:
    """
    Построитель системы траекторий со схождением.
    
    Создает 8 траекторий длины 2, где пары траекторий сходятся в 4 точки.
    
    Использование dt_vector:
    - dt[0], dt[1] - для группы 0 (траектории 0-1)
    - dt[2], dt[3] - для группы 1 (траектории 2-3)
    - dt[4], dt[5] - для группы 2 (траектории 4-5)
    - dt[6], dt[7] - для группы 3 (траектории 6-7)
    
    В каждой группе траектории используют одинаковые dt в обратном порядке для схождения.
    """
    
    def __init__(self, pendulum_system, dt_vector: Optional[np.ndarray] = None):
        """
        Args:
            pendulum_system: Система маятника для расчета динамики
            dt_vector: Вектор из 8 временных шагов (по умолчанию np.ones(8) * 0.1)
        """
        self.pendulum = pendulum_system
        
        # Вектор временных шагов для 8 траекторий
        if dt_vector is None:
            self.dt_vector = np.ones(8) * 0.05
        else:
            assert len(dt_vector) == 8, "dt_vector должен содержать ровно 8 элементов"
            self.dt_vector = np.array(dt_vector)
        
        # Получаем границы управления
        control_bounds = self.pendulum.get_control_bounds()
        self.u_max = control_bounds[1]  # Используем только max, min = -max
        
        # Хранилище траекторий
        self.trajectories: List[Trajectory] = []
        
        # Определяем 8 последовательностей управлений (u нормализован на ±1)
        # Каждая пара должна сходиться в одну точку
        self.control_sequences = [
            # Группа 0: сходятся в точку через forward-forward
            [(1, 1), (-1, 1)],   # Траектория 0: (+u, +T) → (-u, +T)
            [(-1, 1), (1, 1)],   # Траектория 1: (-u, +T) → (+u, +T)
            
            # Группа 1: сходятся через forward-backward 
            [(-1, 1), (1, -1)],  # Траектория 2: (-u, +T) → (+u, -T)
            [(1, -1), (-1, 1)],  # Траектория 3: (+u, -T) → (-u, +T)
            
            # Группа 2: сходятся через backward-backward
            [(1, -1), (-1, -1)], # Траектория 4: (+u, -T) → (-u, -T)
            [(-1, -1), (1, -1)], # Траектория 5: (-u, -T) → (+u, -T)
            
            # Группа 3: сходятся через forward-backward альтернативно
            [(1, 1), (-1, -1)],  # Траектория 6: (+u, +T) → (-u, -T)
            [(-1, -1), (1, 1)],  # Траектория 7: (-u, -T) → (+u, +T)
        ]
        
        # Группы схождения (какие траектории должны прийти в одну точку)
        self.convergence_groups = [
            [0, 1],  # Группа 0
            [2, 3],  # Группа 1  
            [4, 5],  # Группа 2
            [6, 7],  # Группа 3
        ]
    
    def _apply_control(self, state_2d: np.ndarray, control: float, dt: float) -> np.ndarray:
        """
        Применяет управление к состоянию.
        
        Args:
            state_2d: Текущее 2D состояние
            control: Управляющее воздействие (уже масштабированное)
            dt: Временной шаг (может быть отрицательным)
            
        Returns:
            Новое 2D состояние
        """
        return self.pendulum.scipy_rk45_step(state_2d, control, dt)
    
    def build_tree(self, initial_position_2d: np.ndarray, show: bool = False) -> Dict[str, Any]:
        """
        Строит все 8 траекторий из начальной точки.
        
        Args:
            initial_position_2d: Начальная 2D позиция
            
        Returns:
            Словарь с траекториями и информацией о схождении
        """
        # Очищаем предыдущие траектории
        self.trajectories.clear()
        
        if show:
            print(f"🚀 Построение 8 траекторий из точки ({initial_position_2d[0]:.3f}, {initial_position_2d[1]:.3f})")
            print("=" * 60)
        
        # Маппинг: какой элемент dt_vector использовать для каждого кортежа
        # Группа 0: траектории 0-1, используют dt[0-1]
        # Группа 1: траектории 2-3, используют dt[2-3]  
        # Группа 2: траектории 4-5, используют dt[4-5]
        # Группа 3: траектории 6-7, используют dt[6-7]
        dt_mapping = {
            0: [0, 1],  # Траектория 0: dt[0] для 1-го шага, dt[1] для 2-го
            1: [1, 0],  # Траектория 1: dt[1] для 1-го шага, dt[0] для 2-го (обратный порядок)
            2: [2, 3],  # Траектория 2: dt[2] для 1-го шага, dt[3] для 2-го
            3: [3, 2],  # Траектория 3: dt[3] для 1-го шага, dt[2] для 2-го (обратный порядок)
            4: [4, 5],  # Траектория 4: dt[4] для 1-го шага, dt[5] для 2-го
            5: [5, 4],  # Траектория 5: dt[5] для 1-го шага, dt[4] для 2-го (обратный порядок)
            6: [6, 7],  # Траектория 6: dt[6] для 1-го шага, dt[7] для 2-го
            7: [7, 6],  # Траектория 7: dt[7] для 1-го шага, dt[6] для 2-го (обратный порядок)
        }
        
        # Строим каждую траекторию
        for traj_id in range(8):
            # Получаем последовательность управлений для этой траектории
            control_seq = self.control_sequences[traj_id]
            dt_indices = dt_mapping[traj_id]
            
            # Определяем группу схождения
            convergence_group = traj_id // 2  # 0-1 → 0, 2-3 → 1, 4-5 → 2, 6-7 → 3
            
            # Вычисляем точки траектории
            points = [initial_position_2d.copy()]  # Начальная точка
            current_pos = initial_position_2d.copy()
            
            # Строим последовательность с индивидуальными dt для каждого шага
            actual_sequence = []
            for step_idx, (u_sign, dt_sign) in enumerate(control_seq):
                # Масштабируем управление
                u = self.u_max * u_sign
                # Используем соответствующий dt из вектора
                dt = self.dt_vector[dt_indices[step_idx]]
                step_dt = dt * dt_sign  # dt_sign определяет направление времени
                
                # Применяем управление
                next_pos = self._apply_control(current_pos, u, step_dt)
                points.append(next_pos.copy())
                current_pos = next_pos
                
                actual_sequence.append((u, step_dt))
            
            # Создаем объект траектории
            trajectory = Trajectory(
                id=traj_id,
                sequence=actual_sequence,
                points=points,
                convergence_group=convergence_group
            )
            
            self.trajectories.append(trajectory)
            
            if show:
            # Выводим информацию
                seq_str = " → ".join([f"({u:+.1f}, {dt:+.3f})" for u, dt in actual_sequence])
                dt_idx_str = f"dt[{dt_indices[0]}], dt[{dt_indices[1]}]"
                print(f"Траектория {traj_id} (группа {convergence_group}, {dt_idx_str}): {seq_str}")
                print(f"  Точки: начало → ({points[1][0]:.3f}, {points[1][1]:.3f}) → "
                    f"({points[2][0]:.3f}, {points[2][1]:.3f})")
        
        # Анализируем схождение
        convergence_info = self._analyze_convergence(show)
        
        # Создаем структуру для совместимости с визуализацией
        nodes, edges = self._create_graph_structure()
        
        return {
            'trajectories': self.trajectories,
            'convergence_info': convergence_info,
            'dt_vector': self.dt_vector.copy(),
            'nodes': nodes,  # Для совместимости с визуализацией
            'edges': edges,   # Для совместимости с визуализацией
        }
    
    def _analyze_convergence(self, show: bool = False) -> Dict[str, Any]:
        """Анализирует качество схождения траекторий."""
        if show:
            print("\n" + "=" * 60)
            print("📊 АНАЛИЗ СХОЖДЕНИЯ")
            print("=" * 60)
        
        convergence_quality = []
        
        for group_id, traj_indices in enumerate(self.convergence_groups):
            # Получаем конечные точки траекторий в группе
            endpoints = [self.trajectories[i].points[-1] for i in traj_indices]
            
            # Вычисляем среднюю точку и отклонения
            mean_point = np.mean(endpoints, axis=0)
            distances = [np.linalg.norm(p - mean_point) for p in endpoints]
            max_distance = max(distances)
            
            if show:
                print(f"\nГруппа {group_id} (траектории {traj_indices}):")
            for i, idx in enumerate(traj_indices):
                traj = self.trajectories[idx]
                endpoint = endpoints[i]

                if show:
                    print(f"  Траектория {idx}: конечная точка ({endpoint[0]:.4f}, {endpoint[1]:.4f})")
            if show:
                print(f"  Средняя точка: ({mean_point[0]:.4f}, {mean_point[1]:.4f})")
                print(f"  Максимальное отклонение: {max_distance:.6f}")
            
            # Проверяем качество схождения
            converged = max_distance < 1e-4  # Порог для определения схождения
            status = "✅ СОШЛИСЬ" if converged else "⚠️ НЕ СОШЛИСЬ"
            if show:
                print(f"  Статус: {status}")
            
            convergence_quality.append({
                'group_id': group_id,
                'trajectories': traj_indices,
                'mean_point': mean_point,
                'max_deviation': max_distance,
                'converged': converged,
                'endpoints': endpoints
            })
        
        # Общая статистика
        num_converged = sum(1 for g in convergence_quality if g['converged'])
        if show:
            print(f"\n📈 ИТОГО: {num_converged}/4 групп сошлись")
        
        return {
            'groups': convergence_quality,
            'num_converged': num_converged,
            'total_groups': len(self.convergence_groups)
        }
    
    def _create_graph_structure(self) -> Tuple[Dict, List]:
        """
        Создает структуру узлов и рёбер для совместимости с визуализацией.
        """
        from dataclasses import dataclass
        
        @dataclass
        class SimpleNode:
            id: str
            level: int
            position_2d: np.ndarray
            
        @dataclass  
        class SimpleEdge:
            parent_id: str
            child_id: str
            control: float
            dt: float
            is_forward: bool
            control_type: str
        
        nodes = {}
        edges = []
        
        # Корневой узел
        nodes['root'] = SimpleNode('root', 0, self.trajectories[0].points[0])
        
        # Для каждой траектории создаем узлы и рёбра
        for traj in self.trajectories:
            # Промежуточный узел
            mid_id = f'mid_{traj.id}'
            nodes[mid_id] = SimpleNode(mid_id, 1, traj.points[1])
            
            # Конечный узел
            end_id = f'end_{traj.id}'
            nodes[end_id] = SimpleNode(end_id, 2, traj.points[2])
            
            # Ребро от корня к промежуточному узлу
            u1, dt1 = traj.sequence[0]
            edges.append(SimpleEdge(
                'root', mid_id, u1, dt1,
                is_forward=(dt1 > 0),
                control_type='max' if u1 > 0 else 'min'
            ))
            
            # Ребро от промежуточного к конечному узлу
            u2, dt2 = traj.sequence[1]
            edges.append(SimpleEdge(
                mid_id, end_id, u2, dt2,
                is_forward=(dt2 > 0),
                control_type='max' if u2 > 0 else 'min'
            ))
        
        return nodes, edges
    
    def get_convergent_endpoints(self) -> Dict[int, np.ndarray]:
        """
        Возвращает средние точки схождения для каждой группы.
        
        Returns:
            Словарь {group_id: mean_endpoint}
        """
        result = {}
        for group_id, traj_indices in enumerate(self.convergence_groups):
            endpoints = [self.trajectories[i].points[-1] for i in traj_indices]
            result[group_id] = np.mean(endpoints, axis=0)
        return result
    
    def update_dt_vector(self, new_dt_vector: np.ndarray):
        """Обновляет вектор временных шагов."""
        assert len(new_dt_vector) == 8, "new_dt_vector должен содержать ровно 8 элементов"
        self.dt_vector = np.array(new_dt_vector)
        print(f"✅ Обновлен dt_vector: {self.dt_vector}")
    
    def get_trajectory_by_id(self, traj_id: int) -> Optional[Trajectory]:
        """Возвращает траекторию по ID."""
        if 0 <= traj_id < len(self.trajectories):
            return self.trajectories[traj_id]
        return None
    
    def print_summary(self):
        """Выводит краткую сводку о построенных траекториях."""
        print("\n📋 СВОДКА ПО ТРАЕКТОРИЯМ")
        print("=" * 60)
        
        for group_id, traj_indices in enumerate(self.convergence_groups):
            print(f"\nГруппа схождения {group_id}:")
            for idx in traj_indices:
                traj = self.trajectories[idx]
                print(f"  {traj}")
        
        print(f"\nВектор dt: {self.dt_vector}")