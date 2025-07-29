import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class MazePathPlanner:
    def __init__(self, size=20, obstacle_density=0.3):
        self.size = size
        # Создаем лабиринт: 0 - свободный путь, 1 - стена, 2 - начальная точка, 3 - цель
        self.maze = np.zeros((size, size))
        self.generate_maze(obstacle_density)
        
        # Находим свободные клетки для размещения начальной точки и цели
        free_cells = [(i, j) for i in range(size) for j in range(size) if self.maze[i, j] == 0]
        
        # Случайно выбираем начальную точку
        start_idx = random.choice(range(len(free_cells)))
        self.start = free_cells[start_idx]
        self.maze[self.start] = 2
        
        # Удаляем начальную точку из списка свободных клеток
        free_cells.pop(start_idx)
        
        # Случайно выбираем цель
        goal_idx = random.choice(range(len(free_cells)))
        self.goal = free_cells[goal_idx]
        self.maze[self.goal] = 3
        
        # Граф достижимости: точки и из каких точек они достижимы
        self.reachability_graph = {self.start: None}  # None означает, что это начальная точка
        
        # Точки, из которых можно достичь цели (прямо или через другие точки)
        self.can_reach_goal = set()
        
        # Посещенные точки для визуализации
        self.visited = set([self.start])
        
        # История путей для визуализации
        self.path_history = []
        
    def generate_maze(self, obstacle_density):
        """Создаем случайный лабиринт с заданной плотностью препятствий"""
        for i in range(self.size):
            for j in range(self.size):
                # Границы лабиринта всегда стены
                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    self.maze[i, j] = 1
                # Внутренние клетки - случайно
                elif random.random() < obstacle_density:
                    self.maze[i, j] = 1
    
    def is_valid_position(self, pos):
        """Проверяет, является ли позиция допустимой (внутри лабиринта и не стена)"""
        i, j = pos
        return 0 <= i < self.size and 0 <= j < self.size and self.maze[i, j] != 1
    
    def get_nearest_goal_reaching_point(self, point):
        """Находит ближайшую точку, из которой можно достичь цели"""
        if not self.can_reach_goal:
            # Если таких точек нет, направляемся к цели
            return self.goal
        
        min_dist = float('inf')
        nearest_point = None
        
        for goal_point in self.can_reach_goal:
            dist = np.sqrt((point[0] - goal_point[0])**2 + (point[1] - goal_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = goal_point
                
        return nearest_point
    
    def sample_action(self, current_pos, target_pos, std_dev=1.0):
        """
        Сэмплирует действие с матожиданием в направлении цели
        Движение только в 4 направлениях: вверх, вниз, влево, вправо
        """
        # Вычисляем направление к цели
        direction = np.array([target_pos[0] - current_pos[0], target_pos[1] - current_pos[1]])
        
        # Доступные действия (верх, вниз, влево, вправо)
        possible_actions = [
            (current_pos[0] - 1, current_pos[1]),  # вверх
            (current_pos[0] + 1, current_pos[1]),  # вниз
            (current_pos[0], current_pos[1] - 1),  # влево
            (current_pos[0], current_pos[1] + 1)   # вправо
        ]
        
        # Если направление к цели близко к нулю, выбираем случайное направление
        if np.linalg.norm(direction) < 0.01:
            return random.choice(possible_actions)
        
        # Вычисляем вероятности для каждого действия на основе направления к цели
        probs = []
        
        for action in possible_actions:
            # Вектор действия
            action_vec = np.array([action[0] - current_pos[0], action[1] - current_pos[1]])
            
            # Скалярное произведение с направлением к цели (косинус угла)
            dot_product = np.dot(action_vec, direction)
            
            # Преобразуем в вероятность (больше в направлении цели)
            prob = max(0.1, (dot_product + 1) / 2)  # Чтобы вероятность была всегда положительной
            probs.append(prob)
        
        # Нормализуем вероятности
        probs = np.array(probs) / sum(probs)
        
        # Выбираем действие на основе вероятностей
        chosen_action = np.random.choice(len(possible_actions), p=probs)
        
        return possible_actions[chosen_action]
    
    def run_random_sequence(self, start_point, sequence_length=10, num_sequences=10):
        """Запускает несколько случайных последовательностей шагов из заданной точки"""
        reached_points = set()
        
        # Определяем цель для матожидания
        if self.can_reach_goal:
            target = self.get_nearest_goal_reaching_point(start_point)
        else:
            target = self.goal
            
        # Запускаем несколько последовательностей
        for _ in range(num_sequences):
            current = start_point
            path = [current]
            
            for _ in range(sequence_length):
                next_pos = self.sample_action(current, target)
                
                # Если следующая позиция недопустима, пробуем еще раз
                attempts = 0
                while not self.is_valid_position(next_pos) and attempts < 10:
                    next_pos = self.sample_action(current, target)
                    attempts += 1
                
                if self.is_valid_position(next_pos):
                    current = next_pos
                    path.append(current)
                    
                    # Добавляем точку в граф достижимости
                    if current not in self.reachability_graph:
                        self.reachability_graph[current] = start_point
                    
                    # Отмечаем точку как посещенную
                    self.visited.add(current)
                    
                    # Если достигли цели
                    if current == self.goal:
                        self.can_reach_goal.add(start_point)
                        # Сохраняем путь для визуализации
                        self.path_history.append(path)
                        return True
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_goal:
                        self.can_reach_goal.add(start_point)
                        # Сохраняем путь для визуализации
                        self.path_history.append(path)
                        return True
                    
                    reached_points.add(current)
            
            # Сохраняем путь для визуализации
            self.path_history.append(path)
        
        return False
    
    def update_goal_reaching_points(self):
        """Обновляет множество точек, из которых можно достичь цель,
        включая точки, ведущие к уже известным точкам"""
        changed = True
        while changed:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам в графе достижимости
            for point, parent in self.reachability_graph.items():
                # Если точка уже помечена как ведущая к цели, пропускаем
                if point in self.can_reach_goal:
                    continue
                
                # Если из этой точки можно попасть в точку, ведущую к цели
                for goal_point in self.can_reach_goal:
                    if point in self.reachability_graph and point != self.goal:
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки в множество
            self.can_reach_goal.update(new_points)
    
    def plan_path(self, max_iterations=100):
        """Основной алгоритм планирования пути"""
        for iteration in range(max_iterations):
            # Выбираем случайную начальную точку из уже посещенных
            if len(self.visited) > 1:  # Если есть другие точки, кроме начальной
                start_point = random.choice(list(self.visited))
            else:
                start_point = self.start
            
            # Запускаем случайные последовательности из выбранной точки
            reached_goal = self.run_random_sequence(start_point)
            
            # Обновляем множество точек, из которых можно достичь цель
            self.update_goal_reaching_points()
            
            # Если нашли путь к цели, возвращаем успех
            if self.goal in self.reachability_graph or reached_goal:
                print(f"Путь найден на итерации {iteration + 1}")
                return True
                
        print("Максимальное число итераций достигнуто, путь не найден")
        return False
    
    def reconstruct_path(self):
        """Восстанавливает путь от начала до цели, используя граф достижимости"""
        if self.goal not in self.reachability_graph:
            return None
            
        path = [self.goal]
        current = self.goal
        
        while current != self.start:
            current = self.reachability_graph[current]
            path.append(current)
            
        path.reverse()
        return path
    
    def visualize(self, show_all_paths=False):
        """Визуализирует лабиринт и найденный путь"""
        # Создаем копию лабиринта для визуализации
        maze_vis = self.maze.copy()
        
        # Маркируем посещенные клетки
        for i, j in self.visited:
            if maze_vis[i, j] == 0:  # Не перезаписываем начальную точку и цель
                maze_vis[i, j] = 4
        
        # Маркируем точки, из которых можно достичь цели
        for i, j in self.can_reach_goal:
            if maze_vis[i, j] == 4:  # Только для уже посещенных точек
                maze_vis[i, j] = 5
        
        # Убедимся, что начальная точка и цель имеют правильные метки
        maze_vis[self.start] = 2  # Перезаписываем старт
        maze_vis[self.goal] = 3   # Перезаписываем цель
        
        # Создаем цветовую карту
        colors = ['white', 'black', 'green', 'red', 'lightblue', 'yellow']
        cmap = ListedColormap(colors)
        
        # Подготовка легенды
        legend_elements = [
            mpatches.Patch(color='white', label='Свободный путь'),
            mpatches.Patch(color='black', label='Стена'),
            mpatches.Patch(color='green', label='Старт'),
            mpatches.Patch(color='red', label='Цель'),
            mpatches.Patch(color='lightblue', label='Посещенные клетки'),
            mpatches.Patch(color='yellow', label='Клетки, ведущие к цели')
        ]
        
        # Создаем график
        plt.figure(figsize=(12, 10))
        
        # Отображаем лабиринт
        plt.imshow(maze_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-0.5, self.size, 1), [])
        plt.yticks(np.arange(-0.5, self.size, 1), [])
        
        # Добавляем легенду
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        # Отображаем все пути
        if show_all_paths:
            for path in self.path_history:
                xs = [x for x, y in path]
                ys = [y for x, y in path]
                plt.plot(ys, xs, 'b-', alpha=0.2)  # Обратите внимание на обмен x и y из-за особенностей imshow
        
        # Отображаем найденный путь
        final_path = self.reconstruct_path()
        if final_path:
            xs = [x for x, y in final_path]
            ys = [y for x, y in final_path]
            plt.plot(ys, xs, 'r-', linewidth=2)  # Обратите внимание на обмен x и y из-за особенностей imshow
        
        # Дополнительно отмечаем старт и финиш для лучшей видимости
        plt.plot(self.start[1], self.start[0], 'go', markersize=10)
        plt.plot(self.goal[1], self.goal[0], 'ro', markersize=10)
        
        plt.title('Лабиринт и найденный путь')
        plt.tight_layout()
        plt.show()

# Пример использования
if __name__ == "__main__":
    # Создаем планировщик с лабиринтом 50x50
    planner = MazePathPlanner(size=50, obstacle_density=0.3)
    
    # Планируем путь
    success = planner.plan_path(max_iterations=150)
    
    # Визуализируем результат
    planner.visualize(show_all_paths=True)
    
    if success:
        path = planner.reconstruct_path()
        print(f"Длина найденного пути: {len(path)}")