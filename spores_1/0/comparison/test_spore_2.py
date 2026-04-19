import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import random
import time
from collections import deque
import heapq
import pickle
import os

def load_map_data(file_path="map_data.pkl"):
    """Загружает карту и координаты из файла"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        grid = data['grid']
        start_pos = data['start_pos']
        finish_pos = data['finish_pos']
        
        print(f"Карта загружена из файла {file_path}")
        print(f"Размер карты: {grid.shape}")
        print(f"Старт: {start_pos}, Финиш: {finish_pos}")
        print(f"Стен: {np.sum(grid == 1)}")
        
        return grid, start_pos, finish_pos
    except FileNotFoundError:
        print(f"Файл {file_path} не найден, будет сгенерирована новая карта")
        return None, None, None

def generate_map(size=(30, 30), wall_percent=30):
    """Генерирует карту с проходимым путем от старта до финиша"""
    height, width = size
    wall_count = int(wall_percent * width * height / 100)
    wall_count = min(wall_count, width * height - 2)
    
    grid = np.zeros((height, width), dtype=int)
    start_pos = (1, 1)
    finish_pos = (height - 2, width - 2)
    grid[start_pos] = 2  # Старт
    grid[finish_pos] = 3  # Финиш
    
    walls_placed = 0
    positions = [(x, y) for x in range(height) for y in range(width) 
                if (x, y) != start_pos and (x, y) != finish_pos]
    random.shuffle(positions)
    
    for pos in positions:
        if walls_placed >= wall_count:
            break
        grid[pos] = 1
        if not path_exists(grid, start_pos, finish_pos, height, width):
            grid[pos] = 0
        else:
            walls_placed += 1
    
    return grid, start_pos, finish_pos

def path_exists(grid, start_pos, finish_pos, height, width):
    """Проверяет существование пути между стартом и финишем с помощью BFS"""
    visited = set([start_pos])
    queue = deque([start_pos])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        current = queue.popleft()
        if current == finish_pos:
            return True
        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < height and 
                0 <= next_pos[1] < width and 
                grid[next_pos] != 1 and 
                next_pos not in visited):
                visited.add(next_pos)
                queue.append(next_pos)
    return False

def save_map_data(grid, start_pos, finish_pos, file_path="map_data.pkl"):
    """Сохраняет карту и координаты в файл"""
    data = {
        'grid': grid,
        'start_pos': start_pos,
        'finish_pos': finish_pos
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Карта сохранена в файл {file_path}")

class SporePathPlanner:
    def __init__(self, grid, start_pos, finish_pos, max_iterations=150):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start = start_pos
        self.goal = finish_pos
        
        # Граф достижимости
        self.reachability_graph = {self.start: None}
        
        # Точки, из которых можно достичь цели
        self.can_reach_goal = set([self.start])
        
        # Посещенные точки
        self.visited = set([self.start])
        
        # История путей
        self.path_history = []
        
        # Текущая спора
        self.current_spore = self.start
        
        # Список всех использованных спор
        self.all_spores = [self.start]
        
        # Счетчик итераций
        self.iteration = 0
        
        # Максимальное количество итераций
        self.max_iterations = max_iterations
        
        # Найден ли путь
        self.path_found = False
        
        # Для пошагового просмотра найденного пути
        self.path_display_step = 0
        self.final_path = None
        
    def is_valid_position(self, pos):
        """Проверяет, является ли позиция допустимой"""
        i, j = pos
        return 0 <= i < self.height and 0 <= j < self.width and self.grid[i, j] != 1
    
    def get_nearest_goal_reaching_point(self, point):
        """Находит ближайшую точку, из которой можно достичь цели"""
        if not self.can_reach_goal:
            return self.goal
        
        min_dist = float('inf')
        nearest_point = None
        
        for goal_point in self.can_reach_goal:
            dist = np.sqrt((point[0] - goal_point[0])**2 + (point[1] - goal_point[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = goal_point
                
        return nearest_point if nearest_point else self.goal
    
    def sample_action(self, current_pos, target_pos, std_dev=1.0):
        """Сэмплирует действие с матожиданием в направлении цели"""
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
        
        # Вероятности для каждого действия
        probs = []
        for action in possible_actions:
            action_vec = np.array([action[0] - current_pos[0], action[1] - current_pos[1]])
            dot_product = np.dot(action_vec, direction)
            prob = max(0.1, (dot_product + 1) / 2)  # Минимум 0.1 вероятности
            probs.append(prob)
        
        # Нормализуем вероятности
        probs = np.array(probs) / sum(probs)
        
        # Выбираем действие
        chosen_action = np.random.choice(len(possible_actions), p=probs)
        
        return possible_actions[chosen_action]
    
    def run_random_sequence(self, start_point, sequence_length=10, num_sequences=10):
        """Запускает несколько случайных последовательностей шагов из заданной точки"""
        # Адаптивная настройка параметров в зависимости от размера карты
        map_size = max(self.height, self.width)
        sequence_length = max(10, int(map_size * 0.5))
        num_sequences = max(10, int(map_size * 0.3))
        
        # Определяем цель
        if self.can_reach_goal:
            target = self.get_nearest_goal_reaching_point(start_point)
        else:
            target = self.goal
            
        current_paths = []
        reached_goal = False
        
        # Запускаем последовательности
        for _ in range(num_sequences):
            current = start_point
            path = [current]
            
            for _ in range(sequence_length):
                next_pos = self.sample_action(current, target)
                
                # Если недопустимая позиция, пробуем снова
                attempts = 0
                while not self.is_valid_position(next_pos) and attempts < 10:
                    next_pos = self.sample_action(current, target)
                    attempts += 1
                
                if self.is_valid_position(next_pos):
                    current = next_pos
                    path.append(current)
                    
                    # Добавляем в граф достижимости
                    if current not in self.reachability_graph:
                        self.reachability_graph[current] = start_point
                    
                    # Отмечаем как посещенную
                    self.visited.add(current)
                    
                    # Если достигли цели
                    if current == self.goal:
                        self.can_reach_goal.add(start_point)
                        reached_goal = True
                        break
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_goal:
                        self.can_reach_goal.add(start_point)
                        reached_goal = True
                        break
            
            # Сохраняем путь
            if len(path) > 1:
                current_paths.append(path)
        
        self.path_history.extend(current_paths)
        return current_paths, reached_goal
    
    def update_goal_reaching_points(self):
        """Обновляет множество точек, из которых можно достичь цель"""
        changed = True
        while changed:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам в графе достижимости
            for point, parent in self.reachability_graph.items():
                if point in self.can_reach_goal:
                    continue
                
                # Если из этой точки можно попасть в точку, ведущую к цели
                for goal_point in self.can_reach_goal:
                    if goal_point in self.reachability_graph and point != self.goal:
                        # Проверяем, что point является родителем для goal_point
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки
            self.can_reach_goal.update(new_points)
    
    def step(self):
        """Выполняет один шаг алгоритма"""
        if self.iteration >= self.max_iterations or self.path_found:
            return False, f"{'Путь найден!' if self.path_found else 'Достигнуто максимальное число итераций'}"
        
        # Выбираем новую спору случайно
        free_cells = [(i, j) for i in range(self.height) for j in range(self.width) 
                      if self.grid[i, j] == 0 or self.grid[i, j] == 4 or self.grid[i, j] == 5]
        
        if free_cells:
            self.current_spore = random.choice(free_cells)
        else:
            self.current_spore = self.start
        
        # Добавляем спору в список
        if self.current_spore not in self.all_spores:
            self.all_spores.append(self.current_spore)
        
        # Запускаем последовательности из выбранной споры
        paths, reached_goal = self.run_random_sequence(self.current_spore)
        
        # Обновляем точки, из которых можно достичь цель
        self.update_goal_reaching_points()
        
        # Проверяем, найден ли путь
        if self.goal in self.reachability_graph or reached_goal:
            self.path_found = True
            self.final_path = self.reconstruct_path()
            self.path_display_step = 0
            return True, f"Путь найден на итерации {self.iteration + 1}!"
        
        self.iteration += 1
        return True, f"Итерация {self.iteration}/{self.max_iterations}, спора: {self.current_spore}, путей из неё: {len(paths)}"
    
    def reconstruct_path(self):
        """Восстанавливает путь от начала до цели"""
        if self.goal not in self.reachability_graph:
            # Пробуем A* если необходимо
            return self._astar_search(self.grid, self.start, self.goal)
            
        path = [self.goal]
        current = self.goal
        
        while current != self.start:
            current = self.reachability_graph[current]
            path.append(current)
            
        path.reverse()
        return path
    
    def _astar_search(self, grid, start, goal):
        """Поиск пути с помощью A*"""
        # Эвристика - манхэттенское расстояние
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Проверка допустимости позиции
        def is_valid(pos):
            x, y = pos
            return (0 <= x < self.height and 
                    0 <= y < self.width and 
                    grid[x, y] != 1)
        
        # Направления движения
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Открытый список
        open_list = []
        heapq.heappush(open_list, (heuristic(start), 0, start))
        
        # Отслеживание лучшего пути
        came_from = {start: None}
        
        # Стоимость пути от старта
        g_score = {start: 0}
        
        while open_list:
            _, _, current = heapq.heappop(open_list)
            
            # Если достигли цели
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Проверяем соседей
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if not is_valid(next_pos):
                    continue
                
                # Рассчитываем новую стоимость
                tentative_g_score = g_score[current] + 1
                
                # Если нашли лучший путь
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score = tentative_g_score + heuristic(next_pos)
                    heapq.heappush(open_list, (f_score, tentative_g_score, next_pos))
        
        # Путь не существует
        return []
    
    def find_path(self, max_steps=None):
        """Находит путь от старта до финиша, запуская алгоритм"""
        if max_steps is None:
            max_steps = self.max_iterations
            
        for _ in range(max_steps):
            continue_sim, message = self.step()
            if not continue_sim or self.path_found:
                print(message)
                break
        
        return self.final_path, self.visited, self.all_spores, self.can_reach_goal
    
    def visualize_result(self):
        """Визуализирует результат алгоритма"""
        plt.figure(figsize=(12, 10))
        
        # Создаем копию лабиринта для визуализации
        maze_vis = self.grid.copy()
        
        # Маркируем посещенные клетки
        for i, j in self.visited:
            if maze_vis[i, j] == 0:  # Не перезаписываем начальную точку и цель
                maze_vis[i, j] = 4
        
        # Маркируем точки, из которых можно достичь цели
        for i, j in self.can_reach_goal:
            if maze_vis[i, j] == 4:  # Только для уже посещенных точек
                maze_vis[i, j] = 5
        
        # Убедимся, что начальная точка и цель имеют правильные метки
        maze_vis[self.start] = 2
        maze_vis[self.goal] = 3
                
        # Создаем цветовую карту
        colors = ['white', 'black', 'green', 'red', 'lightblue', 'yellow', 'orange']
        cmap = ListedColormap(colors)
        
        # Отображаем лабиринт
        plt.imshow(maze_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        # Отображаем споры
        for spore in self.all_spores:
            if spore != self.start and spore != self.goal:
                plt.plot(spore[1], spore[0], 'o', color='orange', markersize=6)
        
        # Отображаем найденный путь
        if self.final_path:
            path_y = [y for x, y in self.final_path]
            path_x = [x for x, y in self.final_path]
            plt.plot(path_y, path_x, 'r-', linewidth=3)
        
        # Дополнительно отмечаем старт и цель для лучшей видимости
        plt.plot(self.start[1], self.start[0], 'go', markersize=10)
        plt.plot(self.goal[1], self.goal[0], 'ro', markersize=10)
        
        # Легенда
        legend_elements = [
            mpatches.Patch(color='white', label='Свободный путь'),
            mpatches.Patch(color='black', label='Стена'),
            mpatches.Patch(color='green', label='Старт'),
            mpatches.Patch(color='red', label='Цель'),
            mpatches.Patch(color='lightblue', label='Посещенные клетки'),
            mpatches.Patch(color='yellow', label='Клетки, ведущие к цели'),
            mpatches.Patch(color='orange', label='Споры')
        ]
        plt.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, 1.15), ncol=3)
        
        plt.title(f'Результат поиска пути (Итерация {self.iteration}/{self.max_iterations})')
        plt.tight_layout()
        plt.show()

def run_demo(use_file=False, file_path="map_data.pkl"):
    """Запускает демонстрацию алгоритма"""
    if use_file:
        # Пытаемся загрузить карту из файла
        map_data = load_map_data(file_path)
        if map_data[0] is not None:
            grid, start_pos, finish_pos = map_data
        else:
            # Если файл не найден, генерируем новую карту
            grid, start_pos, finish_pos = generate_map()
            # И сохраняем её
            save_map_data(grid, start_pos, finish_pos, file_path)
    else:
        # Генерируем новую карту
        grid, start_pos, finish_pos = generate_map(size=(30, 30), wall_percent=30)
    
    # Визуализируем исходную карту
    plt.figure(figsize=(10, 8))
    colors = ['white', 'black', 'green', 'red']
    cmap = ListedColormap(colors)
    plt.imshow(grid, cmap=cmap)
    plt.title("Исходная карта")
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    legend_elements = [
        mpatches.Patch(color='white', label='Свободный путь'),
        mpatches.Patch(color='black', label='Стена'),
        mpatches.Patch(color='green', label='Старт'),
        mpatches.Patch(color='red', label='Цель')
    ]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.tight_layout()
    plt.show()
    
    # Создаем планировщик пути
    max_iterations = int(np.prod(grid.shape) * 0.25)  # Адаптивное количество итераций
    planner = SporePathPlanner(grid, start_pos, finish_pos, max_iterations=max_iterations)
    
    print(f"Запускаем алгоритм Spore на карте размером {grid.shape[0]}x{grid.shape[1]}")
    print(f"Старт: {start_pos}, Цель: {finish_pos}")
    print(f"Максимальное количество итераций: {max_iterations}")
    
    # Запускаем поиск пути
    start_time = time.time()
    path, visited, spores, can_reach_goal = planner.find_path()
    end_time = time.time()
    
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")
    print(f"Итераций: {planner.iteration}")
    print(f"Посещенных клеток: {len(visited)}")
    print(f"Спор: {len(spores)}")
    
    if path:
        print(f"Путь найден! Длина пути: {len(path)}")
    else:
        print("Путь не найден")
    
    # Визуализируем результат
    planner.visualize_result()
    
    return planner

if __name__ == "__main__":
    # Запускаем с генерацией новой карты, так как map_data.pkl может отсутствовать
    use_file = False  # Изменить на True, если файл map_data.pkl существует
    planner = run_demo(use_file=use_file)