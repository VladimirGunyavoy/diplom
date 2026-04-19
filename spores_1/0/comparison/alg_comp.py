import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
import heapq
import pickle
import os
from collections import deque
import random

def load_map_data(file_path="map_data.pkl"):
    """Загружает карту и координаты из файла"""
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

def visualize_map(grid, start_pos, finish_pos, a_star_path=None, spore_path=None, 
                 a_star_explored=None, spore_explored=None, spores=None, title=None):
    """Визуализирует карту с путями обоих алгоритмов"""
    plt.figure(figsize=(12, 10))
    
    # Создаем копию карты для визуализации
    grid_vis = grid.copy()
    
    # Отмечаем исследованные клетки A*
    if a_star_explored:
        for x, y in a_star_explored:
            if grid_vis[x, y] == 0:  # Не перезаписываем старт, финиш и стены
                grid_vis[x, y] = 4  # 4 - исследованные клетки A*
    
    # Отмечаем исследованные клетки Spore
    if spore_explored:
        for x, y in spore_explored:
            if grid_vis[x, y] == 0:  # Не перезаписываем старт, финиш и стены
                grid_vis[x, y] = 5  # 5 - исследованные клетки Spore
            elif grid_vis[x, y] == 4:  # Если клетка уже исследована A*
                grid_vis[x, y] = 6  # 6 - исследованные обоими алгоритмами
    
    # Отмечаем споры
    if spores:
        for spore in spores:
            x, y = spore[0], spore[1]
            if grid_vis[x, y] != 2 and grid_vis[x, y] != 3:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 7  # 7 - спора
    
    # Отмечаем путь A*
    if a_star_path:
        for x, y in a_star_path:
            if (x, y) != start_pos and (x, y) != finish_pos:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 8  # 8 - путь A*
    
    # Отмечаем путь Spore
    if spore_path:
        for x, y in spore_path:
            if (x, y) != start_pos and (x, y) != finish_pos:  # Не перезаписываем старт и финиш
                if grid_vis[x, y] == 8:  # Если клетка уже в пути A*
                    grid_vis[x, y] = 9  # 9 - общий путь
                else:
                    grid_vis[x, y] = 10  # 10 - путь Spore
    
    # Убедимся, что старт и финиш имеют правильные значения
    grid_vis[start_pos] = 2
    grid_vis[finish_pos] = 3
    
    # Создаем цветовую карту
    colors = [
        'white',      # 0: пустота
        'darkgreen',  # 1: стена
        'blue',       # 2: старт
        'red',        # 3: финиш
        'lightblue',  # 4: исследованные A*
        'lightgreen', # 5: исследованные Spore
        'plum',       # 6: исследованные обоими
        'yellow',     # 7: спора
        'orange',     # 8: путь A*
        'purple',     # 9: общий путь
        'cyan',       # 10: путь Spore
    ]
    cmap = ListedColormap(colors)
    
    # Отображаем карту с сеткой
    plt.imshow(grid_vis, cmap=cmap)
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    # Добавляем легенду
    legend_elements = [
        mpatches.Patch(color='white', label='Свободный путь'),
        mpatches.Patch(color='darkgreen', label='Стена'),
        mpatches.Patch(color='blue', label='Старт'),
        mpatches.Patch(color='red', label='Финиш'),
        mpatches.Patch(color='lightblue', label='Исследовано A*'),
        mpatches.Patch(color='lightgreen', label='Исследовано Spore'),
        mpatches.Patch(color='plum', label='Исследовано обоими'),
        mpatches.Patch(color='yellow', label='Спора'),
        mpatches.Patch(color='orange', label='Путь A*'),
        mpatches.Patch(color='purple', label='Общий путь'),
        mpatches.Patch(color='cyan', label='Путь Spore')
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    # Заголовок
    if title:
        plt.title(title)
    else:
        plt.title(f'Карта {grid.shape[0]}x{grid.shape[1]}')
    
    plt.tight_layout()
    plt.show()

class AStarPathfinder:
    """Реализация алгоритма A* для поиска пути"""
    
    def __init__(self, grid, start_pos, finish_pos, debug=False):
        self.grid = grid
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.height, self.width = grid.shape
        self.explored_cells = set()  # Множество для отслеживания исследованных клеток
        self.debug = debug
    
    def heuristic(self, a, b):
        """
        Вычисляет эвристику для A*.
        Используется манхэттенское расстояние.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos):
        """Возвращает соседние позиции, которые не являются стенами"""
        x, y = pos
        neighbors = []
        
        # Проверяем четыре направления (право, низ, лево, верх)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Проверяем, что сосед находится в пределах сетки и не является стеной
            if (0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx, ny] != 1):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self):
        """Находит путь от старта до финиша с помощью A*"""
        start_time = time.time()
        
        # Инициализируем открытый и закрытый списки
        open_set = []
        heapq.heappush(open_set, (0, self.start_pos))  # (f, position)
        
        # Для каждой позиции хранится информация о предшествующей позиции и стоимости пути
        came_from = {}
        g_score = {self.start_pos: 0}  # Стоимость пути от начала до указанной позиции
        f_score = {self.start_pos: self.heuristic(self.start_pos, self.finish_pos)}  # Оценочная полная стоимость пути
        
        # Множество для быстрой проверки наличия элемента в открытом списке
        open_set_hash = {self.start_pos}
        
        # Добавляем стартовую позицию в список исследованных
        self.explored_cells.add(self.start_pos)
        
        nodes_explored = 0  # Счетчик исследованных узлов
        
        while open_set:
            # Извлекаем позицию с наименьшей оценкой f
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            nodes_explored += 1
            
            # Если достигли финиша, восстанавливаем путь
            if current == self.finish_pos:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start_pos)
                path.reverse()
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                if self.debug:
                    print(f"A* исследовал {nodes_explored} узлов ({len(self.explored_cells)} уникальных клеток)")
                
                return path, elapsed_time, nodes_explored, len(self.explored_cells)
            
            # Проверяем всех соседей текущей позиции
            for neighbor in self.get_neighbors(current):
                # Добавляем соседа в список исследованных клеток
                self.explored_cells.add(neighbor)
                
                # Стоимость пути до соседа через текущую позицию
                temp_g_score = g_score[current] + 1
                
                # Если найден лучший путь до соседа
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + self.heuristic(neighbor, self.finish_pos)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # Если путь не найден
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.debug:
            print(f"A* исследовал {nodes_explored} узлов, но путь не найден")
        return None, elapsed_time, nodes_explored, len(self.explored_cells)

class ImprovedSporeAgent:
    """Улучшенный агент поиска пути с использованием алгоритма Spore."""
    
    def __init__(self, grid, start_pos, finish_pos, max_iterations=100, debug=False):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.max_iterations = max_iterations
        self.debug = debug
        
        # Список спор (позиция, может_достичь_финиша, помечена_для_удаления)
        self.spores = [(start_pos[0], start_pos[1], True, False)]
        
        # Множество клеток, из которых можно достичь финиша
        self.can_reach_finish = {start_pos}
        
        # Пути агентов
        self.agent_paths = []
        
        # Успешные пути
        self.successful_paths = []
        
        # Граф достижимости (ключ: точка, значение: родительская точка)
        self.reachability_graph = {start_pos: None}
        
        # Граф обратной достижимости (для более эффективного восстановления пути)
        # (ключ: точка, значение: список точек, достижимых из ключа)
        self.reverse_graph = {}
        
        # Итоговый путь
        self.final_path = []
        
        # Счетчик итераций
        self.iteration = 0
        
        # Флаг найденного пути
        self.path_found = False
        
        # Исследованные клетки
        self.explored_cells = set([start_pos])
        
        # Расстояние между стартом и финишем для адаптивных параметров
        self.distance_start_finish = np.sqrt((start_pos[0] - finish_pos[0])**2 + (start_pos[1] - finish_pos[1])**2)
        
        if self.debug:
            print(f"Инициализирован ImprovedSporeAgent:")
            print(f"  - Размер карты: {self.height}x{self.width}")
            print(f"  - Старт: {self.start_pos}")
            print(f"  - Финиш: {self.finish_pos}")
            print(f"  - Расстояние между стартом и финишем: {self.distance_start_finish:.2f}")
    
    # Все методы экземпляра ImprovedSporeAgent сохранены, но для краткости здесь не приведены
    # ...
    
    def distance(self, pos1, pos2):
        """Вычисляет евклидово расстояние между двумя позициями"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid_position(self, pos):
        """Проверяет, является ли позиция допустимой"""
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != 1)  # Не стена
    
    def get_nearest_reachable_point(self, pos):
        """Находит ближайшую точку, из которой можно достичь финиш"""
        if not self.can_reach_finish:
            return self.finish_pos
        
        min_dist = float('inf')
        nearest_point = None
        
        # Проверяем расстояние до финиша, если финиш достижим
        if self.finish_pos in self.can_reach_finish:
            min_dist = self.distance(pos, self.finish_pos)
            nearest_point = self.finish_pos
        
        # Проверяем расстояние до достижимых спор
        for goal_point in self.can_reach_finish:
            dist = self.distance(pos, goal_point)
            if dist < min_dist:
                min_dist = dist
                nearest_point = goal_point
        
        return nearest_point if nearest_point else self.finish_pos
    
    def generate_spore_position(self):
        """Генерирует позицию для новой споры"""
        if self.debug:
            print("Генерация новой споры...")
        
        # Оценим, на каком этапе мы находимся
        exploration_ratio = min(1.0, len(self.explored_cells) / (self.height * self.width * 0.3))
        
        # Определим приоритеты размещения спор в зависимости от этапа исследования
        priorities = {
            'near_finish': 0.4 - exploration_ratio * 0.2,
            'near_start': 0.2 - exploration_ratio * 0.1,
            'near_reachable': 0.3 + exploration_ratio * 0.2,
            'random': 0.1 + exploration_ratio * 0.1
        }
        
        # Сначала пытаемся размещать споры на прямой линии между стартом и финишем
        if random.random() < 0.3 and self.iteration < self.max_iterations * 0.3:
            if self.debug:
                print("  Пытаемся разместить на линии между стартом и финишем")
            
            t = random.uniform(0.2, 0.8)  # Параметр интерполяции
            x = int(self.start_pos[0] + t * (self.finish_pos[0] - self.start_pos[0]))
            y = int(self.start_pos[1] + t * (self.finish_pos[1] - self.start_pos[1]))
            new_pos = (x, y)
            
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                if self.debug:
                    print(f"  Создана спора на линии между стартом и финишем: {new_pos}")
                return new_pos
        
        # Выбираем стратегию размещения на основе приоритетов
        r = random.random()
        
        # Размещение рядом с финишем
        if r < priorities['near_finish']:
            radius = max(5, min(10, int(self.distance_start_finish * 0.2)))
            attempts = 0
            while attempts < 30:
                offset_x = int(np.random.normal(0, radius * 0.5))
                offset_y = int(np.random.normal(0, radius * 0.5))
                new_pos = (self.finish_pos[0] + offset_x, self.finish_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                attempts += 1
        
        # Размещение рядом со стартом
        elif r < priorities['near_finish'] + priorities['near_start']:
            radius = max(5, min(10, int(self.distance_start_finish * 0.2)))
            attempts = 0
            while attempts < 20:
                offset_x = int(np.random.normal(0, radius * 0.5))
                offset_y = int(np.random.normal(0, radius * 0.5))
                new_pos = (self.start_pos[0] + offset_x, self.start_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                attempts += 1
        
        # Размещение рядом с достижимыми спорами
        elif r < priorities['near_finish'] + priorities['near_start'] + priorities['near_reachable']:
            reachable_spores = [(x, y) for x, y, can_reach, _ in self.spores if can_reach and not _]
            if reachable_spores:
                # Выбираем с учетом расстояния до финиша
                weights = []
                for spore_pos in reachable_spores:
                    dist_to_finish = self.distance(spore_pos, self.finish_pos)
                    weight = 1.0 / max(1.0, dist_to_finish)
                    weights.append(weight)
                
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    base_spore = random.choices(reachable_spores, weights=weights, k=1)[0]
                else:
                    base_spore = random.choice(reachable_spores)
                
                radius = max(5, min(10, int(self.distance_start_finish * 0.15)))
                attempts = 0
                while attempts < 30:
                    offset_x = int(np.random.normal(0, radius))
                    offset_y = int(np.random.normal(0, radius))
                    new_pos = (base_spore[0] + offset_x, base_spore[1] + offset_y)
                    
                    if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                        return new_pos
                    attempts += 1
        
        # Случайное размещение в неисследованной области
        unexplored = [(x, y) for x in range(self.height) for y in range(self.width) 
                     if (x, y) not in self.explored_cells and self.is_valid_position((x, y))]
        
        if unexplored and random.random() < 0.7:
            attempts = 0
            max_attempts = min(50, len(unexplored))
            while attempts < max_attempts:
                new_pos = random.choice(unexplored)
                if new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                attempts += 1
        
        # Полностью случайное размещение
        attempts = 0
        while attempts < 50:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            new_pos = (x, y)
            
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                return new_pos
            attempts += 1
        
        return None

    def sample_action(self, current_pos, target_pos):
        """Сэмплирует действие в направлении цели с шумом"""
        # Направление к цели
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        
        # Определяем величину шума
        exploration_stage = min(1.0, self.iteration / self.max_iterations)
        noise_std_dev = 0.8 - exploration_stage * 0.5  # От 0.8 до 0.3
        
        # Нормализуем
        length = max(1e-10, np.sqrt(dir_x**2 + dir_y**2))
        dir_x /= length
        dir_y /= length
        
        # Добавляем шум
        dir_x += np.random.normal(0, noise_std_dev)
        dir_y += np.random.normal(0, noise_std_dev)
        
        # Определяем направление
        if abs(dir_x) > abs(dir_y):
            if dir_x > 0:
                return (current_pos[0] + 1, current_pos[1])  # Вниз
            else:
                return (current_pos[0] - 1, current_pos[1])  # Вверх
        else:
            if dir_y > 0:
                return (current_pos[0], current_pos[1] + 1)  # Вправо
            else:
                return (current_pos[0], current_pos[1] - 1)  # Влево

    def run_random_sequence(self, start_point, sequence_length=None, num_sequences=None):
        """Запускает несколько случайных последовательностей из точки"""
        if sequence_length is None:
            sequence_length = max(30, int(self.distance_start_finish * 1.5))
        
        if num_sequences is None:
            num_sequences = max(10, int(self.distance_start_finish * 0.3))
        
        # Определяем цель
        target = self.get_nearest_reachable_point(start_point)
        
        current_paths = []
        reached_goal = False
        
        # Запускаем последовательности
        for seq in range(num_sequences):
            current = start_point
            path = [current]
            
            for step in range(sequence_length):
                # Сэмплируем следующую позицию
                next_pos = self.sample_action(current, target)
                
                # Если позиция недопустима, пробуем еще раз
                attempts = 0
                while not self.is_valid_position(next_pos) and attempts < 10:
                    next_pos = self.sample_action(current, target)
                    attempts += 1
                
                # Если нашли допустимую позицию
                if self.is_valid_position(next_pos):
                    current = next_pos
                    path.append(current)
                    
                    # Добавляем точку в граф достижимости
                    if current not in self.reachability_graph:
                        self.reachability_graph[current] = start_point
                        
                        # Добавляем в обратный граф
                        if start_point not in self.reverse_graph:
                            self.reverse_graph[start_point] = []
                        if current not in self.reverse_graph[start_point]:
                            self.reverse_graph[start_point].append(current)
                    
                    # Отмечаем точку как исследованную
                    self.explored_cells.add(current)
                    
                    # Если достигли финиша
                    if current == self.finish_pos:
                        self.can_reach_finish.add(start_point)
                        reached_goal = True
                        self.successful_paths.append(path)
                        break
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_finish:
                        self.can_reach_finish.add(start_point)
                        reached_goal = True
                        self.successful_paths.append(path)
                        break
                else:
                    # Если не нашли допустимую позицию, заканчиваем последовательность
                    break
            
            # Сохраняем путь
            if len(path) > 1:
                current_paths.append(path)
        
        self.agent_paths.extend(current_paths)
        return current_paths, reached_goal

    def update_reachable_points(self):
        """Обновляет множество точек, из которых можно достичь цель"""
        initial_count = len(self.can_reach_finish)
        
        changed = True
        while changed:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам графа достижимости
            for point, parent in self.reachability_graph.items():
                if point in self.can_reach_finish:
                    continue
                
                # Если из этой точки можно попасть в достижимую точку
                for goal_point in self.can_reach_finish:
                    if goal_point in self.reachability_graph and point != self.finish_pos:
                        # Проверяем через обратный граф, если он существует
                        if point in self.reverse_graph and goal_point in self.reverse_graph[point]:
                            new_points.add(point)
                            changed = True
                            break
                        # Иначе проверяем через прямой граф
                        elif self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки
            self.can_reach_finish.update(new_points)
        
        # Обновляем споры, которые могут достичь финиш
        for i, (x, y, _, marked) in enumerate(self.spores):
            if (x, y) in self.can_reach_finish and not marked:
                self.spores[i] = (x, y, True, marked)
    
    def _astar_search(self, grid, start, goal):
        """Поиск пути с помощью A*"""
        # Эвристика - манхэттенское расстояние
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Функция проверки допустимости позиции
        def is_valid(pos):
            x, y = pos
            return (0 <= x < self.height and 
                    0 <= y < self.width and 
                    grid[x, y] != 1)  # Не стена
        
        # Направления движения (вверх, вниз, влево, вправо)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Открытый список (очередь с приоритетом)
        open_list = []
        heapq.heappush(open_list, (heuristic(start), 0, start))
        
        # Словарь для отслеживания лучшего пути к каждой позиции
        came_from = {start: None}
        
        # Стоимость пути от старта до каждой позиции
        g_score = {start: 0}
        
        while open_list:
            # Извлекаем позицию с наименьшей оценкой f(n) = g(n) + h(n)
            _, _, current = heapq.heappop(open_list)
            
            # Если достигли цели, восстанавливаем путь
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            # Проверяем всех соседей
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                # Проверяем, что сосед находится в допустимой позиции
                if not is_valid(next_pos):
                    continue
                
                # Рассчитываем новую стоимость пути
                tentative_g_score = g_score[current] + 1
                
                # Если мы нашли лучший путь до этой позиции
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score = tentative_g_score + heuristic(next_pos)
                    heapq.heappush(open_list, (f_score, tentative_g_score, next_pos))
        
        # Если очередь пуста и мы не достигли цели, путь не существует
        return []
    
    def _build_path_from_graph(self, start, end):
        """Строит путь от start до end используя граф достижимости"""
        if start == end:
            return [start]
        
        # Используем BFS для поиска пути в графе
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            node, path = queue.popleft()
            
            # Если достигли конечной точки
            if node == end:
                return path
            
            # Проверяем соседей через обратный граф
            if node in self.reverse_graph:
                for neighbor in self.reverse_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            # Проверяем родителя через прямой граф
            if node in self.reachability_graph and self.reachability_graph[node] not in visited:
                parent = self.reachability_graph[node]
                if parent:  # Учитываем, что у начальной точки нет родителя (None)
                    visited.add(parent)
                    queue.append((parent, path + [parent]))
        
        # Если путь не найден через связный граф, пробуем A*
        return self._astar_search(self.grid, start, end)
    
    def reconstruct_path(self):
        """Восстанавливает путь от старта до финиша"""
        # Проверяем наличие прямого пути через A*
        astar_path = self._astar_search(self.grid, self.start_pos, self.finish_pos)
        if astar_path:
            return astar_path

        # Если финиш не в графе достижимости, ищем ближайшую к финишу достижимую точку
        if self.finish_pos not in self.reachability_graph:
            closest_point = None
            min_dist = float('inf')
            
            for point in self.can_reach_finish:
                dist = self.distance(point, self.finish_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point
            
            if closest_point:
                # Путь от старта до ближайшей точки + путь от ближайшей точки до финиша
                path_to_closest = self._build_path_from_graph(self.start_pos, closest_point)
                path_to_finish = self._astar_search(self.grid, closest_point, self.finish_pos)
                
                if path_to_closest and path_to_finish:
                    # Объединяем пути (без дублирования точки)
                    combined_path = path_to_closest[:-1] + path_to_finish
                    return combined_path
        
        # Если финиш в графе, пытаемся восстановить путь из графа
        if self.finish_pos in self.reachability_graph:
            path = self._build_path_from_graph(self.start_pos, self.finish_pos)
            if path and len(path) > 1:
                return path
        
        # Если все методы не сработали, ищем путь по частям через ключевые точки
        # Находим все достижимые споры
        reachable_spores = [(x, y) for x, y, reachable, _ in self.spores if reachable]
        
        # Сортируем по расстоянию до финиша (ближайшие в начале)
        reachable_spores.sort(key=lambda s: self.distance(s, self.finish_pos))
        
        for spore in reachable_spores:
            if spore != self.start_pos:
                # Пробуем построить путь через эту спору
                path_to_spore = self._build_path_from_graph(self.start_pos, spore)
                path_from_spore = self._astar_search(self.grid, spore, self.finish_pos)
                
                if path_to_spore and path_from_spore:
                    # Объединяем пути
                    combined_path = path_to_spore[:-1] + path_from_spore
                    return combined_path
        
        # Если не удалось восстановить путь, возвращаем None
        return None

    def step(self):
        """Выполняет один шаг алгоритма"""
        # Если путь уже найден, немедленно прекращаем работу
        if self.path_found:
            return False, "Путь уже найден"
        
        # Если достигнуто максимальное число итераций
        if self.iteration >= self.max_iterations:
            return False, "Достигнуто максимальное число итераций"
        
        # Генерируем новую спору
        new_spore_pos = self.generate_spore_position()
        
        if new_spore_pos:
            # Добавляем спору
            is_reachable = new_spore_pos in self.can_reach_finish
            self.spores.append((new_spore_pos[0], new_spore_pos[1], is_reachable, False))
            
            # Запускаем агентов
            paths, reached_goal = self.run_random_sequence(new_spore_pos)
            
            # Обновляем достижимые точки
            self.update_reachable_points()
            
            # Проверяем, найден ли путь до финиша
            if self.finish_pos in self.reachability_graph or self.finish_pos in self.can_reach_finish or reached_goal:
                # Пытаемся восстановить путь
                self.final_path = self.reconstruct_path()
                
                if self.final_path and len(self.final_path) > 1:
                    self.path_found = True
                    if self.debug:
                        print(f"Путь найден на итерации {self.iteration + 1}! Длина пути: {len(self.final_path)}")
                    return True, f"Путь найден на итерации {self.iteration + 1}! Длина пути: {len(self.final_path)}"
                else:
                    # Если путь слишком короткий или не восстановлен, продолжаем
                    return True, f"Финиш достижим, но путь не восстановлен. Продолжаем поиск."
            
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}"
        else:
            # Если не удалось создать новую спору, пробуем ещё раз с другими параметрами
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}. Не удалось создать новую спору."

    def find_path(self, visualize=False, visualize_steps=False, visualize_interval=5):
        """Находит путь от старта до финиша"""
        start_time = time.time()
        
        if self.debug:
            print("Начинаю поиск пути с помощью алгоритма Spore...")
        
        # Выполняем итерации алгоритма
        continue_sim = True
        while continue_sim and self.iteration < self.max_iterations:
            # Выполняем шаг алгоритма
            continue_sim, message = self.step()
            if self.debug:
                print(f"  [Итерация {self.iteration}] {message}")
            
            # Если путь найден, выходим из цикла
            if self.path_found:
                break
        
        time_elapsed = time.time() - start_time
        
        # Финальная проверка для восстановления пути
        if not self.final_path or len(self.final_path) < 2:
            self.update_reachable_points()
            self.final_path = self.reconstruct_path()
            
            # Последний шанс - прямой A*
            if not self.final_path:
                self.final_path = self._astar_search(self.grid, self.start_pos, self.finish_pos)
        
        # Результаты
        if self.final_path and len(self.final_path) > 1:
            if self.debug:
                print(f"Путь найден за {time_elapsed:.4f} секунд!")
                print(f"Длина пути: {len(self.final_path)}")
                print(f"Итераций: {self.iteration}")
                print(f"Исследовано клеток: {len(self.explored_cells)}")
                print(f"Создано спор: {len(self.spores)}")
            self.path_found = True
        else:
            if self.debug:
                print(f"Путь не найден за {time_elapsed:.4f} секунд.")
                print(f"Итераций: {self.iteration}")
                print(f"Исследовано клеток: {len(self.explored_cells)}")
                print(f"Создано спор: {len(self.spores)}")
        
        # Визуализируем результат
        if visualize:
            print("Визуализация итогового результата...")
            visualize_map(
                self.grid, 
                self.start_pos, 
                self.finish_pos, 
                spore_path=self.final_path, 
                spore_explored=list(self.explored_cells), 
                spores=self.spores,
                title=f"Результат Spore (Итераций: {self.iteration})"
            )
        
        return self.final_path, time_elapsed, self.iteration, len(self.explored_cells), len(self.spores)

def run_comparison(file_path="map_data.pkl", visualize_result=True):
    """Запускает сравнение алгоритмов A* и Spore"""
    # Загружаем карту
    grid, start_pos, finish_pos = load_map_data(file_path)
    
    # Запускаем A*
    print("\n========== Запуск алгоритма A* ==========")
    astar = AStarPathfinder(grid, start_pos, finish_pos)
    astar_path, astar_time, astar_nodes, astar_cells = astar.find_path()
    
    if astar_path:
        print(f"A* нашел путь за {astar_time:.6f} секунд")
        print(f"Длина пути A*: {len(astar_path)}")
        print(f"Исследовано узлов A*: {astar_nodes}")
        print(f"Исследовано уникальных клеток A*: {astar_cells}")
    else:
        print(f"A* не нашел путь за {astar_time:.6f} секунд")
    
    # Запускаем Spore
    print("\n========== Запуск алгоритма Spore ==========")
    spore = ImprovedSporeAgent(grid, start_pos, finish_pos, max_iterations=50)
    spore_path, spore_time, spore_iterations, spore_cells, spore_count = spore.find_path()
    
    if spore_path:
        print(f"Spore нашел путь за {spore_time:.6f} секунд")
        print(f"Длина пути Spore: {len(spore_path)}")
        print(f"Итераций Spore: {spore_iterations}")
        print(f"Исследовано клеток Spore: {spore_cells}")
        print(f"Создано спор: {spore_count}")
    else:
        print(f"Spore не нашел путь за {spore_time:.6f} секунд")
    
    # Сравнение результатов
    print("\n========== Сравнение результатов ==========")
    if astar_path and spore_path:
        length_diff = len(spore_path) - len(astar_path)
        length_ratio = len(spore_path) / len(astar_path) if len(astar_path) > 0 else float('inf')
        time_ratio = spore_time / astar_time if astar_time > 0 else float('inf')
        cells_ratio = spore_cells / astar_cells if astar_cells > 0 else float('inf')
        
        print(f"Отношение длин путей (Spore/A*): {length_ratio:.2f} ({length_diff:+d} клеток)")
        print(f"Отношение времени выполнения (Spore/A*): {time_ratio:.2f}")
        print(f"Отношение исследованных клеток (Spore/A*): {cells_ratio:.2f}")
    
    # Визуализация сравнения
    if visualize_result:
        print("\nВизуализация сравнения алгоритмов...")
        visualize_map(
            grid, 
            start_pos, 
            finish_pos, 
            a_star_path=astar_path, 
            spore_path=spore_path, 
            a_star_explored=list(astar.explored_cells), 
            spore_explored=list(spore.explored_cells), 
            spores=spore.spores,
            title=f"Сравнение A* и Spore"
        )
    
    return {
        "astar": {
            "path": astar_path,
            "time": astar_time,
            "nodes": astar_nodes,
            "cells": astar_cells
        },
        "spore": {
            "path": spore_path,
            "time": spore_time,
            "iterations": spore_iterations,
            "cells": spore_cells,
            "spores": spore_count
        }
    }

if __name__ == "__main__":
    # Запускаем сравнение
    results = run_comparison(visualize_result=True)
    print("\nГотово!")