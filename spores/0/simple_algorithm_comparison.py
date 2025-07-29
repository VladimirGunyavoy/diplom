import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import heapq
from collections import deque

# Класс для генерации карты
class MapGenerator:
    def __init__(self, size=(30, 30), wall_count=None):
        self.height, self.width = size
        if wall_count is None:
            self.wall_count = int(0.3 * self.width * self.height)
        else:
            self.wall_count = min(wall_count, self.width * self.height - 2)
    
    def generate_map(self):
        # Создаем пустую карту
        grid = np.zeros((self.height, self.width), dtype=int)
        
        # Выбираем позицию старта (верхний левый угол)
        start_pos = (1, 1)
        
        # Выбираем позицию финиша (нижний правый угол)
        finish_pos = (self.height - 2, self.width - 2)
        
        # Размещаем старт и финиш
        grid[start_pos] = 2  # Старт
        grid[finish_pos] = 3  # Финиш
        
        # Размещаем стены
        walls_placed = 0
        positions = [(x, y) for x in range(self.height) for y in range(self.width) 
                     if (x, y) != start_pos and (x, y) != finish_pos]
        random.shuffle(positions)
        
        for pos in positions:
            if walls_placed >= self.wall_count:
                break
            grid[pos] = 1  # Стена
            
            # Проверяем, существует ли путь
            if not self._path_exists(grid, start_pos, finish_pos):
                grid[pos] = 0  # Если путь не существует, убираем стену
            else:
                walls_placed += 1
        
        return grid, start_pos, finish_pos
    
    def _path_exists(self, grid, start_pos, finish_pos):
        # Проверяем, существует ли путь между стартом и финишем с помощью BFS
        visited = set([start_pos])
        queue = deque([start_pos])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Вправо, вниз, влево, вверх
        
        while queue:
            current = queue.popleft()
            
            if current == finish_pos:
                return True
            
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    grid[next_pos] != 1 and  # Не стена
                    next_pos not in visited):
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def visualize_map(self, grid, start_pos, finish_pos, path=None, title=None):
        plt.figure(figsize=(10, 10))
        
        # Создаем копию карты для визуализации
        grid_vis = grid.copy()
        
        # Если путь указан, отмечаем его на карте
        if path:
            for x, y in path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 4  # 4 - путь
        
        # Создаем цветовую карту
        colors = ['white', 'darkgreen', 'blue', 'red', 'yellow']
        cmap = ListedColormap(colors)
        
        # Отображаем карту
        plt.imshow(grid_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        # Заголовок
        if title:
            plt.title(title)
        else:
            plt.title(f'Карта {self.height}x{self.width}')
        
        plt.show()  # Блокирующий вызов - ждем закрытия окна

# Класс для A* алгоритма
class AStarAgent:
    def __init__(self, grid, start_pos, finish_pos):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.explored_cells = []
    
    def heuristic(self, pos):
        # Манхэттенское расстояние
        return abs(pos[0] - self.finish_pos[0]) + abs(pos[1] - self.finish_pos[1])
    
    def is_valid_position(self, pos):
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != 1)  # Не стена
    
    def find_path(self, visualize=False):
        start_time = time.time()
        
        # Открытый список (приоритетная очередь)
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.start_pos), 0, self.start_pos))
        
        # Для восстановления пути
        came_from = {self.start_pos: None}
        
        # Стоимость от старта до текущей позиции
        g_score = {self.start_pos: 0}
        
        # Счетчик шагов
        steps = 0
        
        # Направления движения
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Вправо, вниз, влево, вверх
        
        while open_set:
            steps += 1
            _, _, current = heapq.heappop(open_set)
            
            self.explored_cells.append(current)
            
            # Если достигли финиша
            if current == self.finish_pos:
                # Восстанавливаем путь
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                
                # Визуализируем результат
                if visualize:
                    grid_vis = self.grid.copy()
                    
                    # Отмечаем исследованные клетки
                    for x, y in self.explored_cells:
                        if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                            grid_vis[x, y] = 5  # 5 - исследованная клетка
                    
                    # Отмечаем путь
                    for x, y in path:
                        if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                            grid_vis[x, y] = 4  # 4 - путь
                    
                    # Создаем цветовую карту
                    colors = ['white', 'darkgreen', 'blue', 'red', 'yellow', 'lightblue']
                    cmap = ListedColormap(colors)
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(grid_vis, cmap=cmap)
                    plt.title(f'A* Algorithm (Шаги: {steps})')
                    plt.show()  # Блокирующий вызов
                
                time_elapsed = time.time() - start_time
                return path, time_elapsed, steps
            
            # Исследуем соседей
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_position(next_pos):
                    continue
                
                # Рассчитываем новую стоимость пути
                tentative_g_score = g_score[current] + 1
                
                # Если нашли лучший путь
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(next_pos)
                    heapq.heappush(open_set, (f_score, steps, next_pos))
        
        # Путь не найден
        time_elapsed = time.time() - start_time
        return [], time_elapsed, steps
    
    def visualize_result(self):
        # Визуализируем результат поиска
        if not hasattr(self, 'path'):
            print("Нет пути для визуализации")
            return
        
        grid_vis = self.grid.copy()
        
        # Отмечаем исследованные клетки
        for x, y in self.explored_cells:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем путь
        for x, y in self.path:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 4  # 4 - путь
        
        # Создаем цветовую карту
        colors = ['white', 'darkgreen', 'blue', 'red', 'yellow', 'lightblue']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_vis, cmap=cmap)
        plt.title('Результат поиска пути A*')
        plt.show()  # Блокирующий вызов

# Класс для Spore алгоритма
class SporeAgent:
    def __init__(self, grid, start_pos, finish_pos, max_iterations=100):
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.max_iterations = max_iterations
        
        # Список спор (position, can_reach_finish, marked_for_removal)
        self.spores = [(start_pos[0], start_pos[1], True, False)]
        
        # Множество клеток, из которых можно достичь финиш
        self.can_reach_finish = {start_pos}
        
        # Пути агентов
        self.agent_paths = []
        
        # Успешные пути
        self.successful_paths = []
        
        # Граф достижимости
        self.reachability_graph = {start_pos: None}
        
        # Итоговый путь
        self.final_path = []
        
        # Счетчик итераций
        self.iteration = 0
        
        # Флаг найденного пути
        self.path_found = False
        
        # Исследованные клетки
        self.explored_cells = set()
    
    def distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid_position(self, pos):
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != 1)  # Не стена
    
        # Исправьте метод generate_spore_position:
    def generate_spore_position(self):
        # Генерируем позицию для новой споры
        print("Создаем новую спору...")
        
        # С вероятностью 15% пытаемся разместить рядом со стартом
        if random.random() < 0.15:
            print("  Пытаемся разместить рядом со стартом")
            attempts = 0
            while attempts < 20:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                new_pos = (self.start_pos[0] + offset_x, self.start_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    print(f"  Создана спора рядом со стартом: {new_pos}")
                    return new_pos
                attempts += 1
        
        # С вероятностью 30% пытаемся разместить рядом с финишем
        if random.random() < 0.3:
            print("  Пытаемся разместить рядом с финишем")
            attempts = 0
            while attempts < 20:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                new_pos = (self.finish_pos[0] + offset_x, self.finish_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    print(f"  Создана спора рядом с финишем: {new_pos}")
                    return new_pos
                attempts += 1
        
        # Пытаемся разместить рядом с достижимыми спорами
        reachable_spores = [(x, y) for x, y, can_reach, _ in self.spores if can_reach and not _]
        if reachable_spores:
            print(f"  Пытаемся разместить рядом с {len(reachable_spores)} достижимыми спорами")
            attempts = 0
            while attempts < 30:
                base_spore = random.choice(reachable_spores)
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                new_pos = (base_spore[0] + offset_x, base_spore[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    print(f"  Создана спора рядом с достижимой спорой: {new_pos}")
                    return new_pos
                attempts += 1
        
        # Если не удалось разместить рядом с достижимыми спорами, размещаем случайно
        print("  Размещаем спору случайно")
        attempts = 0
        while attempts < 50:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            new_pos = (x, y)
            
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                print(f"  Создана случайная спора: {new_pos}")
                return new_pos
            attempts += 1
        
        print("  Не удалось создать новую спору")
        return None
    
    def get_nearest_reachable_point(self, pos):
        # Находим ближайшую точку, из которой можно достичь финиш
        if not self.can_reach_finish:
            return self.finish_pos
        
        min_dist = float('inf')
        nearest_point = None
        
        for goal_point in self.can_reach_finish:
            dist = self.distance(pos, goal_point)
            if dist < min_dist:
                min_dist = dist
                nearest_point = goal_point
        
        return nearest_point if nearest_point else self.finish_pos
    
    def sample_action(self, current_pos, target_pos):
        # Сэмплируем действие в направлении цели с шумом
        
        # Направление к цели
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        
        # Нормализуем
        length = max(1, np.sqrt(dir_x**2 + dir_y**2))
        dir_x /= length
        dir_y /= length
        
        # Добавляем шум
        dir_x += np.random.normal(0, 0.5)
        dir_y += np.random.normal(0, 0.5)
        
        # Определяем направление (вверх, вниз, влево, вправо)
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
    
    
    # Исправьте метод run_random_sequence:
    def run_random_sequence(self, start_point, sequence_length=30, num_sequences=10):
        # Запускаем несколько случайных последовательностей из точки
        print(f"Запускаем {num_sequences} последовательностей из точки {start_point}")
        
        # Определяем цель
        target = self.get_nearest_reachable_point(start_point)
        print(f"  Цель для агентов: {target}")
        
        current_paths = []
        reached_goal = False
        
        # Запускаем несколько последовательностей
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
                    
                    # Отмечаем точку как исследованную
                    self.explored_cells.add(current)
                    
                    # Если достигли финиша
                    if current == self.finish_pos:
                        print(f"  Последовательность {seq+1}: Достигнут финиш!")
                        self.can_reach_finish.add(start_point)
                        reached_goal = True
                        self.successful_paths.append(path)
                        break
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_finish:
                        print(f"  Последовательность {seq+1}: Достигнута точка, из которой можно достичь финиш!")
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
                print(f"  Последовательность {seq+1}: Длина пути: {len(path)}")
        
        self.agent_paths.extend(current_paths)
        return current_paths, reached_goal

    def update_reachable_points(self):
        # Обновляем множество точек, из которых можно достичь цель
        changed = True
        while changed:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам графа достижимости
            for point, parent in self.reachability_graph.items():
                if point in self.can_reach_finish:
                    continue
                
                # Если из точки можно попасть в достижимую точку
                for goal_point in self.can_reach_finish:
                    if goal_point in self.reachability_graph and point != self.finish_pos:
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки
            self.can_reach_finish.update(new_points)
    
    def step(self):
        # Выполняем один шаг алгоритма
        if self.iteration >= self.max_iterations or self.path_found:
            return False, "Алгоритм завершен"
        
        # Генерируем новую спору
        new_spore_pos = self.generate_spore_position()
        
        if new_spore_pos:
            # Добавляем спору
            self.spores.append((new_spore_pos[0], new_spore_pos[1], False, False))
            
            # Запускаем агентов
            paths, reached_goal = self.run_random_sequence(new_spore_pos)
            
            # Обновляем достижимые точки
            self.update_reachable_points()
            
            # Проверяем, найден ли путь
            if reached_goal or self.finish_pos in self.reachability_graph:
                self.path_found = True
                self.final_path = self.reconstruct_path()
                return True, "Путь найден!"
        
        self.iteration += 1
        return True, f"Итерация {self.iteration}/{self.max_iterations}"
    
    def reconstruct_path(self):
        # Восстанавливаем путь от старта до финиша
        if self.finish_pos in self.reachability_graph:
            path = [self.finish_pos]
            current = self.finish_pos
            
            while current != self.start_pos and current is not None:
                current = self.reachability_graph[current]
                if current is not None:
                    path.append(current)
            
            path.reverse()
            return path
        
        # Если нет прямого пути, строим граф из всех путей
        graph = {}
        
        for path in self.agent_paths + self.successful_paths:
            for i in range(len(path) - 1):
                if path[i] not in graph:
                    graph[path[i]] = []
                if path[i+1] not in graph[path[i]]:
                    graph[path[i]].append(path[i+1])
        
        # BFS для поиска пути
        queue = deque([self.start_pos])
        visited = {self.start_pos: None}
        
        while queue:
            current = queue.popleft()
            
            if current == self.finish_pos:
                # Восстанавливаем путь
                path = []
                while current:
                    path.append(current)
                    current = visited[current]
                path.reverse()
                return path
            
            if current in graph:
                for next_pos in graph[current]:
                    if next_pos not in visited:
                        queue.append(next_pos)
                        visited[next_pos] = current
        
        return []
        
    # В классе SporeAgent исправьте метод find_path:
    def find_path(self, visualize=False):
        start_time = time.time()
        
        print("Начинаю поиск пути с помощью Spore...")
        
        # Добавим спору в старте, если её ещё нет
        if not self.spores:
            self.spores.append((self.start_pos[0], self.start_pos[1], True, False))
            print(f"Добавлена начальная спора в позиции {self.start_pos}")
        
        # Добавляем старт в множество точек, из которых можно достичь финиш
        self.can_reach_finish = {self.start_pos}
        
        # Выполняем итерации алгоритма
        continue_sim = True
        while continue_sim and self.iteration < self.max_iterations:
            print(f"Выполняется итерация {self.iteration+1}/{self.max_iterations}")
            continue_sim, message = self.step()
            print(f"  {message}")
            
            # Визуализируем процесс каждые 10 итераций
            if visualize and self.iteration % 10 == 0:
                self.visualize_step()
        
        time_elapsed = time.time() - start_time
        
        # Финальная проверка для восстановления пути
        if not self.final_path and self.spores:
            print("Попытка восстановить путь...")
            self.update_reachable_points()
            self.final_path = self.reconstruct_path()
            if self.final_path:
                print(f"Путь успешно восстановлен, длина: {len(self.final_path)}")
        
        # Визуализируем результат
        if visualize and self.final_path:
            self.visualize_result()
        
        return self.final_path, time_elapsed, self.iteration
    
    def visualize_step(self):
        # Визуализируем текущее состояние алгоритма
        grid_vis = self.grid.copy()
        
        # Отмечаем исследованные клетки
        for x, y in self.explored_cells:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем споры
        for x, y, can_reach, _ in self.spores:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                if can_reach:
                    grid_vis[x, y] = 6  # 6 - спора, которая может достичь финиша
                else:
                    grid_vis[x, y] = 4  # 4 - обычная спора
        
        # Отмечаем найденный путь
        if self.final_path:
            for x, y in self.final_path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 7  # 7 - путь
        
        # Создаем цветовую карту
        colors = ['white', 'darkgreen', 'blue', 'red', 'yellow', 'lightblue', 'purple', 'orange']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_vis, cmap=cmap)
        plt.title(f'Spore Algorithm (Итерация {self.iteration}/{self.max_iterations})')
        plt.show()  # Блокирующий вызов
    
    def visualize_result(self):
        # Визуализируем результат алгоритма
        grid_vis = self.grid.copy()
        
        # Отмечаем исследованные клетки
        for x, y in self.explored_cells:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем споры
        for x, y, can_reach, _ in self.spores:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                if can_reach:
                    grid_vis[x, y] = 6  # 6 - спора, которая может достичь финиша
                else:
                    grid_vis[x, y] = 4  # 4 - обычная спора
        
        # Отмечаем найденный путь
        if self.final_path:
            for x, y in self.final_path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 7  # 7 - путь
        
        # Создаем цветовую карту
        colors = ['white', 'darkgreen', 'blue', 'red', 'yellow', 'lightblue', 'purple', 'orange']
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_vis, cmap=cmap)
        plt.title(f'Результат Spore (Итераций: {self.iteration})')
        plt.show()  # Блокирующий вызов

# Функция для запуска сравнения
def run_comparison():
    # Создаем карту
    print("Генерация карты...")
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Визуализируем карту
    print("Визуализация исходной карты...")
    generator.visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Запускаем A*
    print("\nЗапуск A*...")
    a_star_agent = AStarAgent(grid, start_pos, finish_pos)
    a_star_path, a_star_time, a_star_steps = a_star_agent.find_path(visualize=True)
    
    if a_star_path:
        print(f"A*: Путь найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
        print(f"A*: Длина пути: {len(a_star_path)}")
    else:
        print(f"A*: Путь не найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
    
    # Запускаем Spore
    print("\nЗапуск Spore...")
    spore_agent = SporeAgent(grid, start_pos, finish_pos, max_iterations=100)
    spore_path, spore_time, spore_iterations = spore_agent.find_path(visualize=True)
    
    if spore_path:
        print(f"Spore: Путь найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
        print(f"Spore: Длина пути: {len(spore_path)}")
    else:
        print(f"Spore: Путь не найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
    
    # Сравниваем результаты
    print("\nСравнение результатов:")
    print(f"A*: {a_star_time:.4f}с, {a_star_steps} шагов, длина пути: {len(a_star_path) if a_star_path else 0}")
    print(f"Spore: {spore_time:.4f}с, {spore_iterations} итераций, длина пути: {len(spore_path) if spore_path else 0}")

if __name__ == "__main__":
    run_comparison()