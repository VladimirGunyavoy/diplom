import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import heapq
from collections import deque
import pickle
import os

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

def visualize_map(grid, start_pos, finish_pos, path=None, explored=None, spores=None, title=None):
    """Визуализирует карту"""
    plt.figure(figsize=(10, 10))
    
    # Создаем копию карты для визуализации
    grid_vis = grid.copy()
    
    # Отмечаем исследованные клетки
    if explored:
        for x, y in explored:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
    
    # Отмечаем споры
    if spores:
        for spore in spores:
            x, y = spore[0], spore[1]
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                # Спора достижимая или обычная
                grid_vis[x, y] = 6 if spore[2] else 4
    
    # Отмечаем путь
    if path:
        for x, y in path:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 7  # 7 - путь
    
    # Создаем цветовую карту
    # 0: пустота, 1: стена, 2: старт, 3: финиш, 4: спора, 5: исследованные, 6: достижимая спора, 7: путь
    colors = ['white', 'darkgreen', 'blue', 'red', 'yellow', 'lightblue', 'purple', 'orange']
    cmap = ListedColormap(colors)
    
    # Отображаем карту с сеткой
    plt.imshow(grid_vis, cmap=cmap)
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    # Заголовок
    if title:
        plt.title(title)
    else:
        plt.title(f'Карта {grid.shape[0]}x{grid.shape[1]}')
    
    plt.show()

class ImprovedSporeAgent:
    """
    Улучшенный агент поиска пути с использованием алгоритма Spore.
    """
    
    def __init__(self, grid, start_pos, finish_pos, max_iterations=100, debug=True):
        """
        Инициализация Spore агента.
        
        Параметры:
        - grid: матрица карты (0 - свободное пространство, 1 - стена, 2 - старт, 3 - финиш)
        - start_pos: кортеж (x, y) - координаты стартовой позиции
        - finish_pos: кортеж (x, y) - координаты целевой позиции
        - max_iterations: максимальное количество итераций алгоритма
        - debug: вывод отладочной информации
        """
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
        
        # Расстояние между стартом и финишем для адаптивных параметров
        self.distance_start_finish = np.sqrt((start_pos[0] - finish_pos[0])**2 + (start_pos[1] - finish_pos[1])**2)
        
        if self.debug:
            print(f"Инициализирован ImprovedSporeAgent:")
            print(f"  - Размер карты: {self.height}x{self.width}")
            print(f"  - Старт: {self.start_pos}")
            print(f"  - Финиш: {self.finish_pos}")
            print(f"  - Расстояние между стартом и финишем: {self.distance_start_finish:.2f}")
    
    def distance(self, pos1, pos2):
        """Вычисляет евклидово расстояние между двумя позициями"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid_position(self, pos):
        """Проверяет, является ли позиция допустимой"""
        x, y = pos
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != 1)  # Не стена
    
    def generate_spore_position(self):
        """Генерирует позицию для новой споры"""
        if self.debug:
            print("Генерация новой споры...")
        
        # Оценим, на каком этапе мы находимся
        exploration_ratio = min(1.0, len(self.explored_cells) / (self.height * self.width * 0.3))
        
        # Определим приоритеты размещения спор в зависимости от этапа исследования
        priorities = {
            'near_finish': 0.4 - exploration_ratio * 0.2,  # Высокий приоритет вначале, затем снижается
            'near_start': 0.2 - exploration_ratio * 0.1,   # Средний приоритет вначале, затем снижается
            'near_reachable': 0.3 + exploration_ratio * 0.2,  # Средний приоритет, затем возрастает
            'random': 0.1 + exploration_ratio * 0.1        # Низкий приоритет, затем возрастает
        }
        
        # Сначала пытаемся размещать споры на прямой линии между стартом и финишем
        if random.random() < 0.3 and self.iteration < self.max_iterations * 0.3:
            if self.debug:
                print("  Пытаемся разместить на линии между стартом и финишем")
            
            # Рассчитываем параметр t от 0 до 1
            # t=0 соответствует старту, t=1 соответствует финишу
            t = random.uniform(0.2, 0.8)  # Избегаем концов линии
            
            # Линейная интерполяция между стартом и финишем
            x = int(self.start_pos[0] + t * (self.finish_pos[0] - self.start_pos[0]))
            y = int(self.start_pos[1] + t * (self.finish_pos[1] - self.start_pos[1]))
            new_pos = (x, y)
            
            # Проверяем валидность позиции
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                if self.debug:
                    print(f"  Создана спора на линии между стартом и финишем: {new_pos}")
                return new_pos
        
        # Выбираем стратегию размещения на основе приоритетов
        r = random.random()
        
        # Размещение рядом с финишем
        if r < priorities['near_finish']:
            if self.debug:
                print("  Пытаемся разместить рядом с финишем")
            
            # Адаптивный радиус в зависимости от размера карты
            radius = max(5, min(10, int(self.distance_start_finish * 0.2)))
            
            attempts = 0
            while attempts < 30:
                # Используем нормальное распределение для близости к финишу
                offset_x = int(np.random.normal(0, radius * 0.5))  # Меньшее стандартное отклонение для большей концентрации
                offset_y = int(np.random.normal(0, radius * 0.5))
                new_pos = (self.finish_pos[0] + offset_x, self.finish_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    if self.debug:
                        print(f"  Создана спора рядом с финишем: {new_pos}")
                    return new_pos
                attempts += 1
        
        # Размещение рядом со стартом
        elif r < priorities['near_finish'] + priorities['near_start']:
            if self.debug:
                print("  Пытаемся разместить рядом со стартом")
            
            radius = max(5, min(10, int(self.distance_start_finish * 0.2)))
            
            attempts = 0
            while attempts < 20:
                offset_x = int(np.random.normal(0, radius * 0.5))
                offset_y = int(np.random.normal(0, radius * 0.5))
                new_pos = (self.start_pos[0] + offset_x, self.start_pos[1] + offset_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    if self.debug:
                        print(f"  Создана спора рядом со стартом: {new_pos}")
                    return new_pos
                attempts += 1
        
        # Размещение рядом с достижимыми спорами
        elif r < priorities['near_finish'] + priorities['near_start'] + priorities['near_reachable']:
            reachable_spores = [(x, y) for x, y, can_reach, _ in self.spores if can_reach and not _]
            if reachable_spores:
                if self.debug:
                    print(f"  Пытаемся разместить рядом с достижимыми спорами ({len(reachable_spores)})")
                
                # Выбираем случайную достижимую спору, отдавая предпочтение тем, 
                # которые находятся ближе к финишу
                weights = []
                for spore_pos in reachable_spores:
                    # Обратное расстояние до финиша как вес (ближе = выше вес)
                    dist_to_finish = self.distance(spore_pos, self.finish_pos)
                    weight = 1.0 / max(1.0, dist_to_finish)
                    weights.append(weight)
                
                # Нормализуем веса
                if sum(weights) > 0:
                    weights = [w/sum(weights) for w in weights]
                    # Выбираем споры с учетом весов
                    base_spore = random.choices(reachable_spores, weights=weights, k=1)[0]
                else:
                    base_spore = random.choice(reachable_spores)
                
                attempts = 0
                radius = max(5, min(10, int(self.distance_start_finish * 0.15)))
                
                while attempts < 30:
                    offset_x = int(np.random.normal(0, radius))
                    offset_y = int(np.random.normal(0, radius))
                    new_pos = (base_spore[0] + offset_x, base_spore[1] + offset_y)
                    
                    if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                        if self.debug:
                            print(f"  Создана спора рядом с достижимой спорой: {new_pos}")
                        return new_pos
                    attempts += 1
        
        # Случайное размещение в неисследованной области
        if self.debug:
            print("  Размещаем спору в случайной неисследованной области")
        
        # Сначала пытаемся разместить в неисследованной области
        unexplored = [(x, y) for x in range(self.height) for y in range(self.width) 
                     if (x, y) not in self.explored_cells and self.is_valid_position((x, y))]
        
        if unexplored and random.random() < 0.7:  # 70% шанс использовать неисследованную область
            attempts = 0
            max_attempts = min(50, len(unexplored))
            
            while attempts < max_attempts:
                new_pos = random.choice(unexplored)
                
                if new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    if self.debug:
                        print(f"  Создана спора в неисследованной области: {new_pos}")
                    return new_pos
                
                attempts += 1
        
        # Полностью случайное размещение как последнее средство
        attempts = 0
        while attempts < 50:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            new_pos = (x, y)
            
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                if self.debug:
                    print(f"  Создана случайная спора: {new_pos}")
                return new_pos
            attempts += 1
        
        if self.debug:
            print("  Не удалось создать новую спору")
        return None
    
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
    
    def sample_action(self, current_pos, target_pos):
        """Сэмплирует действие в направлении цели с шумом"""
        # Направление к цели
        dir_x = target_pos[0] - current_pos[0]
        dir_y = target_pos[1] - current_pos[1]
        
        # Определяем величину шума в зависимости от стадии алгоритма
        # В начале больше исследования (больше шума), затем больше фокуса на цели (меньше шума)
        exploration_stage = min(1.0, self.iteration / self.max_iterations)
        noise_std_dev = 0.8 - exploration_stage * 0.5  # От 0.8 до 0.3
        
        # Нормализуем
        length = max(1e-10, np.sqrt(dir_x**2 + dir_y**2))
        dir_x /= length
        dir_y /= length
        
        # Добавляем шум
        dir_x += np.random.normal(0, noise_std_dev)
        dir_y += np.random.normal(0, noise_std_dev)
        
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
    
    def run_random_sequence(self, start_point, sequence_length=None, num_sequences=None):
        """Запускает несколько случайных последовательностей из точки"""
        # Адаптивная настройка параметров в зависимости от размера карты
        if sequence_length is None:
            # Длина последовательности равна расстоянию между стартом и финишем
            sequence_length = max(30, int(self.distance_start_finish * 1.5))
        
        if num_sequences is None:
            # Больше последовательностей на больших картах
            num_sequences = max(10, int(self.distance_start_finish * 0.3))
        
        if self.debug:
            print(f"Запуск {num_sequences} последовательностей длиной {sequence_length} из точки {start_point}")
        
        # Определяем цель - по умолчанию это финиш, но если есть более близкая достижимая точка, выбираем её
        target = self.get_nearest_reachable_point(start_point)
        if self.debug:
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
                        if self.debug:
                            print(f"  Последовательность {seq+1}: Достигнут финиш!")
                        self.can_reach_finish.add(start_point)
                        reached_goal = True
                        self.successful_paths.append(path)
                        break
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_finish:
                        if self.debug:
                            print(f"  Последовательность {seq+1}: Достигнута достижимая точка!")
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
                if self.debug:
                    print(f"  Последовательность {seq+1}: Длина пути: {len(path)}")
        
        self.agent_paths.extend(current_paths)
        return current_paths, reached_goal
    
    def update_reachable_points(self):
        """Обновляет множество точек, из которых можно достичь цель"""
        if self.debug:
            print("Обновление достижимых точек...")
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
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки
            self.can_reach_finish.update(new_points)
        
        # Обновляем споры, которые могут достичь финиш
        for i, (x, y, _, marked) in enumerate(self.spores):
            if (x, y) in self.can_reach_finish and not marked:
                self.spores[i] = (x, y, True, marked)
        
        if self.debug:
            final_count = len(self.can_reach_finish)
            if final_count > initial_count:
                print(f"  Добавлено {final_count - initial_count} новых достижимых точек")
            print(f"  Всего достижимых точек: {final_count}")
    
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
            if self.finish_pos in self.can_reach_finish or reached_goal:
                self.path_found = True
                self.final_path = self.reconstruct_path()
                
                if self.final_path:
                    return True, f"Путь найден на итерации {self.iteration + 1}! Длина пути: {len(self.final_path)}"
                else:
                    # Если путь восстановить не удалось, продолжаем
                    self.path_found = False
                    return True, f"Финиш достижим, но путь не восстановлен. Продолжаем поиск."
            
            # Проверяем прогресс - если мы близки к финишу
            min_dist_to_finish = float('inf')
            for x, y, _, _ in self.spores:
                dist = self.distance((x, y), self.finish_pos)
                min_dist_to_finish = min(min_dist_to_finish, dist)
                
            progress_msg = f"Ближайшая спора к финишу: {min_dist_to_finish:.2f}, "
            progress_msg += f"Исследовано: {len(self.explored_cells)}/{self.height * self.width}"
            
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}. {progress_msg}"
        else:
            # Если не удалось создать новую спору, пробуем ещё раз с другими параметрами
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}. Не удалось создать новую спору."
    def reconstruct_path(self):
        # Проверки перед обращением к графу достижимости
        visited_points = set()  # Для предотвращения циклов
        
        if self.finish_pos in self.reachability_graph:
            path = [self.finish_pos]
            current = self.finish_pos
            visited_points.add(current)
            
            while current != self.start_pos and current is not None:
                if current in self.reachability_graph:
                    next_point = self.reachability_graph[current]
                    # Предотвращение циклов
                    if next_point in visited_points:
                        break
                    
                    if next_point is not None:
                        path.append(next_point)
                        visited_points.add(next_point)
                    
                    current = next_point
                else:
                    break
            
            path.reverse()
            return path
    
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
    
    def find_path(self, visualize=False, visualize_steps=False, visualize_interval=5):
        """Находит путь от старта до финиша"""
        start_time = time.time()
        
        print("Начинаю поиск пути с помощью улучшенного алгоритма Spore...")
        
        # Визуализируем начальное состояние
        if visualize_steps:
            print(f"Визуализация начального состояния...")
            visualize_map(
                self.grid, 
                self.start_pos, 
                self.finish_pos, 
                path=self.final_path, 
                explored=list(self.explored_cells), 
                spores=self.spores,
                title=f"Spore Algorithm (Итерация 0/{self.max_iterations})"
            )
        
        # Выполняем итерации алгоритма
        continue_sim = True
        while continue_sim and self.iteration < self.max_iterations:
            # Визуализируем текущее состояние, если требуется
            if visualize_steps and self.iteration % visualize_interval == 0 and self.iteration > 0:
                print(f"Визуализация состояния на итерации {self.iteration}...")
                visualize_map(
                    self.grid, 
                    self.start_pos, 
                    self.finish_pos, 
                    path=self.final_path, 
                    explored=list(self.explored_cells), 
                    spores=self.spores,
                    title=f"Spore Algorithm (Итерация {self.iteration}/{self.max_iterations})"
                )
            
            # Выполняем шаг алгоритма
            continue_sim, message = self.step()
            print(f"  [Итерация {self.iteration}] {message}")
            
            # Если путь найден, выходим из цикла
            if self.path_found:
                break
        
        time_elapsed = time.time() - start_time
        
        # Финальная проверка для восстановления пути
        if not self.final_path:
            print("Выполняем финальную реконструкцию пути...")
            self.update_reachable_points()
            self.final_path = self.reconstruct_path()
        
        # Результаты
        if self.final_path:
            print(f"Путь найден за {time_elapsed:.4f} секунд!")
            print(f"Длина пути: {len(self.final_path)}")
            print(f"Итераций: {self.iteration}")
            print(f"Исследовано клеток: {len(self.explored_cells)}")
            print(f"Создано спор: {len(self.spores)}")
        else:
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
                path=self.final_path, 
                explored=list(self.explored_cells), 
                spores=self.spores,
                title=f"Результат Spore (Итераций: {self.iteration})"
            )
        
        return self.final_path, time_elapsed, self.iteration

def main():
    # Параметры
    map_file = "map_data.pkl"  # Путь к файлу с картой
    visualize = True  # Визуализировать результат
    visualize_steps = True  # Визуализировать промежуточные шаги
    visualize_interval = 10  # Интервал визуализации (каждые N итераций)
    max_iterations = 100  # Максимальное количество итераций
    debug = True  # Выводить отладочную информацию
    
    # Загружаем карту
    grid, start_pos, finish_pos = load_map_data(map_file)
    
    # Визуализируем исходную карту
    print("Визуализация исходной карты...")
    visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Запускаем улучшенный алгоритм Spore
    spore_agent = ImprovedSporeAgent(grid, start_pos, finish_pos, max_iterations=max_iterations, debug=debug)
    path, time_elapsed, iterations = spore_agent.find_path(
        visualize=visualize, 
        visualize_steps=visualize_steps,
        visualize_interval=visualize_interval
    )
    
    print("Готово!")

if __name__ == "__main__":
    main()