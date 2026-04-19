import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import matplotlib.animation as animation
from collections import deque


class SporeAgent:
    """
    Агент поиска пути с использованием алгоритма Spore.
    """
    
    def __init__(self, grid, start_pos, finish_pos, max_iterations=200):
        """
        Инициализация Spore агента.
        
        Параметры:
        - grid: матрица карты (0 - свободное пространство, 1 - стена, 2 - старт, 3 - финиш)
        - start_pos: кортеж (x, y) - координаты стартовой позиции
        - finish_pos: кортеж (x, y) - координаты целевой позиции
        - max_iterations: максимальное количество итераций алгоритма
        """
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.max_iterations = max_iterations
        
        # Список спор (позиция, может_достичь_финиша, помечена_для_удаления)
        self.spores = [(*self.start_pos, True, False)]  # Начинаем со старта как споры
        
        # Множество клеток, из которых можно достичь финиша
        self.can_reach_finish = {self.start_pos}
        
        # Пути агентов [(pos, pos, ...), (...), ...]
        self.agent_paths = []
        
        # Успешные пути, ведущие к финишу
        self.successful_paths = []
        
        # Граф достижимости (точка -> предыдущая точка на пути)
        self.reachability_graph = {self.start_pos: None}
        
        # Итоговый путь от старта до финиша
        self.final_path = []
        
        # Счетчик итераций
        self.iteration = 0
        
        # Флаг найденного пути
        self.path_found = False
        
        # Для визуализации
        self.explored_cells = set()  # Множество исследованных клеток
    
    def distance(self, pos1, pos2):
        """
        Вычисляет евклидово расстояние между двумя позициями.
        
        Параметры:
        - pos1: первая позиция
        - pos2: вторая позиция
        
        Возвращает:
        - расстояние между позициями
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid_position(self, pos):
        """
        Проверяет, является ли позиция допустимой.
        
        Параметры:
        - pos: проверяемая позиция
        
        Возвращает:
        - True, если позиция допустима, иначе False
        """
        x, y = pos
        # Проверяем, что позиция внутри границ карты и не является стеной
        return (0 <= x < self.height and 
                0 <= y < self.width and 
                self.grid[x, y] != 1)
    
    def get_nearest_reachable_point(self, pos):
        """
        Находит ближайшую точку, из которой можно достичь финиша.
        
        Параметры:
        - pos: текущая позиция
        
        Возвращает:
        - ближайшая точка, из которой можно достичь финиша
        """
        if not self.can_reach_finish:
            return self.finish_pos
        
        min_dist = float('inf')
        nearest_point = None
        
        # Сначала проверяем, ближе ли финиш, чем любая достижимая спора
        finish_dist = self.distance(pos, self.finish_pos)
        if finish_dist < min_dist:
            min_dist = finish_dist
            nearest_point = self.finish_pos
        
        # Проверяем расстояние до достижимых спор
        for spore_pos in self.can_reach_finish:
            dist = self.distance(pos, spore_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_point = spore_pos
        
        return nearest_point if nearest_point else self.finish_pos
    
    def sample_action(self, current_pos, target_pos, std_dev=1.0):
        """
        Сэмплирует действие с матожиданием в направлении цели.
        Движение только в 4 направлениях: вверх, вниз, влево, вправо.
        
        Параметры:
        - current_pos: текущая позиция
        - target_pos: целевая позиция
        - std_dev: стандартное отклонение для шума
        
        Возвращает:
        - следующая позиция
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
            norm_dir = direction / np.linalg.norm(direction)
            dot_product = np.dot(action_vec, norm_dir)
            
            # Преобразуем в вероятность (больше в направлении цели)
            prob = max(0.1, (dot_product + 1) / 2)  # Чтобы вероятность была всегда положительной
            probs.append(prob)
        
        # Нормализуем вероятности
        probs = np.array(probs) / sum(probs)
        
        # Выбираем действие на основе вероятностей
        chosen_action = np.random.choice(len(possible_actions), p=probs)
        
        return possible_actions[chosen_action]
    
    def run_random_sequence(self, start_point, sequence_length=10, num_sequences=5):
        """
        Запускает несколько случайных последовательностей шагов из заданной точки.
        
        Параметры:
        - start_point: начальная точка
        - sequence_length: длина последовательности
        - num_sequences: количество последовательностей
        
        Возвращает:
        - current_paths: список путей
        - reached_goal: True, если достигнута цель, иначе False
        """
        # Определяем цель для матожидания
        if self.can_reach_finish:
            target = self.get_nearest_reachable_point(start_point)
        else:
            target = self.finish_pos
            
        current_paths = []
        reached_goal = False
        
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
            
            # Сохраняем путь для визуализации
            current_paths.append(path)
        
        self.agent_paths.extend(current_paths)
        return current_paths, reached_goal
    
    def update_reachable_points(self):
        """
        Обновляет множество точек, из которых можно достичь цель,
        включая точки, ведущие к уже известным точкам.
        """
        changed = True
        while changed:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам в графе достижимости
            for point, parent in self.reachability_graph.items():
                # Если точка уже помечена как ведущая к цели, пропускаем
                if point in self.can_reach_finish:
                    continue
                
                # Если из этой точки можно попасть в точку, ведущую к цели
                for goal_point in self.can_reach_finish:
                    # Проверяем, что goal_point есть в графе достижимости
                    if goal_point in self.reachability_graph and point != self.finish_pos:
                        # Проверяем, что point является родителем для goal_point
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки в множество
            self.can_reach_finish.update(new_points)
    
    def generate_spore_position(self):
        """
        Генерирует позицию для новой споры.
        
        Возвращает:
        - позиция для новой споры
        """
        # Начнем с поиска позиции рядом с достижимыми спорами
        reachable_spores = [(x, y) for x, y, can_reach, _ in self.spores if can_reach and not _]
        
        # Если есть достижимые споры, пытаемся разместить новую спору рядом с ними
        if reachable_spores:
            attempts = 0
            while attempts < 50:
                base_spore = random.choice(reachable_spores)
                offset_x = int(np.random.normal(0, 3))
                offset_y = int(np.random.normal(0, 3))
                new_x = base_spore[0] + offset_x
                new_y = base_spore[1] + offset_y
                new_pos = (new_x, new_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                
                attempts += 1
        
        # Если не удалось разместить рядом с достижимыми спорами, пробуем другие стратегии
        
        # С вероятностью 20% пробуем разместить рядом со стартом
        if random.random() < 0.2:
            attempts = 0
            while attempts < 30:
                offset_x = int(np.random.normal(0, 3))
                offset_y = int(np.random.normal(0, 3))
                new_x = self.start_pos[0] + offset_x
                new_y = self.start_pos[1] + offset_y
                new_pos = (new_x, new_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                
                attempts += 1
        
        # С вероятностью 20% пробуем разместить рядом с финишем
        if random.random() < 0.2:
            attempts = 0
            while attempts < 30:
                offset_x = int(np.random.normal(0, 3))
                offset_y = int(np.random.normal(0, 3))
                new_x = self.finish_pos[0] + offset_x
                new_y = self.finish_pos[1] + offset_y
                new_pos = (new_x, new_y)
                
                if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                    return new_pos
                
                attempts += 1
        
        # В остальных случаях выбираем случайную позицию на карте
        for _ in range(100):  # Ограничиваем количество попыток
            new_x = random.randint(0, self.height - 1)
            new_y = random.randint(0, self.width - 1)
            new_pos = (new_x, new_y)
            
            if self.is_valid_position(new_pos) and new_pos not in [(x, y) for x, y, _, _ in self.spores]:
                return new_pos
        
        # Если не удалось найти позицию, возвращаем None
        return None
    
    def step(self):
        """
        Выполняет один шаг алгоритма - создает новую спору и запускает из нее агентов.
        
        Возвращает:
        - continue_sim: True, если симуляция должна продолжаться, иначе False
        - message: информационное сообщение
        """
        if self.iteration >= self.max_iterations or self.path_found:
            return False, f"{'Путь найден!' if self.path_found else 'Достигнуто максимальное число итераций'}"
        
        # Генерируем новую спору
        new_spore_pos = self.generate_spore_position()
        
        if new_spore_pos:
            # Добавляем спору в список
            self.spores.append((*new_spore_pos, False, False))  # (x, y, can_reach, marked_for_removal)
            
            # Запускаем случайные последовательности из новой споры
            paths, reached_goal = self.run_random_sequence(new_spore_pos)
            
            # Обновляем множество точек, из которых можно достичь цель
            self.update_reachable_points()
            
            # Проверяем, найден ли путь
            if reached_goal or self.finish_pos in self.reachability_graph:
                self.path_found = True
                self.final_path = self.reconstruct_path()
                return True, f"Путь найден на итерации {self.iteration + 1}!"
            
            # Проверяем, можно ли достичь финиш из старта через другие споры
            if self.start_pos in self.can_reach_finish:
                self.path_found = True
                self.final_path = self.reconstruct_path()
                return True, f"Путь найден на итерации {self.iteration + 1} (через промежуточные споры)!"
            
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}, спора: {new_spore_pos}, путей из неё: {len(paths)}"
        else:
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}, не удалось создать новую спору"
    
    def reconstruct_path(self):
        """
        Восстанавливает путь от начала до цели, используя граф достижимости.
        
        Возвращает:
        - список позиций, образующих путь
        """
        if self.finish_pos in self.reachability_graph:
            # Простой случай: финиш достигнут напрямую
            path = [self.finish_pos]
            current = self.finish_pos
            
            while current != self.start_pos and current is not None:
                current = self.reachability_graph[current]
                if current is not None:
                    path.append(current)
            
            path.reverse()
            return path
        
        # Сложный случай: нужно найти путь через промежуточные споры
        # Строим граф из всех путей агентов
        graph = {}
        
        for path in self.agent_paths + self.successful_paths:
            for i in range(len(path) - 1):
                if path[i] not in graph:
                    graph[path[i]] = []
                if path[i+1] not in graph[path[i]]:
                    graph[path[i]].append(path[i+1])
        
        # Используем BFS для поиска пути от старта до финиша
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
        
        # Если не нашли путь, возвращаем пустой список
        return []
    
    def find_path(self, visualize=False):
        """
        Находит путь от старта до финиша с помощью алгоритма Spore.
        
        Параметры:
        - visualize: если True, выполняет визуализацию процесса поиска
        
        Возвращает:
        - path: список позиций, образующих путь (пустой список, если путь не найден)
        - time_elapsed: время, затраченное на поиск пути
        - steps: количество итераций алгоритма
        """
        # Замеряем время выполнения
        start_time = time.time()
        
        # Настройка визуализации, если требуется
        if visualize:
            fig, ax = plt.subplots(figsize=(10, 10))
            img = ax.imshow(self.grid, cmap=ListedColormap(['white', 'black', 'green', 'red', 'yellow', 'lightblue']))
            plt.title('Spore Algorithm')
            plt.ion()  # Включаем интерактивный режим
            plt.show()
        
        # Выполняем итерации алгоритма
        continue_sim = True
        while continue_sim and self.iteration < self.max_iterations:
            continue_sim, message = self.step()
            
            # Обновляем визуализацию, если требуется
            if visualize and self.iteration % 5 == 0:  # Обновляем каждые 5 итераций
                self._visualize_search(ax, img, False)
        
        # Фиксируем затраченное время
        time_elapsed = time.time() - start_time
        
        # Завершаем визуализацию, если она была включена
        if visualize:
            # Визуализируем найденный путь
            self._visualize_search(ax, img, True)
            plt.ioff()  # Отключаем интерактивный режим
            plt.show()
        
        return self.final_path, time_elapsed, self.iteration
    
    def _visualize_search(self, ax, img, final=False):
        """
        Обновляет визуализацию процесса поиска пути.
        
        Параметры:
        - ax: объект осей matplotlib
        - img: объект изображения matplotlib
        - final: если True, визуализирует финальный путь
        """
        # Создаем копию карты для визуализации
        grid_vis = self.grid.copy()
        
        # Отмечаем споры и исследованные клетки
        for x, y in self.explored_cells:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем споры
        for x, y, can_reach, marked in self.spores:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                if can_reach:
                    grid_vis[x, y] = 6  # 6 - спора, которая может достичь финиша
                else:
                    grid_vis[x, y] = 4  # 4 - обычная спора
        
        # Отмечаем найденный путь
        if final and self.final_path:
            for x, y in self.final_path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 7  # 7 - путь
        
        # Обновляем изображение с расширенной цветовой картой
        colors = ['white', 'black', 'green', 'red', 'yellow', 'lightblue', 'purple', 'orange']
        img.set_cmap(ListedColormap(colors))
        img.set_data(grid_vis)
        
        # Обновляем заголовок
        ax.set_title(f'Spore Algorithm (Итерация {self.iteration}/{self.max_iterations})')
        
        plt.draw()
        plt.pause(0.01)  # Пауза для обновления отображения
    
    def visualize_result(self, show_explored=True):
        """
        Визуализирует результат поиска пути.
        
        Параметры:
        - show_explored: если True, показывает исследованные клетки
        """
        plt.figure(figsize=(10, 10))
        
        # Создаем копию карты для визуализации
        grid_vis = self.grid.copy()
        
        # Отмечаем исследованные клетки
        if show_explored:
            for x, y in self.explored_cells:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем споры
        for x, y, can_reach, marked in self.spores:
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
        colors = ['white', 'black', 'green', 'red', 'yellow', 'lightblue', 'purple', 'orange']
        cmap = ListedColormap(colors)
        
        # Отображаем карту
        plt.imshow(grid_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        # Заголовок
        plt.title(f'Результат поиска пути Spore (Итераций: {self.iteration})')
        
        plt.show()


if __name__ == "__main__":
    # Пример использования
    from comparison.map_generator_0 import MapGenerator
    
    # Генерируем карту
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Визуализируем исходную карту
    generator.visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Создаем Spore агент
    agent = SporeAgent(grid, start_pos, finish_pos, max_iterations=100)
    
    # Находим путь с визуализацией
    path, time_elapsed, steps = agent.find_path(visualize=True)
    
    if path:
        print(f"Путь найден за {time_elapsed:.4f} секунд, {steps} итераций")
        print(f"Длина пути: {len(path)}")
    else:
        print(f"Путь не найден за {time_elapsed:.4f} секунд, {steps} итераций")
    
    # Визуализируем результат
    agent.visualize_result()