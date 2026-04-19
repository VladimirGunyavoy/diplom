import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

class MazePathPlanner:
    def __init__(self, size=50, obstacle_density=0.3):
        self.size = size
        self.obstacle_density = obstacle_density
        self.initialize_maze()
        
        # Множество для хранения уникальных спор
        self.used_spores = set()
        
    def initialize_maze(self):
        # Создаем лабиринт: 0 - свободный путь, 1 - стена, 2 - начальная точка, 3 - цель
        self.maze = np.zeros((self.size, self.size))
        self.generate_maze(self.obstacle_density)
        
        # Находим свободные клетки для размещения начальной точки и цели
        free_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.maze[i, j] == 0]
        
        # Находим клетки с хотя бы одним свободным соседом
        valid_start_cells = []
        for i, j in free_cells:
            # Проверяем, что хотя бы одна из соседних клеток свободна (не стена)
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            has_free_neighbor = False
            
            for ni, nj in neighbors:
                if 0 <= ni < self.size and 0 <= nj < self.size and self.maze[ni, nj] != 1:
                    has_free_neighbor = True
                    break
                    
            if has_free_neighbor:
                valid_start_cells.append((i, j))
        
        # Если есть подходящие клетки, выбираем одну случайно для старта
        if valid_start_cells:
            start_idx = random.choice(range(len(valid_start_cells)))
            self.start = valid_start_cells[start_idx]
        else:
            # Если нет подходящих клеток, выбираем любую свободную и очищаем место рядом
            start_idx = random.choice(range(len(free_cells)))
            self.start = free_cells[start_idx]
            
            # Очищаем хотя бы одну соседнюю клетку от стены
            neighbors = [(self.start[0]-1, self.start[1]), 
                          (self.start[0]+1, self.start[1]), 
                          (self.start[0], self.start[1]-1), 
                          (self.start[0], self.start[1]+1)]
            
            for ni, nj in neighbors:
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    self.maze[ni, nj] = 0  # Убираем стену
                    break
        
        self.maze[self.start] = 2
        
        # Удаляем начальную точку из списка свободных клеток
        if self.start in free_cells:
            free_cells.remove(self.start)
        
        # Выбираем цель на расстоянии не менее половины размера карты
        min_distance = self.size / 2
        valid_goal_cells = []
        
        for i, j in free_cells:
            dist = np.sqrt((i - self.start[0])**2 + (j - self.start[1])**2)
            if dist >= min_distance:
                # Проверяем, что хотя бы одна из соседних клеток свободна (не стена)
                neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                has_free_neighbor = False
                
                for ni, nj in neighbors:
                    if 0 <= ni < self.size and 0 <= nj < self.size and self.maze[ni, nj] != 1:
                        has_free_neighbor = True
                        break
                        
                if has_free_neighbor:
                    valid_goal_cells.append((i, j))
        
        # Если есть подходящие клетки, выбираем одну случайно
        if valid_goal_cells:
            goal_idx = random.choice(range(len(valid_goal_cells)))
            self.goal = valid_goal_cells[goal_idx]
        else:
            # Если нет подходящих клеток, выбираем любую свободную и очищаем место рядом
            goal_cells = [cell for cell in free_cells if 
                          np.sqrt((cell[0] - self.start[0])**2 + (cell[1] - self.start[1])**2) >= min_distance / 2]
            
            if goal_cells:
                goal_idx = random.choice(range(len(goal_cells)))
                self.goal = goal_cells[goal_idx]
            else:
                # В крайнем случае берем любую
                goal_idx = random.choice(range(len(free_cells)))
                self.goal = free_cells[goal_idx]
            
            # Очищаем хотя бы одну соседнюю клетку от стены
            neighbors = [(self.goal[0]-1, self.goal[1]), 
                         (self.goal[0]+1, self.goal[1]), 
                         (self.goal[0], self.goal[1]-1), 
                         (self.goal[0], self.goal[1]+1)]
            
            for ni, nj in neighbors:
                if 0 <= ni < self.size and 0 <= nj < self.size and (ni, nj) != self.start:
                    self.maze[ni, nj] = 0  # Убираем стену
                    break
        
        self.maze[self.goal] = 3
        
        # Граф достижимости: точки и из каких точек они достижимы
        self.reachability_graph = {self.start: None}  # None означает, что это начальная точка
        
        # Точки, из которых можно достичь цели (прямо или через другие точки)
        self.can_reach_goal = set()
        
        # Посещенные точки для визуализации
        self.visited = set([self.start])
        
        # История путей для визуализации
        self.path_history = []
        
        # Текущая спора (точка из которой выпускается агент)
        self.current_spore = self.start
        
        # Список всех использованных спор
        self.all_spores = [self.start]
        
        # Счетчик итераций
        self.iteration = 0
        
        # Максимальное количество итераций
        self.max_iterations = 200
        
        # Найден ли путь
        self.path_found = False
        
        # Для пошагового просмотра найденного пути
        self.path_display_step = 0
        self.final_path = None
        
        # Флаг для отображения пути (может быть включен/выключен)
        self.show_path = True
        
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
    
    def run_random_sequence(self, start_point, sequence_length=10, num_sequences=8):
        """Запускает несколько случайных последовательностей шагов из заданной точки"""
        # Определяем цель для матожидания
        if self.can_reach_goal:
            target = self.get_nearest_goal_reaching_point(start_point)
        else:
            target = self.goal
            
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
                    
                    # Отмечаем точку как посещенную
                    self.visited.add(current)
                    
                    # Если достигли цели
                    if current == self.goal:
                        self.can_reach_goal.add(start_point)
                        reached_goal = True
                    
                    # Если достигли точки, из которой можно достичь цель
                    if current in self.can_reach_goal:
                        self.can_reach_goal.add(start_point)
                        reached_goal = True
            
            # Сохраняем путь для визуализации
            current_paths.append(path)
        
        self.path_history.extend(current_paths)
        return current_paths, reached_goal
    
    def update_goal_reaching_points(self):
        """Обновляет множество точек, из которых можно достичь цель,
        включая точки, ведущие к уже известным точкам"""
        changed = True
        max_iterations = 100  # Ограничение на количество итераций
        iterations = 0
        
        while changed and iterations < max_iterations:
            changed = False
            new_points = set()
            iterations += 1
            
            # Проходим по всем точкам в графе достижимости
            for point, parent in self.reachability_graph.items():
                # Если точка уже помечена как ведущая к цели, пропускаем
                if point in self.can_reach_goal:
                    continue
                
                # Если из этой точки можно попасть в точку, ведущую к цели
                for goal_point in self.can_reach_goal:
                    # Проверяем, что goal_point есть в графе достижимости
                    if goal_point in self.reachability_graph and point != self.goal:
                        # Проверяем, что point является родителем для goal_point
                        if self.reachability_graph[goal_point] == point:
                            new_points.add(point)
                            changed = True
                            break
            
            # Добавляем новые точки в множество
            self.can_reach_goal.update(new_points)
            
        if iterations >= max_iterations:
            print(f"Предупреждение: достигнуто максимальное число итераций ({max_iterations}) в update_goal_reaching_points()")
    
    def step(self):
        """Выполняет один шаг алгоритма - создает новую спору и запускает из нее агентов"""
        if self.iteration >= self.max_iterations:
            return False, f"Достигнуто максимальное число итераций"
        
        message_prefix = "Путь найден! " if self.path_found else ""
        
        # Выбираем новую спору только из непосещенных и неиспользованных клеток
        free_cells = [(i, j) for i in range(self.size) for j in range(self.size) 
                    if self.maze[i, j] == 0 and (i, j) not in self.used_spores and (i, j) not in self.visited]
        
        if not free_cells:
            return False, "Нет уникальных непосещенных клеток для создания новой споры."
        
        # Выбираем случайную клетку для новой споры
        self.current_spore = random.choice(free_cells)
        
        # Добавляем спору в список всех спор и в множество использованных
        self.all_spores.append(self.current_spore)
        self.used_spores.add(self.current_spore)
        
        # Для отладки: выводим текущую спору и размер множества использованных спор
        print(f"Новая спора: {self.current_spore}, всего использовано спор: {len(self.used_spores)}")
        
        # Запускаем случайные последовательности из выбранной споры
        paths, reached_goal = self.run_random_sequence(self.current_spore)
        
        # Обновляем множество точек, из которых можно достичь цель
        self.update_goal_reaching_points()
        
        # Проверяем, найден ли путь (если еще не найден)
        if not self.path_found and (self.goal in self.reachability_graph or reached_goal):
            self.path_found = True
            self.final_path = self.reconstruct_path()
            self.path_display_step = 0
            return True, f"{message_prefix}Путь найден на итерации {self.iteration + 1}!"
        
        self.iteration += 1
        return True, f"{message_prefix}Итерация {self.iteration}/{self.max_iterations}, спора: {self.current_spore}, путей из неё: {len(paths)}"
    
    def reset(self):
        """Сбрасывает состояние алгоритма и создает новый лабиринт"""
        self.initialize_maze()
        return True, "Алгоритм перезапущен с новым лабиринтом!"
    
    def reconstruct_path(self):
        """Восстанавливает путь от начала до цели, используя граф достижимости"""
        if self.goal not in self.reachability_graph:
            print("Цель не достигнута. Пытаемся найти альтернативный путь...")
            # Пробуем найти альтернативный путь
            goal_reachable_points = [point for point in self.can_reach_goal if point in self.reachability_graph]
            
            if not goal_reachable_points:
                print("Нет точек, из которых можно достичь цели.")
                return None
            
            # Выбираем ближайшую к цели точку
            nearest_point = min(goal_reachable_points, 
                            key=lambda p: np.sqrt((p[0] - self.goal[0])**2 + (p[1] - self.goal[1])**2))
            
            # Строим путь от начала до ближайшей точки
            path = [nearest_point]
            current = nearest_point
            
            # Безопасное построение пути с проверкой циклов
            visited = {current}
            max_steps = len(self.reachability_graph) + 1  # Защита от бесконечного цикла
            step_count = 0
            
            try:
                while current != self.start and step_count < max_steps:
                    if current not in self.reachability_graph:
                        print(f"Ошибка: узел {current} отсутствует в графе достижимости")
                        return None
                    
                    current = self.reachability_graph[current]
                    
                    # Проверка на циклы
                    if current in visited:
                        print(f"Обнаружен цикл в пути при прохождении через узел {current}")
                        return None
                    
                    visited.add(current)
                    path.append(current)
                    step_count += 1
                
                if step_count >= max_steps:
                    print("Превышено максимальное количество шагов при построении пути")
                    return None
                
                # Добавляем путь к цели (прямая линия)
                path.reverse()
                print(f"Путь восстановлен: {path + [self.goal]}")
                return path + [self.goal]  # Добавляем цель в конец пути
            except Exception as e:
                print(f"Ошибка при восстановлении пути: {e}")
                return None
                    
        # Стандартное построение пути, если цель в графе достижимости
        path = [self.goal]
        current = self.goal
        
        # Безопасное построение пути с проверкой циклов
        visited = {current}
        max_steps = len(self.reachability_graph) + 1  # Защита от бесконечного цикла
        step_count = 0
        
        try:
            while current != self.start and step_count < max_steps:
                if current not in self.reachability_graph:
                    print(f"Ошибка: узел {current} отсутствует в графе достижимости")
                    return None
                
                current = self.reachability_graph[current]
                
                # Проверка на циклы
                if current in visited:
                    print(f"Обнаружен цикл в пути при прохождении через узел {current}")
                    return None
                
                visited.add(current)
                path.append(current)
                step_count += 1
            
            if step_count >= max_steps:
                print("Превышено максимальное количество шагов при построении пути")
                return None
                
            path.reverse()
            print(f"Путь восстановлен: {path}")
            return path
        except Exception as e:
            print(f"Ошибка при восстановлении пути: {e}")
            return None



    def create_visualization(self):
            """Создает интерактивную визуализацию лабиринта и алгоритма"""
            plt.ion()  # Включаем интерактивный режим
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Функция для обновления визуализации
            def update_plot():
                ax.clear()
                
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
                
                # Отмечаем все споры (кроме текущей) специальной меткой (7)
                for spore in self.all_spores:
                    if spore != self.current_spore and spore != self.start and spore != self.goal:
                        if maze_vis[spore] == 0 or maze_vis[spore] == 4 or maze_vis[spore] == 5:
                            maze_vis[spore] = 7
                
                # Текущая спора имеет особую метку (6)
                if self.current_spore != self.start and self.current_spore != self.goal:
                    maze_vis[self.current_spore] = 6
                
                # Подсвечиваем споры, через которые можно пройти к финишу (метка 9)
                for spore in self.can_reach_goal:
                    if spore in self.all_spores:
                        maze_vis[spore] = 9
                
                # Создаем цветовую карту
                colors = ['white', 'black', 'green', 'red', 'lightblue', 'yellow', 'orange', 'purple', 'gray', 'pink']
                cmap = ListedColormap(colors)
                
                # Отображаем лабиринт
                ax.imshow(maze_vis, cmap=cmap)
                ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
                
                # Отображаем последние пути (только из текущей споры)
                if self.path_history:
                    for path in self.path_history[-10:]:  # Показываем только 10 последних путей для наглядности
                        if len(path) > 0 and path[0] == self.current_spore:
                            xs = [y for x, y in path]  # Обратите внимание на обмен x и y из-за особенностей imshow
                            ys = [x for x, y in path]
                            ax.plot(xs, ys, 'b-', alpha=0.3)
                
                # Отображаем найденный путь, если он есть и флаг отображения пути включен
                if self.show_path:
                    final_path = self.reconstruct_path()
                    if final_path:
                        xs = [y for x, y in final_path]  # Обратите внимание на обмен x и y из-за особенностей imshow
                        ys = [x for x, y in final_path]
                        ax.plot(xs, ys, 'r-', linewidth=2)
                
                # Дополнительно отмечаем старт, цель и все споры для лучшей видимости
                ax.plot(self.start[1], self.start[0], 'go', markersize=10)
                ax.plot(self.goal[1], self.goal[0], 'ro', markersize=10)
                
                # Отмечаем все предыдущие споры
                for spore in self.all_spores:
                    if spore != self.current_spore and spore != self.start and spore != self.goal:
                        ax.plot(spore[1], spore[0], 'o', color='purple', markersize=6)
                
                # Отмечаем текущую спору
                if self.current_spore != self.start and self.current_spore != self.goal:
                    ax.plot(self.current_spore[1], self.current_spore[0], 'o', color='orange', markersize=8)
                
                # Подготовка легенды
                legend_elements = [
                    mpatches.Patch(color='white', label='Свободный путь'),
                    mpatches.Patch(color='black', label='Стена'),
                    mpatches.Patch(color='green', label='Старт'),
                    mpatches.Patch(color='red', label='Цель'),
                    mpatches.Patch(color='lightblue', label='Посещенные клетки'),
                    mpatches.Patch(color='yellow', label='Клетки, ведущие к цели'),
                    mpatches.Patch(color='orange', label='Текущая спора'),
                    mpatches.Patch(color='purple', label='Предыдущие споры'),
                    mpatches.Patch(color='pink', label='Споры, ведущие к цели')
                ]
                ax.legend(handles=legend_elements, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.15), ncol=3)
                
                ax.set_title(f'Лабиринт и поиск пути (Итерация {self.iteration}/{self.max_iterations})')
                
                try:
                    plt.tight_layout()
                    fig.canvas.draw_idle()
                except Exception as e:
                    print(f"Ошибка при обновлении графика: {e}")
            
            # Функция для выполнения шага при нажатии кнопки
            def on_key(event):
                try:
                    if event.key == 'n':
                        continue_sim, message = self.step()
                        update_plot()
                        plt.suptitle(message, fontsize=12)
                        if not continue_sim:
                            print(message)
                    elif event.key == 'r':
                        continue_sim, message = self.reset()
                        update_plot()
                        plt.suptitle(message, fontsize=12)
                    elif event.key == 'p':
                        # Переключаем режим отображения пути
                        self.show_path = not self.show_path
                        update_plot()
                        plt.suptitle(f"{'Путь скрыт' if not self.show_path else 'Путь отображается'}. Нажмите 'p' для переключения.", fontsize=12)
                    elif event.key == 'q':
                        plt.close(fig)
                        print("Программа завершена пользователем.")
                except Exception as e:
                    print(f"Ошибка при обработке нажатия клавиши: {e}")
            
            # Добавляем обработчик клавиш
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            # Создаем кнопки для управления
            ax_button_next = plt.axes([0.7, 0.01, 0.15, 0.05])
            btn_next = Button(ax_button_next, 'Новая спора (n)')
            
            ax_button_reset = plt.axes([0.5, 0.01, 0.15, 0.05])
            btn_reset = Button(ax_button_reset, 'Перезапуск (r)')
            
            ax_button_path = plt.axes([0.3, 0.01, 0.15, 0.05])
            btn_path = Button(ax_button_path, 'Скрыть/показать путь (p)')
            
            def on_button_next_click(event):
                try:
                    continue_sim, message = self.step()
                    update_plot()
                    plt.suptitle(message, fontsize=12)
                    if not continue_sim:
                        print(message)
                except Exception as e:
                    print(f"Ошибка при обработке нажатия кнопки 'Новая спора': {e}")
            
            def on_button_reset_click(event):
                try:
                    continue_sim, message = self.reset()
                    update_plot()
                    plt.suptitle(message, fontsize=12)
                except Exception as e:
                    print(f"Ошибка при обработке нажатия кнопки 'Перезапуск': {e}")
                    
            def on_button_path_click(event):
                try:
                    self.show_path = not self.show_path
                    update_plot()
                    plt.suptitle(f"{'Путь скрыт' if not self.show_path else 'Путь отображается'}. Нажмите 'p' для переключения.", fontsize=12)
                except Exception as e:
                    print(f"Ошибка при обработке нажатия кнопки 'Скрыть/показать путь': {e}")
            
            btn_next.on_clicked(on_button_next_click)
            btn_reset.on_clicked(on_button_reset_click)
            btn_path.on_clicked(on_button_path_click)
            
            # Инициализируем начальное отображение
            try:
                update_plot()
                
                plt.suptitle(f"Нажмите 'n' для новой споры, 'r' для перезапуска, 'p' для скрытия/показа пути, 'q' для выхода.", 
                            fontsize=12)
                
                plt.show(block=True)  # block=True нужен для предотвращения возврата функции до закрытия окна
            except Exception as e:
                print(f"Ошибка при инициализации визуализации: {e}")


# Пример использования
if __name__ == "__main__":
    # Создаем планировщик с лабиринтом 50x50
    planner = MazePathPlanner(size=20, obstacle_density=0.3)
    
    # Запускаем интерактивную визуализацию
    planner.create_visualization()