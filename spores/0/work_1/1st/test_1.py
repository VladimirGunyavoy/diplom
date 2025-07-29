import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

class MazePathPlanner:
    def __init__(self, size=50, obstacle_density=0.3):
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
        
        # Ищем свободные клетки, не окруженные со всех сторон стенами
        valid_goal_cells = []
        for i, j in free_cells:
            # Проверяем, что хотя бы одна из соседних клеток свободна (не стена)
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            has_free_neighbor = False
            
            for ni, nj in neighbors:
                if 0 <= ni < size and 0 <= nj < size and self.maze[ni, nj] != 1:
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
            goal_idx = random.choice(range(len(free_cells)))
            self.goal = free_cells[goal_idx]
            
            # Очищаем хотя бы одну соседнюю клетку от стены
            neighbors = [(self.goal[0]-1, self.goal[1]), 
                         (self.goal[0]+1, self.goal[1]), 
                         (self.goal[0], self.goal[1]-1), 
                         (self.goal[0], self.goal[1]+1)]
            
            for ni, nj in neighbors:
                if 0 <= ni < size and 0 <= nj < size and (ni, nj) != self.start:
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
        
        # Счетчик итераций
        self.iteration = 0
        
        # Максимальное количество итераций
        self.max_iterations = 200
        
        # Найден ли путь
        self.path_found = False
        
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
    
    def step(self):
        """Выполняет один шаг алгоритма - создает новую спору и запускает из нее агентов"""
        if self.iteration >= self.max_iterations or self.path_found:
            return False, f"{'Путь найден!' if self.path_found else 'Достигнуто максимальное число итераций'}"
        
        # Выбираем новую спору
        if len(self.visited) > 1:
            self.current_spore = random.choice(list(self.visited))
        else:
            self.current_spore = self.start
        
        # Запускаем случайные последовательности из выбранной споры
        paths, reached_goal = self.run_random_sequence(self.current_spore)
        
        # Обновляем множество точек, из которых можно достичь цель
        self.update_goal_reaching_points()
        
        # Проверяем, найден ли путь
        if self.goal in self.reachability_graph or reached_goal:
            self.path_found = True
            return True, f"Путь найден на итерации {self.iteration + 1}!"
        
        self.iteration += 1
        return True, f"Итерация {self.iteration}/{self.max_iterations}, спора: {self.current_spore}, путей из неё: {len(paths)}"
    
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
    
    def create_visualization(self):
        """Создает интерактивную визуализацию лабиринта и алгоритма"""
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
            
            # Текущая спора имеет особую метку (6)
            if self.current_spore != self.start and self.current_spore != self.goal:
                maze_vis[self.current_spore] = 6
            
            # Создаем цветовую карту
            colors = ['white', 'black', 'green', 'red', 'lightblue', 'yellow', 'orange']
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
            
            # Отображаем найденный путь, если он есть
            final_path = self.reconstruct_path()
            if final_path:
                xs = [y for x, y in final_path]  # Обратите внимание на обмен x и y из-за особенностей imshow
                ys = [x for x, y in final_path]
                ax.plot(xs, ys, 'r-', linewidth=2)
            
            # Дополнительно отмечаем старт, цель и текущую спору для лучшей видимости
            ax.plot(self.start[1], self.start[0], 'go', markersize=10)
            ax.plot(self.goal[1], self.goal[0], 'ro', markersize=10)
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
                mpatches.Patch(color='orange', label='Текущая спора')
            ]
            ax.legend(handles=legend_elements, loc='upper center', 
                     bbox_to_anchor=(0.5, 1.15), ncol=3)
            
            ax.set_title(f'Лабиринт и поиск пути (Итерация {self.iteration}/{self.max_iterations})')
            
            plt.tight_layout()
            fig.canvas.draw_idle()
        
        # Функция для выполнения шага при нажатии кнопки
        def on_key(event):
            if event.key == 'n':
                continue_sim, message = self.step()
                update_plot()
                plt.suptitle(message, fontsize=12)
                if not continue_sim:
                    print(message)
            elif event.key == 'q':
                plt.close(fig)
                print("Программа завершена пользователем.")
        
        # Добавляем обработчик клавиш
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Создаем кнопку для выполнения шага (альтернатива клавише 'f')
        ax_button = plt.axes([0.7, 0.01, 0.2, 0.05])
        btn = Button(ax_button, 'Новая спора (n)')
        
        def on_button_click(event):
            continue_sim, message = self.step()
            update_plot()
            plt.suptitle(message, fontsize=12)
            if not continue_sim:
                print(message)
        
        btn.on_clicked(on_button_click)
        
        # Инициализируем начальное отображение
        update_plot()
        
        plt.suptitle(f"Нажмите 'n' для новой споры, 'q' для выхода. Итерация {self.iteration}/{self.max_iterations}", 
                    fontsize=12)
        
        plt.show()

# Пример использования
if __name__ == "__main__":
    # Создаем планировщик с лабиринтом 50x50
    planner = MazePathPlanner(size=50, obstacle_density=0.3)
    
    # Запускаем интерактивную визуализацию
    planner.create_visualization()