import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import matplotlib.animation as animation

class AStarAgent:
    """
    Агент поиска пути с использованием алгоритма A*.
    """
    
    def __init__(self, grid, start_pos, finish_pos):
        """
        Инициализация A* агента.
        
        Параметры:
        - grid: матрица карты (0 - свободное пространство, 1 - стена, 2 - старт, 3 - финиш)
        - start_pos: кортеж (x, y) - координаты стартовой позиции
        - finish_pos: кортеж (x, y) - координаты целевой позиции
        """
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        
        # Для визуализации
        self.explored_cells = []  # Список посещенных клеток
        self.path = []  # Найденный путь
        
        # Направления движения (вверх, вниз, влево, вправо)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def heuristic(self, pos):
        """
        Эвристическая функция (манхэттенское расстояние).
        
        Параметры:
        - pos: текущая позиция
        
        Возвращает:
        - расстояние от текущей позиции до цели
        """
        return abs(pos[0] - self.finish_pos[0]) + abs(pos[1] - self.finish_pos[1])
    
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
    
    def find_path(self, visualize=False):
        """
        Находит путь от старта до финиша с помощью алгоритма A*.
        
        Параметры:
        - visualize: если True, выполняет визуализацию процесса поиска
        
        Возвращает:
        - path: список позиций, образующих путь (пустой список, если путь не найден)
        - time_elapsed: время, затраченное на поиск пути
        - steps: количество шагов алгоритма
        """
        # Замеряем время выполнения
        start_time = time.time()
        
        # Очищаем данные предыдущего запуска
        self.explored_cells = []
        self.path = []
        
        # Открытый список (приоритетная очередь)
        open_set = []
        heapq.heappush(open_set, (self.heuristic(self.start_pos), 0, self.start_pos))
        
        # Словарь для отслеживания лучшего пути
        came_from = {self.start_pos: None}
        
        # Словарь для хранения стоимости пути от старта до каждой позиции
        g_score = {self.start_pos: 0}
        
        # Счетчик шагов
        steps = 0
        
        # Настройка визуализации, если требуется
        if visualize:
            fig, ax = plt.subplots(figsize=(10, 10))
            img = ax.imshow(self.grid, cmap=ListedColormap(['white', 'black', 'green', 'red', 'yellow', 'lightblue']))
            plt.title('A* Algorithm')
            plt.ion()  # Включаем интерактивный режим
            plt.show()
        
        # Пока открытый список не пуст
        while open_set:
            steps += 1
            
            # Извлекаем позицию с наименьшей оценкой f_score
            f_score, _, current = heapq.heappop(open_set)
            
            # Добавляем текущую позицию в список исследованных клеток
            self.explored_cells.append(current)
            
            # Если достигли финиша, восстанавливаем путь
            if current == self.finish_pos:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                
                # Фиксируем затраченное время
                time_elapsed = time.time() - start_time
                
                # Сохраняем найденный путь
                self.path = path
                
                # Завершаем визуализацию, если она была включена
                if visualize:
                    # Визуализируем найденный путь
                    self._visualize_search(ax, img, True)
                    plt.ioff()  # Отключаем интерактивный режим
                    plt.show()
                
                return path, time_elapsed, steps
            
            # Исследуем соседей
            for dx, dy in self.directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                next_pos = (next_x, next_y)
                
                # Проверяем, что сосед находится в допустимой позиции
                if not self.is_valid_position(next_pos):
                    continue
                
                # Рассчитываем новую стоимость пути
                tentative_g_score = g_score[current] + 1
                
                # Если мы нашли лучший путь до этой позиции
                if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                    came_from[next_pos] = current
                    g_score[next_pos] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(next_pos)
                    heapq.heappush(open_set, (f_score, steps, next_pos))
            
            # Обновляем визуализацию, если требуется
            if visualize and steps % 10 == 0:  # Обновляем каждые 10 шагов
                self._visualize_search(ax, img, False)
        
        # Если открытый список пуст и мы не достигли финиша, путь не существует
        time_elapsed = time.time() - start_time
        
        # Завершаем визуализацию, если она была включена
        if visualize:
            plt.ioff()  # Отключаем интерактивный режим
            plt.show()
        
        return [], time_elapsed, steps
    
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
        
        # Отмечаем исследованные клетки
        for x, y in self.explored_cells:
            if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - исследованная клетка
        
        # Отмечаем найденный путь
        if final and self.path:
            for x, y in self.path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 4  # 4 - путь
        
        # Обновляем изображение
        img.set_data(grid_vis)
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
        
        # Отмечаем найденный путь
        if self.path:
            for x, y in self.path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 4  # 4 - путь
        
        # Создаем цветовую карту
        colors = ['white', 'black', 'green', 'red', 'yellow', 'lightblue']
        cmap = ListedColormap(colors)
        
        # Отображаем карту
        plt.imshow(grid_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        # Заголовок
        plt.title('Результат поиска пути A*')
        
        plt.show()


if __name__ == "__main__":
    # Пример использования
    from comparison.map_generator_0 import MapGenerator
    
    # Генерируем карту
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Визуализируем исходную карту
    generator.visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Создаем A* агент
    agent = AStarAgent(grid, start_pos, finish_pos)
    
    # Находим путь с визуализацией
    path, time_elapsed, steps = agent.find_path(visualize=True)
    
    if path:
        print(f"Путь найден за {time_elapsed:.4f} секунд, {steps} шагов")
        print(f"Длина пути: {len(path)}")
    else:
        print(f"Путь не найден за {time_elapsed:.4f} секунд, {steps} шагов")
    
    # Визуализируем результат
    agent.visualize_result()