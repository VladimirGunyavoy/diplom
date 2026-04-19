import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import heapq
from collections import deque

class MapGenerator:
    """
    Генератор карт для тестирования алгоритмов поиска пути.
    
    Атрибуты:
    - size: размер карты (ширина, высота)
    - wall_count: количество стен на карте
    - min_distance: минимальное расстояние между стартом и финишем
    """
    
    def __init__(self, size=(50, 50), wall_count=None, min_distance=None):
        """
        Инициализация генератора карт.
        
        Параметры:
        - size: кортеж (ширина, высота) карты
        - wall_count: количество стен на карте (если None, используется 30% от площади)
        - min_distance: минимальное расстояние между стартом и финишем (если None, используется 60% от диагонали)
        """
        self.width, self.height = size
        
        # Если количество стен не указано, используем 30% от площади
        if wall_count is None:
            self.wall_count = int(0.3 * self.width * self.height)
        else:
            self.wall_count = min(wall_count, self.width * self.height - 2)  # Оставляем как минимум 2 клетки для старта и финиша
        
        # Если минимальное расстояние не указано, используем 60% от диагонали
        if min_distance is None:
            self.min_distance = int(0.6 * np.sqrt(self.width**2 + self.height**2))
        else:
            self.min_distance = min_distance
    
    def generate_map(self, ensure_path=True):
        """
        Генерирует карту с заданными параметрами.
        
        Параметры:
        - ensure_path: если True, убеждается, что между стартом и финишем существует путь
        
        Возвращает:
        - grid: матрица NxM, где 0 - пустое пространство, 1 - стена, 2 - старт, 3 - финиш
        - start_pos: кортеж (x, y) - координаты старта
        - finish_pos: кортеж (x, y) - координаты финиша
        """
        # Создаем пустую карту
        grid = np.zeros((self.height, self.width), dtype=int)
        
        # Выбираем позиции старта и финиша
        start_pos, finish_pos = self._place_start_finish()
        
        # Размещаем старт и финиш на карте
        grid[start_pos[0], start_pos[1]] = 2  # Старт
        grid[finish_pos[0], finish_pos[1]] = 3  # Финиш
        
        # Размещаем стены
        self._place_walls(grid, start_pos, finish_pos)
        
        # Проверяем, существует ли путь между стартом и финишем, если требуется
        if ensure_path:
            while not self._path_exists(grid, start_pos, finish_pos):
                # Если путь не существует, очищаем некоторые стены
                self._clear_path_obstacles(grid, start_pos, finish_pos)
        
        return grid, start_pos, finish_pos
    
    def _place_start_finish(self):
        """
        Выбирает позиции старта и финиша на достаточном расстоянии друг от друга.
        
        Возвращает:
        - start_pos: кортеж (x, y) - координаты старта
        - finish_pos: кортеж (x, y) - координаты финиша
        """
        # Выбираем случайную позицию для старта
        start_x = random.randint(0, self.height - 1)
        start_y = random.randint(0, self.width - 1)
        start_pos = (start_x, start_y)
        
        # Выбираем позицию для финиша на расстоянии не менее min_distance от старта
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            finish_x = random.randint(0, self.height - 1)
            finish_y = random.randint(0, self.width - 1)
            finish_pos = (finish_x, finish_y)
            
            # Вычисляем евклидово расстояние между стартом и финишем
            distance = np.sqrt((finish_x - start_x)**2 + (finish_y - start_y)**2)
            
            # Если расстояние достаточно большое, принимаем позицию
            if distance >= self.min_distance:
                return start_pos, finish_pos
            
            attempts += 1
        
        # Если не удалось найти подходящую позицию, размещаем финиш на максимальном расстоянии
        # (в противоположном углу)
        finish_x = self.height - 1 - start_x
        finish_y = self.width - 1 - start_y
        finish_pos = (finish_x, finish_y)
        
        return start_pos, finish_pos
    
    def _place_walls(self, grid, start_pos, finish_pos):
        """
        Размещает стены на карте.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        """
        # Создаем список всех позиций, кроме старта и финиша
        positions = [(x, y) for x in range(self.height) for y in range(self.width)
                    if (x, y) != start_pos and (x, y) != finish_pos]
        
        # Перемешиваем список позиций
        random.shuffle(positions)
        
        # Размещаем стены
        walls_placed = 0
        for pos in positions:
            if walls_placed >= self.wall_count:
                break
            
            grid[pos[0], pos[1]] = 1  # 1 - стена
            walls_placed += 1
        
        # Добавляем несколько линейных стен (вертикальных и горизонтальных)
        linear_walls_count = min(10, self.wall_count // 10)  # Максимум 10 линейных стен
        for _ in range(linear_walls_count):
            if walls_placed >= self.wall_count:
                break
                
            # Выбираем случайное направление (горизонтальное или вертикальное)
            is_horizontal = random.choice([True, False])
            
            if is_horizontal:
                # Горизонтальная стена
                row = random.randint(0, self.height - 1)
                start_col = random.randint(0, self.width - 10)  # Оставляем место для стены длиной до 10
                length = min(random.randint(5, 10), self.width - start_col)
                
                for col in range(start_col, start_col + length):
                    if (row, col) != start_pos and (row, col) != finish_pos and grid[row, col] == 0:
                        grid[row, col] = 1
                        walls_placed += 1
                        
                        if walls_placed >= self.wall_count:
                            break
            else:
                # Вертикальная стена
                col = random.randint(0, self.width - 1)
                start_row = random.randint(0, self.height - 10)  # Оставляем место для стены длиной до 10
                length = min(random.randint(5, 10), self.height - start_row)
                
                for row in range(start_row, start_row + length):
                    if (row, col) != start_pos and (row, col) != finish_pos and grid[row, col] == 0:
                        grid[row, col] = 1
                        walls_placed += 1
                        
                        if walls_placed >= self.wall_count:
                            break
    
    def _path_exists(self, grid, start_pos, finish_pos):
        """
        Проверяет, существует ли путь между стартом и финишем с помощью BFS.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        
        Возвращает:
        - True, если путь существует, иначе False
        """
        # Копируем карту для безопасности
        temp_grid = grid.copy()
        
        # Создаем очередь для BFS
        queue = deque([start_pos])
        visited = {start_pos}
        
        # Направления для соседей (вверх, вниз, влево, вправо)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            current = queue.popleft()
            
            # Если достигли финиша, путь существует
            if current == finish_pos:
                return True
            
            # Проверяем всех соседей
            for dx, dy in directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                
                # Проверяем, что сосед находится в пределах карты
                if 0 <= next_x < self.height and 0 <= next_y < self.width:
                    # Проверяем, что сосед не стена и не был посещен
                    if temp_grid[next_x, next_y] != 1 and (next_x, next_y) not in visited:
                        queue.append((next_x, next_y))
                        visited.add((next_x, next_y))
        
        # Если очередь пуста и мы не достигли финиша, путь не существует
        return False
    
    def _clear_path_obstacles(self, grid, start_pos, finish_pos):
        """
        Очищает некоторые препятствия, чтобы создать путь между стартом и финишем.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        """
        # Используем A* для поиска приблизительного пути
        path = self._find_approximate_path(grid, start_pos, finish_pos)
        
        if path:
            # Очищаем случайные препятствия вдоль приблизительного пути
            for pos in path:
                # С некоторой вероятностью очищаем препятствие
                if random.random() < 0.7:  # 70% вероятность очистки
                    grid[pos[0], pos[1]] = 0
        else:
            # Если не удалось найти приблизительный путь, очищаем случайные препятствия
            wall_positions = [(x, y) for x in range(self.height) for y in range(self.width)
                           if grid[x, y] == 1]
            
            # Очищаем 10% стен
            num_to_clear = max(1, len(wall_positions) // 10)
            for pos in random.sample(wall_positions, num_to_clear):
                grid[pos[0], pos[1]] = 0
    
    def _find_approximate_path(self, grid, start_pos, finish_pos):
        """
        Находит приблизительный путь между стартом и финишем, игнорируя препятствия.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        
        Возвращает:
        - список позиций, образующих приблизительный путь
        """
        # Эвристика - евклидово расстояние
        def heuristic(pos):
            return np.sqrt((pos[0] - finish_pos[0])**2 + (pos[1] - finish_pos[1])**2)
        
        # Приоритетная очередь для A*
        open_set = []
        heapq.heappush(open_set, (heuristic(start_pos), start_pos))
        
        # Словарь для отслеживания лучшего пути
        came_from = {start_pos: None}
        
        # Словарь для хранения стоимости пути от старта до каждой позиции
        g_score = {start_pos: 0}
        
        # Направления для соседей (вверх, вниз, влево, вправо)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # Если достигли финиша, восстанавливаем путь
            if current == finish_pos:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dx, dy in directions:
                next_x, next_y = current[0] + dx, current[1] + dy
                
                # Проверяем, что сосед находится в пределах карты
                if 0 <= next_x < self.height and 0 <= next_y < self.width:
                    next_pos = (next_x, next_y)
                    
                    # Рассчитываем новую стоимость пути
                    # Штрафуем за стены, но не запрещаем их
                    tentative_g_score = g_score[current] + (5 if grid[next_x, next_y] == 1 else 1)
                    
                    # Если мы нашли лучший путь до этой позиции
                    if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                        came_from[next_pos] = current
                        g_score[next_pos] = tentative_g_score
                        f_score = tentative_g_score + heuristic(next_pos)
                        heapq.heappush(open_set, (f_score, next_pos))
        
        # Если не нашли путь, возвращаем пустой список
        return []
    
    def visualize_map(self, grid, start_pos=None, finish_pos=None, path=None, title=None):
        """
        Визуализирует карту и, при наличии, путь.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта (если None, берется из карты)
        - finish_pos: позиция финиша (если None, берется из карты)
        - path: список позиций, образующих путь (если None, путь не отображается)
        - title: заголовок графика
        """
        plt.figure(figsize=(10, 10))
        
        # Создаем копию карты для визуализации
        grid_vis = grid.copy()
        
        # Если путь указан, отмечаем его на карте (значение 4)
        if path:
            for x, y in path:
                if grid_vis[x, y] not in [2, 3]:  # Не перезаписываем старт и финиш
                    grid_vis[x, y] = 4
        
        # Создаем цветовую карту
        colors = ['white', 'black', 'green', 'red', 'yellow']
        cmap = ListedColormap(colors)
        
        # Отображаем карту
        plt.imshow(grid_vis, cmap=cmap)
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        
        # Если указаны start_pos и finish_pos, отмечаем их на карте
        if start_pos:
            plt.plot(start_pos[1], start_pos[0], 'go', markersize=10)
        
        if finish_pos:
            plt.plot(finish_pos[1], finish_pos[0], 'ro', markersize=10)
        
        # Заголовок
        if title:
            plt.title(title)
        else:
            plt.title(f'Карта {self.height}x{self.width} с {np.sum(grid == 1)} стенами')
        
        plt.show()
    
    def save_map(self, grid, filename):
        """
        Сохраняет карту в файл.
        
        Параметры:
        - grid: матрица карты
        - filename: имя файла для сохранения
        """
        np.save(filename, grid)
    
    @staticmethod
    def load_map(filename):
        """
        Загружает карту из файла.
        
        Параметры:
        - filename: имя файла для загрузки
        
        Возвращает:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        """
        grid = np.load(filename)
        
        # Находим позиции старта и финиша
        start_pos = tuple(map(int, np.where(grid == 2)))
        if len(start_pos[0]) > 0:
            start_pos = (start_pos[0][0], start_pos[1][0])
        else:
            start_pos = None
        
        finish_pos = tuple(map(int, np.where(grid == 3)))
        if len(finish_pos[0]) > 0:
            finish_pos = (finish_pos[0][0], finish_pos[1][0])
        else:
            finish_pos = None
        
        return grid, start_pos, finish_pos


if __name__ == "__main__":
    # Пример использования
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    print(f"Карта {generator.height}x{generator.width}")
    print(f"Старт: {start_pos}, Финиш: {finish_pos}")
    print(f"Стен: {np.sum(grid == 1)}")
    
    # Визуализируем карту
    generator.visualize_map(grid, start_pos, finish_pos)