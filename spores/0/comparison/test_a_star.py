import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
import heapq
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

def visualize_map(grid, start_pos, finish_pos, path=None, explored=None, title=None):
    """Визуализирует карту с путем и исследованными клетками"""
    plt.figure(figsize=(12, 10))
    
    # Создаем копию карты для визуализации
    grid_vis = grid.copy()
    
    # Если есть исследованные клетки, отмечаем их
    if explored:
        for x, y in explored:
            if grid_vis[x, y] == 0:  # Только свободные клетки
                grid_vis[x, y] = 4  # 4 - исследованные клетки
    
    # Если путь указан, отмечаем его на карте
    if path:
        for x, y in path:
            if (x, y) != start_pos and (x, y) != finish_pos:  # Не перезаписываем старт и финиш
                grid_vis[x, y] = 5  # 5 - путь
    
    # Убедимся, что старт и финиш имеют правильные значения
    grid_vis[start_pos] = 2
    grid_vis[finish_pos] = 3
    
    # Создаем цветовую карту
    colors = ['white', 'darkgreen', 'blue', 'red', 'lightblue', 'orange']
    cmap = ListedColormap(colors)
    
    # Отображаем карту
    plt.imshow(grid_vis, cmap=cmap)
    plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    # Создаем легенду
    legend_elements = [
        mpatches.Patch(color='white', label='Свободный путь'),
        mpatches.Patch(color='darkgreen', label='Стена'),
        mpatches.Patch(color='blue', label='Старт'),
        mpatches.Patch(color='red', label='Финиш'),
        mpatches.Patch(color='lightblue', label='Исследованные клетки'),
        mpatches.Patch(color='orange', label='Найденный путь')
    ]
    plt.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Заголовок
    if title:
        plt.title(title)
    else:
        plt.title(f'Карта {grid.shape[0]}x{grid.shape[1]}')
    
    plt.tight_layout()
    plt.show()

class AStarPathfinder:
    """Реализация алгоритма A* для поиска пути"""
    
    def __init__(self, grid, start_pos, finish_pos):
        self.grid = grid
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.height, self.width = grid.shape
        self.explored_cells = set()  # Множество для отслеживания исследованных клеток
    
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
                
                print(f"A* исследовал {nodes_explored} узлов ({len(self.explored_cells)} уникальных клеток)")
                
                return path, elapsed_time
            
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
        print(f"A* исследовал {nodes_explored} узлов, но путь не найден")
        return None, elapsed_time

def main():
    # Загружаем карту
    file_path = "map_data.pkl"
    grid, start_pos, finish_pos = load_map_data(file_path)
    
    # Отключаем визуализацию исходной карты
    print("Визуализация исходной карты...")
    visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Создаем экземпляр A* и ищем путь
    print("Поиск пути с помощью алгоритма A*...")
    pathfinder = AStarPathfinder(grid, start_pos, finish_pos)
    path, elapsed_time = pathfinder.find_path()
    
    if path:
        print(f"Путь найден за {elapsed_time:.6f} секунд!")
        print(f"Длина пути: {len(path)}")
        
        # Отключаем визуализацию результата
        print("Визуализация найденного пути...")
        # visualize_map(grid, start_pos, finish_pos, path=path, 
        #              explored=pathfinder.explored_cells,
        #              title=f"A* (время: {elapsed_time:.6f} с, длина пути: {len(path)})")

if __name__ == "__main__":
    main()