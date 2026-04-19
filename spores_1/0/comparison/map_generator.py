import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pickle
from collections import deque

class MapGenerator:
    def __init__(self, size=(30, 30), wall_percent=None):
        self.height, self.width = size
        if wall_percent is None:
            self.wall_count = int(0.3 * self.width * self.height)
        else:
            self.wall_count = int(wall_percent * self.width * self.height / 100)
        self.wall_count = min(self.wall_count, self.width * self.height - 2)
    
    def generate_map(self):
        """Генерирует карту с проходимым путем от старта до финиша"""
        grid = np.zeros((self.height, self.width), dtype=int)
        start_pos = (1, 1)
        finish_pos = (self.height - 2, self.width - 2)
        grid[start_pos] = 2  # Старт
        grid[finish_pos] = 3  # Финиш
        
        walls_placed = 0
        positions = [(x, y) for x in range(self.height) for y in range(self.width) 
                    if (x, y) != start_pos and (x, y) != finish_pos]
        random.shuffle(positions)
        
        for pos in positions:
            if walls_placed >= self.wall_count:
                break
            grid[pos] = 1
            if not self._path_exists(grid, start_pos, finish_pos):
                grid[pos] = 0
            else:
                walls_placed += 1
        
        return grid, start_pos, finish_pos
    
    def _path_exists(self, grid, start_pos, finish_pos):
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
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    grid[next_pos] != 1 and 
                    next_pos not in visited):
                    visited.add(next_pos)
                    queue.append(next_pos)
        return False

def save_map_data(grid, start_pos, finish_pos, file_path):
    """Сохраняет карту и координаты в файл"""
    data = {
        'grid': grid,
        'start_pos': start_pos,
        'finish_pos': finish_pos
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def generate_maze_collection():
    # Список размеров (высота, ширина)
    sizes = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (100, 100)]
    
    # Список процентов заполнения стенами
    wall_percents = [10, 20, 30, 40, 50, 60]
    
    # Создаем основную папку для всех лабиринтов
    base_folder = "maze_collection"
    os.makedirs(base_folder, exist_ok=True)
    
    # Проходим по всем комбинациям размеров и процентов
    for size in sizes:
        height, width = size
        size_folder = f"{base_folder}/maze_{height}x{width}"
        os.makedirs(size_folder, exist_ok=True)
        
        for percent in wall_percents:
            # Генерируем и сохраняем лабиринт
            generator = MapGenerator(size=size, wall_percent=percent)
            grid, start_pos, finish_pos = generator.generate_map()
            
            # Формируем имя файла
            file_name = f"{size_folder}/maze_{height}x{width}_{percent}percent.pkl"
            
            # Сохраняем лабиринт
            save_map_data(grid, start_pos, finish_pos, file_name)
            print(f"Сгенерирован лабиринт: {height}x{width}, {percent}% стен")

def main():
    print("Начинаем генерацию коллекции лабиринтов...")
    generate_maze_collection()
    print("Генерация завершена!")
    print("Лабиринты сохранены в папке 'maze_collection'")
    print("Структура: maze_collection/maze_HxW/maze_HxW_Ppercent.pkl")
    print("где H - высота, W - ширина, P - процент заполнения")

if __name__ == "__main__":
    main()