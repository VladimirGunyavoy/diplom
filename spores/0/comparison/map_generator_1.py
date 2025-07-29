import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from collections import deque
import pickle
import os

class MapGenerator:
    def __init__(self, size=(30, 30), wall_count=None):
        self.height, self.width = size
        if wall_count is None:
            self.wall_count = int(0.3 * self.width * self.height)
        else:
            self.wall_count = min(wall_count, self.width * self.height - 2)
    
    def generate_map(self):
        """Генерирует карту с проходимым путем от старта до финиша"""
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
        """Проверяет существование пути между стартом и финишем с помощью BFS"""
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
        """Визуализирует карту"""
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
        
        plt.show()

def save_map_data(grid, start_pos, finish_pos, file_path="map_data.pkl"):
    """Сохраняет карту и координаты в файл"""
    data = {
        'grid': grid,
        'start_pos': start_pos,
        'finish_pos': finish_pos
    }
    
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    
    # Сохраняем данные в файл
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Карта сохранена в файл {file_path}")
    print(f"Размер карты: {grid.shape}")
    print(f"Старт: {start_pos}, Финиш: {finish_pos}")
    print(f"Стен: {np.sum(grid == 1)}")

def main():
    # Настройки генерации карты
    bord = 100
    size = (bord, bord)  # Размер карты
    wall_count = 2000  # Количество стен
    file_path = "map_data.pkl"  # Путь к файлу для сохранения
    
    # Генерируем карту
    print(f"Генерация карты размера {size}...")
    generator = MapGenerator(size=size, wall_count=wall_count)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Сохраняем карту в файл
    save_map_data(grid, start_pos, finish_pos, file_path)
    
    # Визуализируем сгенерированную карту
    print("Визуализация карты...")
    generator.visualize_map(grid, start_pos, finish_pos, title="Сгенерированная карта")
    
    print("Готово!")

if __name__ == "__main__":
    main()