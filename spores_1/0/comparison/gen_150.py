import os
import pickle
import random
import numpy as np

def generate_maze(width, height, wall_percent):
    # Создаем пустую сетку (0 — путь, 1 — стена)
    grid = np.zeros((height, width), dtype=int)
    
    # Устанавливаем границы как стены
    grid[0, :] = 1  # Верхняя граница
    grid[-1, :] = 1  # Нижняя граница
    grid[:, 0] = 1  # Левая граница
    grid[:, -1] = 1  # Правая граница
    
    # Вычисляем количество клеток для стен
    total_cells = width * height
    border_cells = 2 * (width + height - 2)  # Клетки границ
    inner_cells = total_cells - border_cells
    wall_cells = int(inner_cells * wall_percent / 100)
    
    # Заполняем внутренние клетки случайными стенами
    inner_positions = [(i, j) for i in range(1, height-1) for j in range(1, width-1)]
    random.shuffle(inner_positions)
    
    for pos in inner_positions[:wall_cells]:
        grid[pos[0], pos[1]] = 1
    
    # Устанавливаем стартовую и конечную позиции
    start_pos = (1, 1)  # Левый верхний угол
    finish_pos = (height-2, width-2)  # Правый нижний угол
    
    # Убедимся, что старт и финиш не стены
    grid[start_pos[0], start_pos[1]] = 0
    grid[finish_pos[0], finish_pos[1]] = 0
    
    return grid, start_pos, finish_pos

def save_maze(size, wall_percent):
    width, height = map(int, size.split('x'))
    grid, start_pos, finish_pos = generate_maze(width, height, wall_percent)
    
    # Создаем папку, если она не существует
    maze_folder = os.path.join("maze_collection", f"maze_{size}")
    os.makedirs(maze_folder, exist_ok=True)
    
    # Сохраняем в формате словаря, как ожидает load_map_data
    maze_data = {
        'grid': grid,
        'start_pos': start_pos,
        'finish_pos': finish_pos
    }
    
    # Сохраняем лабиринт в файл
    maze_file = f"maze_{size}_{wall_percent}percent.pkl"
    maze_path = os.path.join(maze_folder, maze_file)
    
    with open(maze_path, 'wb') as f:
        pickle.dump(maze_data, f)
    
    print(f"Сохранен лабиринт: {maze_path}")

# Генерируем лабиринты размером 150x150 с разными процентами стен
sizes = ["90x90"]
wall_percents = [10, 20, 30, 40, 50, 60]  # Примерные проценты, можно изменить

for size in sizes:
    for wall_percent in wall_percents:
        save_maze(size, wall_percent)

print("Генерация лабиринтов завершена!")