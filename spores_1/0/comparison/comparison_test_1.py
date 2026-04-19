import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from datetime import datetime
import random
import heapq
from collections import deque
import sys


# Импортируем классы из других файлов
# Если файлы находятся в текущей директории
try:
    from comparison.map_generator_0 import MapGenerator
    from a_star_agent import AStarAgent
    from spore_agent import SporeAgent
except ImportError:
    # Если файлы не импортируются, создаем их инлайн
    print("Не удалось импортировать модули. Используем встроенные реализации.")
    
    class MapGenerator:
        """
        Генератор карт для тестирования алгоритмов поиска пути.
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
        
        def visualize_map(self, grid, start_pos=None, finish_pos=None, path=None, title=None, block=False):
            """
            Визуализирует карту и, при наличии, путь.
            
            Параметры:
            - grid: матрица карты
            - start_pos: позиция старта (если None, берется из карты)
            - finish_pos: позиция финиша (если None, берется из карты)
            - path: список позиций, образующих путь (если None, путь не отображается)
            - title: заголовок графика
            - block: блокировать выполнение до закрытия окна
            """
            from matplotlib.colors import ListedColormap
            
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
            
            plt.tight_layout()
            plt.ion()  # Включаем интерактивный режим
            plt.draw()
            plt.pause(0.1)
            
            if block:
                plt.ioff()  # Отключаем интерактивный режим
                plt.show(block=block)
        
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
            from matplotlib.colors import ListedColormap
            
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
                colors = ['white', 'black', 'green', 'red', 'yellow', 'lightblue']
                cmap = ListedColormap(colors)
                img = ax.imshow(self.grid, cmap=cmap)
                plt.title('A* Algorithm')
                plt.ion()  # Включаем интерактивный режим
                plt.draw()
                plt.pause(0.1)
            
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
                        plt.pause(1.0)  # Небольшая пауза для отображения результата
                    
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
                plt.close()
            
            return [], time_elapsed, steps
        
        def _visualize_search(self, ax, img, final=False):
            """
            Обновляет визуализацию процесса поиска пути.
            
            Параметры:
            - ax: объект осей matplotlib
            - img: объект изображения matplotlib
            - final: если True, визуализирует финальный путь
            """
            from matplotlib.colors import ListedColormap
            
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
            
            # Обновляем заголовок
            ax.set_title(f'A* Algorithm (Steps: {len(self.explored_cells)})')
            
            plt.draw()
            plt.pause(0.01)  # Пауза для обновления отображения
        
        def visualize_result(self, show_explored=True):
            """
            Визуализирует результат поиска пути.
            
            Параметры:
            - show_explored: если True, показывает исследованные клетки
            """
            from matplotlib.colors import ListedColormap
            
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
            
            plt.ion()  # Включаем интерактивный режим
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)


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
            from matplotlib.colors import ListedColormap
            
            # Замеряем время выполнения
            start_time = time.time()
            
            # Настройка визуализации, если требуется
            if visualize:
                fig, ax = plt.subplots(figsize=(10, 10))
                # Цветовая карта: пусто, стена, старт, финиш, спора, исследованная клетка, спора достижимая, путь
                colors = ['white', 'black', 'green', 'red', 'yellow', 'lightblue', 'purple', 'orange']
                cmap = ListedColormap(colors)
                img = ax.imshow(self.grid, cmap=cmap)
                plt.title('Spore Algorithm')
                plt.ion()  # Включаем интерактивный режим
                plt.draw()
                plt.pause(0.1)
            
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
                plt.pause(1.0)  # Небольшая пауза для отображения результата
            
            return self.final_path, time_elapsed, self.iteration
        
        def _visualize_search(self, ax, img, final=False):
            """
            Обновляет визуализацию процесса поиска пути.
            
            Параметры:
            - ax: объект осей matplotlib
            - img: объект изображения matplotlib
            - final: если True, визуализирует финальный путь
            """
            from matplotlib.colors import ListedColormap
            
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
            from matplotlib.colors import ListedColormap
            
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
            
            plt.ion()  # Включаем интерактивный режим
            plt.draw()
            plt.pause(0.1)
            plt.show(block=False)


class AlgorithmComparison:
    """
    Класс для сравнения алгоритмов A* и Spore.
    """
    
    def __init__(self, 
                 num_maps=20, 
                 map_sizes=[(30, 30)], 
                 wall_densities=[0.3], 
                 spore_runs=10, 
                 a_star_runs=1,
                 results_dir="results"):
        """
        Инициализация сравнения алгоритмов.
        
        Параметры:
        - num_maps: количество карт для тестирования
        - map_sizes: список размеров карт для тестирования
        - wall_densities: список плотностей стен для тестирования
        - spore_runs: количество запусков Spore на каждой карте
        - a_star_runs: количество запусков A* на каждой карте
        - results_dir: директория для сохранения результатов
        """
        self.num_maps = num_maps
        self.map_sizes = map_sizes
        self.wall_densities = wall_densities
        self.spore_runs = spore_runs
        self.a_star_runs = a_star_runs
        self.results_dir = results_dir
        
        # Создаем директорию для результатов, если она не существует
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Создаем директорию для карт, если она не существует
        self.maps_dir = os.path.join(results_dir, "maps")
        if not os.path.exists(self.maps_dir):
            os.makedirs(self.maps_dir)
    
    def generate_maps(self):
        """
        Генерирует карты для тестирования.
        
        Возвращает:
        - список карт в формате (grid, start_pos, finish_pos, filename)
        """
        maps = []
        
        for size in self.map_sizes:
            for density in self.wall_densities:
                wall_count = int(size[0] * size[1] * density)
                
                for i in range(self.num_maps):
                    generator = MapGenerator(size=size, wall_count=wall_count)
                    grid, start_pos, finish_pos = generator.generate_map()
                    
                    # Создаем имя файла
                    filename = f"map_{size[0]}x{size[1]}_d{int(density*100)}_{i}.npy"
                    filepath = os.path.join(self.maps_dir, filename)
                    
                    # Сохраняем карту
                    generator.save_map(grid, filepath)
                    
                    maps.append((grid, start_pos, finish_pos, filename))
        
        return maps
    
    def run_a_star(self, grid, start_pos, finish_pos, visualize=False):
        """
        Запускает алгоритм A* на заданной карте.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        - visualize: если True, выполняет визуализацию
        
        Возвращает:
        - path: найденный путь
        - time_elapsed: затраченное время
        - steps: количество шагов
        - success: успешность поиска пути
        """
        agent = AStarAgent(grid, start_pos, finish_pos)
        path, time_elapsed, steps = agent.find_path(visualize=visualize)
        
        success = len(path) > 0
        
        if visualize and success:
            agent.visualize_result()
        
        return path, time_elapsed, steps, success
    
    def run_spore(self, grid, start_pos, finish_pos, max_iterations=200, visualize=False):
        """
        Запускает алгоритм Spore на заданной карте.
        
        Параметры:
        - grid: матрица карты
        - start_pos: позиция старта
        - finish_pos: позиция финиша
        - max_iterations: максимальное количество итераций
        - visualize: если True, выполняет визуализацию
        
        Возвращает:
        - path: найденный путь
        - time_elapsed: затраченное время
        - iterations: количество итераций
        - success: успешность поиска пути
        """
        agent = SporeAgent(grid, start_pos, finish_pos, max_iterations=max_iterations)
        path, time_elapsed, iterations = agent.find_path(visualize=visualize)
        
        success = len(path) > 0
        
        if visualize and success:
            agent.visualize_result()
        
        return path, time_elapsed, iterations, success
    
    def run_comparison(self, visualize=False, verbose=True):
        """
        Запускает сравнение алгоритмов A* и Spore.
        
        Параметры:
        - visualize: если True, выполняет визуализацию
        - verbose: если True, выводит информацию о прогрессе
        
        Возвращает:
        - results: DataFrame с результатами
        """
        # Генерируем карты
        if verbose:
            print("Генерация карт...")
        maps = self.generate_maps()
        
        # Подготавливаем результаты
        results = []
        
        # Текущее время для имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем CSV файл для записи результатов
        csv_filename = os.path.join(self.results_dir, f"comparison_results_{timestamp}.csv")
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['map_filename', 'algorithm', 'run', 'time', 'steps', 'path_length', 'success']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Запускаем алгоритмы на каждой карте
            for idx, (grid, start_pos, finish_pos, map_filename) in enumerate(maps):
                if verbose:
                    print(f"Тестирование карты {idx+1}/{len(maps)}: {map_filename}")
                
                # Запускаем A*
                for run in range(self.a_star_runs):
                    if verbose:
                        print(f"  Запуск A* {run+1}/{self.a_star_runs}...")
                    
                    vis = visualize and run == 0  # Визуализируем только первый запуск
                    path, time_elapsed, steps, success = self.run_a_star(grid, start_pos, finish_pos, visualize=vis)
                    
                    result = {
                        'map_filename': map_filename,
                        'algorithm': 'A*',
                        'run': run,
                        'time': time_elapsed,
                        'steps': steps,
                        'path_length': len(path) if success else 0,
                        'success': int(success)
                    }
                    
                    results.append(result)
                    writer.writerow(result)
                    csvfile.flush()  # Записываем результаты сразу, чтобы не потерять их при сбое
                    
                    if verbose:
                        print(f"    {'Успех' if success else 'Неудача'}, время: {time_elapsed:.4f}с, шагов: {steps}, длина пути: {len(path) if success else 0}")
                
                # Запускаем Spore
                for run in range(self.spore_runs):
                    if verbose:
                        print(f"  Запуск Spore {run+1}/{self.spore_runs}...")
                    
                    vis = visualize and run == 0  # Визуализируем только первый запуск
                    path, time_elapsed, iterations, success = self.run_spore(grid, start_pos, finish_pos, visualize=vis)
                    
                    result = {
                        'map_filename': map_filename,
                        'algorithm': 'Spore',
                        'run': run,
                        'time': time_elapsed,
                        'steps': iterations,
                        'path_length': len(path) if success else 0,
                        'success': int(success)
                    }
                    
                    results.append(result)
                    writer.writerow(result)
                    csvfile.flush()  # Записываем результаты сразу, чтобы не потерять их при сбое
                    
                    if verbose:
                        print(f"    {'Успех' if success else 'Неудача'}, время: {time_elapsed:.4f}с, итераций: {iterations}, длина пути: {len(path) if success else 0}")
        
        if verbose:
            print(f"Результаты сохранены в {csv_filename}")
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results=None, csv_filename=None):
        """
        Анализирует результаты сравнения алгоритмов.
        
        Параметры:
        - results: DataFrame с результатами (если None, загружает из файла)
        - csv_filename: имя файла для загрузки результатов (если results=None)
        
        Возвращает:
        - summary: DataFrame с сводкой результатов
        """
        if results is None and csv_filename is not None:
            # Загружаем результаты из файла
            results = pd.read_csv(csv_filename)
        elif results is None:
            # Находим последний файл с результатами
            files = [f for f in os.listdir(self.results_dir) if f.startswith('comparison_results_') and f.endswith('.csv')]
            if not files:
                raise FileNotFoundError("Не найдены файлы с результатами сравнения")
            
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(self.results_dir, f)))
            csv_filename = os.path.join(self.results_dir, latest_file)
            results = pd.read_csv(csv_filename)
        
        # Группировка по алгоритму и расчет средних значений
        summary = results.groupby('algorithm').agg({
            'time': ['mean', 'std', 'min', 'max'],
            'steps': ['mean', 'std', 'min', 'max'],
            'path_length': ['mean', 'std', 'min', 'max'],
            'success': ['mean', 'count']
        }).reset_index()
        
        # Переименовываем колонки для лучшей читаемости
        summary.columns = ['algorithm', 
                          'time_mean', 'time_std', 'time_min', 'time_max',
                          'steps_mean', 'steps_std', 'steps_min', 'steps_max',
                          'path_length_mean', 'path_length_std', 'path_length_min', 'path_length_max',
                          'success_rate', 'total_runs']
        
        return summary
    
    def plot_results(self, results=None, csv_filename=None, save_plots=True):
        """
        Строит графики по результатам сравнения алгоритмов.
        
        Параметры:
        - results: DataFrame с результатами (если None, загружает из файла)
        - csv_filename: имя файла для загрузки результатов (если results=None)
        - save_plots: если True, сохраняет графики в файлы
        """
        if results is None and csv_filename is not None:
            # Загружаем результаты из файла
            results = pd.read_csv(csv_filename)
        elif results is None:
            # Находим последний файл с результатами
            files = [f for f in os.listdir(self.results_dir) if f.startswith('comparison_results_') and f.endswith('.csv')]
            if not files:
                raise FileNotFoundError("Не найдены файлы с результатами сравнения")
            
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(self.results_dir, f)))
            csv_filename = os.path.join(self.results_dir, latest_file)
            results = pd.read_csv(csv_filename)
        
        # Текущее время для имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем фигуру для времени
        plt.figure(figsize=(12, 6))
        
        # Группируем по карте и алгоритму, вычисляем среднее время
        time_data = results.groupby(['map_filename', 'algorithm'])['time'].mean().reset_index()
        
        # Создаем сводку по картам для сортировки
        map_difficulty = time_data[time_data['algorithm'] == 'A*'].set_index('map_filename')['time']
        
        # Сортируем карты по сложности (времени A*)
        sorted_maps = map_difficulty.sort_values().index
        
        # Получаем данные для построения графика
        a_star_times = []
        spore_times = []
        
        for map_name in sorted_maps:
            a_star_time = time_data[(time_data['map_filename'] == map_name) & (time_data['algorithm'] == 'A*')]['time'].values[0]
            spore_time = time_data[(time_data['map_filename'] == map_name) & (time_data['algorithm'] == 'Spore')]['time'].values[0]
            
            a_star_times.append(a_star_time)
            spore_times.append(spore_time)
        
        # Строим график времени
        plt.figure(figsize=(14, 6))
        bar_width = 0.35
        index = np.arange(len(sorted_maps))
        
        plt.bar(index, a_star_times, bar_width, label='A*', alpha=0.8)
        plt.bar(index + bar_width, spore_times, bar_width, label='Spore', alpha=0.8)
        
        plt.xlabel('Карта')
        plt.ylabel('Время (секунды)')
        plt.title('Сравнение времени выполнения A* и Spore')
        plt.xticks(index + bar_width / 2, [os.path.basename(m) for m in sorted_maps], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, f"time_comparison_{timestamp}.png"))
        
        plt.figure(figsize=(10, 6))
        
        # Боксплот для времени
        plt.subplot(2, 2, 1)
        sns_data = results[['algorithm', 'time']].copy()
        plt.boxplot([sns_data[sns_data['algorithm'] == 'A*']['time'], 
                    sns_data[sns_data['algorithm'] == 'Spore']['time']], 
                   labels=['A*', 'Spore'])
        plt.title('Распределение времени')
        plt.ylabel('Время (секунды)')
        
        # Боксплот для шагов
        plt.subplot(2, 2, 2)
        plt.boxplot([results[results['algorithm'] == 'A*']['steps'], 
                    results[results['algorithm'] == 'Spore']['steps']], 
                   labels=['A*', 'Spore'])
        plt.title('Распределение шагов')
        plt.ylabel('Количество шагов')
        
        # Боксплот для длины пути
        plt.subplot(2, 2, 3)
        plt.boxplot([results[(results['algorithm'] == 'A*') & (results['success'] == 1)]['path_length'], 
                    results[(results['algorithm'] == 'Spore') & (results['success'] == 1)]['path_length']], 
                   labels=['A*', 'Spore'])
        plt.title('Распределение длины пути (только успешные)')
        plt.ylabel('Длина пути')
        
        # Гистограмма успешности
        plt.subplot(2, 2, 4)
        success_rate = results.groupby('algorithm')['success'].mean()
        plt.bar(success_rate.index, success_rate.values)
        plt.title('Показатель успешности')
        plt.ylabel('Доля успешных запусков')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.results_dir, f"metrics_comparison_{timestamp}.png"))
        
        plt.ion()
        plt.draw()
        plt.pause(0.1)
        plt.show(block=False)

    
def run_single_example():
    """
    Запускает пример сравнения алгоритмов на одной карте с визуализацией.
    """
    # Генерируем карту
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Визуализируем исходную карту
    generator.visualize_map(grid, start_pos, finish_pos, title="Исходная карта", block=False)
    
    # Запускаем A*
    print("\nЗапуск A*:")
    a_star_agent = AStarAgent(grid, start_pos, finish_pos)
    a_star_path, a_star_time, a_star_steps = a_star_agent.find_path(visualize=True)
    
    if a_star_path:
        print(f"A*: Путь найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
        print(f"A*: Длина пути: {len(a_star_path)}")
        a_star_agent.visualize_result()
    else:
        print(f"A*: Путь не найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
    
    # Запускаем Spore
    print("\nЗапуск Spore:")
    spore_agent = SporeAgent(grid, start_pos, finish_pos, max_iterations=100)
    spore_path, spore_time, spore_iterations = spore_agent.find_path(visualize=True)
    
    if spore_path:
        print(f"Spore: Путь найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
        print(f"Spore: Длина пути: {len(spore_path)}")
        spore_agent.visualize_result()
    else:
        print(f"Spore: Путь не найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
    
    # Удерживаем графики открытыми до нажатия клавиши
    print("\nНажмите любую клавишу, чтобы завершить программу...")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # По умолчанию запускаем полное сравнение
    run_full_comparison = True
    visualize = False
    
    # Парсим аргументы командной строки
    if len(sys.argv) > 1:
        if "--example" in sys.argv:
            run_full_comparison = False
        if "--visualize" in sys.argv:
            visualize = True
    
    if run_full_comparison:
        # Параметры сравнения
        num_maps = 20
        map_sizes = [(30, 30)]
        wall_densities = [0.2, 0.3, 0.4]
        spore_runs = 10
        a_star_runs = 1
        
        print(f"Запуск сравнения алгоритмов:")
        print(f"  - Карт: {num_maps}")
        print(f"  - Размеры карт: {map_sizes}")
        print(f"  - Плотности стен: {wall_densities}")
        print(f"  - Запусков Spore: {spore_runs}")
        print(f"  - Запусков A*: {a_star_runs}")
        print(f"  - Визуализация: {'Да' if visualize else 'Нет'}")
        
        # Создаем объект сравнения
        comparison = AlgorithmComparison(
            num_maps=num_maps,
            map_sizes=map_sizes,
            wall_densities=wall_densities,
            spore_runs=spore_runs,
            a_star_runs=a_star_runs
        )
        
        # Запускаем сравнение
        results = comparison.run_comparison(visualize=visualize)
        
        # Анализируем результаты
        summary = comparison.analyze_results(results)
        print("\nСводка результатов:")
        print(summary)
        
        # Строим графики
        comparison.plot_results(results)
        
        # Удерживаем графики открытыми до нажатия клавиши
        input("Нажмите Enter, чтобы завершить программу...")
    else:
        # Запускаем пример на одной карте
        run_single_example()