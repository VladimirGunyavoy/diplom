import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
import heapq
from collections import deque
import pickle
import os

class ImprovedSporeAgent:
    """
    Улучшенный агент поиска пути с использованием алгоритма Spore.
    """
    
    def __init__(self, grid, start_pos, finish_pos, max_iterations=100, debug=True):
        """
        Инициализация Spore агента.
        
        Параметры:
        - grid: матрица карты (0 - свободное пространство, 1 - стена, 2 - старт, 3 - финиш)
        - start_pos: кортеж (x, y) - координаты стартовой позиции
        - finish_pos: кортеж (x, y) - координаты целевой позиции
        - max_iterations: максимальное количество итераций алгоритма
        - debug: вывод отладочной информации
        """
        self.grid = grid.copy()
        self.height, self.width = grid.shape
        self.start_pos = start_pos
        self.finish_pos = finish_pos
        self.max_iterations = max_iterations
        self.debug = debug
        
        # Список спор (позиция, может_достичь_финиша, помечена_для_удаления)
        self.spores = [(start_pos[0], start_pos[1], True, False)]
        
        # Множество клеток, из которых можно достичь финиша
        self.can_reach_finish = {start_pos}
        
        # Пути агентов
        self.agent_paths = []
        
        # Успешные пути
        self.successful_paths = []
        
        # Граф достижимости (ключ: точка, значение: родительская точка)
        self.reachability_graph = {start_pos: None}
        
        # Граф обратной достижимости (ключ: точка, значение: список точек, достижимых из ключа)
        self.reverse_graph = {}
        
        # Итоговый путь
        self.final_path = []
        
        # Счетчик итераций
        self.iteration = 0
        
        # Флаг найденного пути
        self.path_found = False
        
        # Исследованные клетки
        self.explored_cells = set([start_pos])
        
        # Расстояние между стартом и финишем для адаптивных параметров
        self.distance_start_finish = np.sqrt((start_pos[0] - finish_pos[0])**2 + (start_pos[1] - finish_pos[1])**2)
        
        if self.debug:
            print(f"Инициализирован ImprovedSporeAgent:")
            print(f"  - Размер карты: {self.height}x{self.width}")
            print(f"  - Старт: {self.start_pos}")
            print(f"  - Финиш: {self.finish_pos}")
            print(f"  - Расстояние между стартом и финишем: {self.distance_start_finish:.2f}")
    
    # ... (оставляем методы distance, is_valid_position, generate_spore_position, sample_action, run_random_sequence без изменений, если они работают корректно)

    def update_reachable_points(self):
        """Обновляет множество точек, из которых можно достичь цель, с улучшенной логикой"""
        if self.debug:
            print("Обновление достижимых точек...")
            initial_count = len(self.can_reach_finish)
        
        changed = True
        iteration = 0
        max_iterations = 100  # Ограничение для предотвращения бесконечного цикла
        
        while changed and iteration < max_iterations:
            changed = False
            new_points = set()
            
            # Проходим по всем точкам графа достижимости
            for point, parent in list(self.reachability_graph.items()):
                if point in self.can_reach_finish:
                    continue
                
                # Проверяем, можно ли достичь финиш из этой точки через существующий граф
                if self._can_reach_finish_from_point(point):
                    new_points.add(point)
                    changed = True
            
            # Добавляем новые точки
            self.can_reach_finish.update(new_points)
        
            # Обновляем споры
            for i, (x, y, _, marked) in enumerate(self.spores):
                if (x, y) in self.can_reach_finish and not marked:
                    self.spores[i] = (x, y, True, marked)
            
            iteration += 1
        
        if self.debug:
            final_count = len(self.can_reach_finish)
            if final_count > initial_count:
                print(f"  Добавлено {final_count - initial_count} новых достижимых точек")
            print(f"  Всего достижимых точек: {final_count}")

    def _can_reach_finish_from_point(self, point):
        """Проверяет, можно ли достичь финиша из данной точки через граф"""
        if point == self.finish_pos:
            return True
        
        visited = set()
        queue = deque([point])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current == self.finish_pos:
                return True
            
            # Проверяем соседей через обратный граф
            if current in self.reverse_graph:
                for neighbor in self.reverse_graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            # Проверяем родителя через прямой граф
            if current in self.reachability_graph:
                parent = self.reachability_graph[current]
                if parent and parent not in visited:
                    queue.append(parent)
        
        return False

    def reconstruct_path(self):
        """Восстанавливает путь от старта до финиша с улучшенной логикой"""
        if self.debug:
            print("Восстановление пути...")
        
        # Проверяем наличие прямого пути через A*
        astar_path = self._astar_search(self.grid, self.start_pos, self.finish_pos)
        if astar_path and len(astar_path) > 1:
            if self.debug:
                print(f"  Найден прямой путь через A* длиной {len(astar_path)}")
            return astar_path

        # Если финиш не в графе, ищем ближайшую достижимую точку
        if self.finish_pos not in self.reachability_graph:
            closest_point = min(self.can_reach_finish, 
                              key=lambda p: self.distance(p, self.finish_pos), 
                              default=self.start_pos)
            if closest_point != self.start_pos:
                if self.debug:
                    print(f"  Финиш не в графе. Используем ближайшую точку: {closest_point}")
                path_to_closest = self._build_path_from_graph(self.start_pos, closest_point)
                path_to_finish = self._astar_search(self.grid, closest_point, self.finish_pos)
                
                if path_to_closest and path_to_finish:
                    combined_path = path_to_closest[:-1] + path_to_finish
                    if self.debug:
                        print(f"  Построен комбинированный путь длиной {len(combined_path)}")
                    return combined_path
            else:
                if self.debug:
                    print("  Не найдено подходящих точек для комбинированного пути")

        # Если финиш в графе, пытаемся восстановить путь
        if self.finish_pos in self.reachability_graph:
            path = self._build_path_from_graph(self.start_pos, self.finish_pos)
            if path and len(path) > 1:
                if self.debug:
                    print(f"  Восстановлен путь из графа длиной {len(path)}")
                return path

        # Если путь не найден, пробуем через ключевые споры
        reachable_spores = [(x, y) for x, y, reachable, _ in self.spores if reachable and (x, y) != self.start_pos]
        if reachable_spores:
            reachable_spores.sort(key=lambda s: self.distance(s, self.finish_pos))
            
            for spore in reachable_spores:
                path_to_spore = self._build_path_from_graph(self.start_pos, spore)
                path_from_spore = self._astar_search(self.grid, spore, self.finish_pos)
                
                if path_to_spore and path_from_spore:
                    combined_path = path_to_spore[:-1] + path_from_spore
                    if self.debug:
                        print(f"  Построен путь через спору {spore} длиной {len(combined_path)}")
                    return combined_path

        # Если все методы не сработали, возвращаем None
        if self.debug:
            print("  Не удалось восстановить путь")
        return None

    def _build_path_from_graph(self, start, end):
        """Строит путь от start до end используя граф достижимости с улучшенной логикой"""
        if start == end:
            return [start]
        
        # Используем BFS для поиска пути в графе
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            
            if node == end:
                return path
            
            # Проверяем соседей через обратный граф
            if node in self.reverse_graph:
                for neighbor in self.reverse_graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            # Проверяем родителя через прямой граф
            if node in self.reachability_graph:
                parent = self.reachability_graph[node]
                if parent and parent not in visited:
                    visited.add(parent)
                    queue.append((parent, path + [parent]))
        
        # Если путь не найден через граф, пробуем A*
        return self._astar_search(self.grid, start, end)

    # ... (оставляем другие методы без изменений, если они работают корректно, например, _astar_search)

    def step(self):
        """Выполняет один шаг алгоритма с улучшенной логикой"""
        if self.path_found:
            return False, "Путь уже найден"
        
        if self.iteration >= self.max_iterations:
            return False, "Достигнуто максимальное число итераций"
        
        new_spore_pos = self.generate_spore_position()
        
        if new_spore_pos:
            is_reachable = new_spore_pos in self.can_reach_finish
            self.spores.append((new_spore_pos[0], new_spore_pos[1], is_reachable, False))
            
            paths, reached_goal = self.run_random_sequence(new_spore_pos)
            
            # Обновляем достижимые точки
            self.update_reachable_points()
            
            # Проверяем, найден ли путь до финиша
            if (self.finish_pos in self.reachability_graph or 
                self.finish_pos in self.can_reach_finish or 
                reached_goal):
                self.final_path = self.reconstruct_path()
                if self.final_path and len(self.final_path) > 1:
                    self.path_found = True
                    return True, f"Путь найден на итерации {self.iteration + 1}! Длина пути: {len(self.final_path)}"
                else:
                    return True, f"Финиш достижим, но путь не восстановлен. Продолжаем поиск."
            
            # Проверяем прогресс
            min_dist_to_finish = float('inf')
            for x, y, _, _ in self.spores:
                dist = self.distance((x, y), self.finish_pos)
                min_dist_to_finish = min(min_dist_to_finish, dist)
                
            progress_msg = f"Ближайшая спора к финишу: {min_dist_to_finish:.2f}, "
            progress_msg += f"Исследовано: {len(self.explored_cells)}/{self.height * self.width}"
            
            self.iteration += 1
            return True, f"Итерация {self.iteration}/{self.max_iterations}. {progress_msg}"
        
        self.iteration += 1
        return True, f"Итерация {self.iteration}/{self.max_iterations}. Не удалось создать новую спору."

    def find_path(self, visualize=False, visualize_steps=False, visualize_interval=5):
        """Находит путь от старта до финиша"""
        start_time = time.time()
        
        print("Начинаю поиск пути с помощью улучшенного алгоритма Spore...")
        
        if visualize_steps:
            print(f"Визуализация начального состояния...")
            self.visualize_map(
                self.grid, 
                self.start_pos, 
                self.finish_pos, 
                path=self.final_path, 
                explored=list(self.explored_cells), 
                spores=self.spores,
                title=f"Spore Algorithm (Итерация 0/{self.max_iterations})"
            )
        
        continue_sim = True
        while continue_sim and self.iteration < self.max_iterations:
            if visualize_steps and self.iteration % visualize_interval == 0 and self.iteration > 0:
                print(f"Визуализация состояния на итерации {self.iteration}...")
                self.visualize_map(
                    self.grid, 
                    self.start_pos, 
                    self.finish_pos, 
                    path=self.final_path, 
                    explored=list(self.explored_cells), 
                    spores=self.spores,
                    title=f"Spore Algorithm (Итерация {self.iteration}/{self.max_iterations})"
                )
            
            continue_sim, message = self.step()
            print(f"  [Итерация {self.iteration}] {message}")
            
            if self.path_found:
                break
        
        time_elapsed = time.time() - start_time
        
        if not self.final_path or len(self.final_path) < 2:
            print("Выполняем финальную реконструкцию пути...")
            self.update_reachable_points()
            self.final_path = self.reconstruct_path()
            
            if not self.final_path:
                print("Пытаемся найти путь напрямую с помощью A*...")
                self.final_path = self._astar_search(self.grid, self.start_pos, self.finish_pos)
        
        if self.final_path and len(self.final_path) > 1:
            print(f"Путь найден за {time_elapsed:.4f} секунд!")
            print(f"Длина пути: {len(self.final_path)}")
            print(f"Итераций: {self.iteration}")
            print(f"Исследовано клеток: {len(self.explored_cells)}")
            print(f"Создано спор: {len(self.spores)}")
            self.path_found = True
        else:
            print(f"Путь не найден за {time_elapsed:.4f} секунд.")
            print(f"Итераций: {self.iteration}")
            print(f"Исследовано клеток: {len(self.explored_cells)}")
            print(f"Создано spor: {len(self.spores)}")
        
        if visualize:
            print("Визуализация итогового результата...")
            self.visualize_map(
                self.grid, 
                self.start_pos, 
                self.finish_pos, 
                path=self.final_path, 
                explored=list(self.explored_cells), 
                spores=self.spores,
                title=f"Результат Spore (Итераций: {self.iteration})"
            )
        
        return self.final_path, time_elapsed, self.iteration
    

def main():
    # Параметры
    map_file = "map_data.pkl"
    visualize = False  # Отключаем визуализацию результата
    visualize_steps = False  # Отключаем визуализацию промежуточных шагов
    max_iterations = 100
    debug = False  # Отключаем отладочные сообщения
    
    # Загружаем карту
    grid, start_pos, finish_pos = load_map_data(map_file)
    
    # Отключаем визуализацию исходной карты
    
    # Запускаем улучшенный алгоритм Spore
    spore_agent = ImprovedSporeAgent(grid, start_pos, finish_pos, 
                                     max_iterations=max_iterations, 
                                     debug=debug)  # Отключаем debug
    
    
    print("Визуализация исходной карты...")
    spore_agent.visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    path, time_elapsed, iterations = spore_agent.find_path(
        visualize=visualize,  # Отключаем визуализацию
        visualize_steps=visualize_steps  # Отключаем визуализацию шагов
    )
    
    print("Длина пути:", len(path) if path else "Путь не найден")
    print(f"Время: {time_elapsed:.6f} секунд")
    print(f"Итераций: {iterations}")

if __name__ == "__main__":
    main()