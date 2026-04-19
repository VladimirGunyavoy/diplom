import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from datetime import datetime
from comparison.map_generator_0 import MapGenerator
from a_star_agent import AStarAgent
from spore_agent import SporeAgent
import json

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
        
        plt.show()

    
def run_single_example():
    """
    Запускает пример сравнения алгоритмов на одной карте с визуализацией.
    """
    # Генерируем карту
    generator = MapGenerator(size=(30, 30), wall_count=200)
    grid, start_pos, finish_pos = generator.generate_map()
    
    # Визуализируем исходную карту
    generator.visualize_map(grid, start_pos, finish_pos, title="Исходная карта")
    
    # Запускаем A*
    print("\nЗапуск A*:")
    a_star_agent = AStarAgent(grid, start_pos, finish_pos)
    a_star_path, a_star_time, a_star_steps = a_star_agent.find_path(visualize=True)
    
    if a_star_path:
        print(f"A*: Путь найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
        print(f"A*: Длина пути: {len(a_star_path)}")
    else:
        print(f"A*: Путь не найден за {a_star_time:.4f} секунд, {a_star_steps} шагов")
    
    # Запускаем Spore
    print("\nЗапуск Spore:")
    spore_agent = SporeAgent(grid, start_pos, finish_pos, max_iterations=100)
    spore_path, spore_time, spore_iterations = spore_agent.find_path(visualize=True)
    
    if spore_path:
        print(f"Spore: Путь найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
        print(f"Spore: Длина пути: {len(spore_path)}")
    else:
        print(f"Spore: Путь не найден за {spore_time:.4f} секунд, {spore_iterations} итераций")
    
    # Визуализируем результаты
    a_star_agent.visualize_result()
    spore_agent.visualize_result()


if __name__ == "__main__":
    import sys
    
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
    else:
        # Запускаем пример на одной карте
        run_single_example()