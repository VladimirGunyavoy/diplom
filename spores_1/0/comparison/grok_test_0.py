import os
import pickle
import time
import csv
from collections import OrderedDict

# Импортируем классы агентов из отдельных файлов
from test_spore_1 import ImprovedSporeAgent, load_map_data as load_spore_map
from test_a_star import AStarPathfinder, load_map_data as load_astar_map

def test_agents_on_mazes():
    # Путь к папке с лабиринтами
    base_folder = "maze_collection"
    
    # Файл для сохранения результатов
    results_file = "agent_test_results.csv"
    
    # Заголовки для CSV файла
    headers = [
        "Maze Size", "Wall Percent", 
        "Spore Path Length", "Spore Time (s)", "Spore Iterations", "Spore Explored Cells",
        "A* Path Length", "A* Time (s)", "A* Explored Cells"
    ]
    
    # Список для хранения результатов
    results = []
    
    # Проходим по всем папкам с размерами
    for size_folder in os.listdir(base_folder):
        size_path = os.path.join(base_folder, size_folder)
        if not os.path.isdir(size_path):
            continue
            
        print(f"Обработка размера: {size_folder}")
        
        # Проходим по всем файлам лабиринтов в папке
        for maze_file in os.listdir(size_path):
            if not maze_file.endswith(".pkl"):
                continue
                
            maze_path = os.path.join(size_path, maze_file)
            
            # Загружаем карту (используем любую из функций загрузки, они должны быть идентичны)
            grid, start_pos, finish_pos = load_spore_map(maze_path)
            
            # Извлекаем размер и процент заполнения из имени файла
            parts = maze_file.split('_')
            maze_size = parts[1]  # e.g., "100x100"
            wall_percent = parts[2].replace("percent.pkl", "")  # e.g., "10"
            
            print(f"  Тестирование лабиринта: {maze_size}, {wall_percent}% стен")
            
            # Тестируем ImprovedSporeAgent
            spore_agent = ImprovedSporeAgent(grid, start_pos, finish_pos, max_iterations=200, debug=False)
            spore_path, spore_time, spore_iterations = spore_agent.find_path()
            spore_explored = len(spore_agent.explored_cells)
            spore_path_len = len(spore_path) if spore_path else 0
            
            # Тестируем AStarPathfinder
            astar_agent = AStarPathfinder(grid, start_pos, finish_pos)
            astar_path, astar_time = astar_agent.find_path()
            astar_explored = len(astar_agent.explored_cells)
            astar_path_len = len(astar_path) if astar_path else 0
            
            # Собираем результаты
            result = OrderedDict([
                ("Maze Size", maze_size),
                ("Wall Percent", wall_percent),
                ("Spore Path Length", spore_path_len),
                ("Spore Time (s)", f"{spore_time:.6f}"),
                ("Spore Iterations", spore_iterations),
                ("Spore Explored Cells", spore_explored),
                ("A* Path Length", astar_path_len),
                ("A* Time (s)", f"{astar_time:.6f}"),
                ("A* Explored Cells", astar_explored)
            ])
            results.append(result)
            
            print(f"    Spore: Путь={spore_path_len}, Время={spore_time:.6f}с, Итераций={spore_iterations}, Исследовано={spore_explored}")
            print(f"    A*: Путь={astar_path_len}, Время={astar_time:.6f}с, Исследовано={astar_explored}")
    
    # Сохраняем результаты в CSV файл
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nРезультаты сохранены в файл: {results_file}")
    print(f"Всего протестировано лабиринтов: {len(results)}")

def main():
    print("Запуск тестирования агентов на всех лабиринтах...")
    test_agents_on_mazes()
    print("Тестирование завершено!")

if __name__ == "__main__":
    main()