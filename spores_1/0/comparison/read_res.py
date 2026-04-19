import csv
from prettytable import PrettyTable

def display_test_results(file_path="agent_test_results.csv"):
    # Создаем таблицу с помощью PrettyTable
    table = PrettyTable()
    
    # Устанавливаем заголовки столбцов
    headers = [
        "Maze Size", "Wall %", 
        "Spore Path", "Spore Time (s)", "Spore Iters", "Spore Explored",
        "A* Path", "A* Time (s)", "A* Explored"
    ]
    table.field_names = headers
    
    # Настраиваем выравнивание столбцов
    table.align["Maze Size"] = "l"  # Выравнивание влево
    table.align["Wall %"] = "r"     # Выравнивание вправо
    table.align["Spore Path"] = "r"
    table.align["Spore Time (s)"] = "r"
    table.align["Spore Iters"] = "r"
    table.align["Spore Explored"] = "r"
    table.align["A* Path"] = "r"
    table.align["A* Time (s)"] = "r"
    table.align["A* Explored"] = "r"
    
    # Читаем данные из CSV файла
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Добавляем строки в таблицу
            for row in reader:
                table.add_row([
                    row["Maze Size"],
                    row["Wall Percent"],
                    row["Spore Path Length"],
                    row["Spore Time (s)"],
                    row["Spore Iterations"],
                    row["Spore Explored Cells"],
                    row["A* Path Length"],
                    row["A* Time (s)"],
                    row["A* Explored Cells"]
                ])
            
            # Печатаем таблицу
            print("\nРезультаты тестирования агентов:")
            print(table)
            print(f"\nВсего протестировано лабиринтов: {reader.line_num - 1}")
            
    except FileNotFoundError:
        print(f"Ошибка: Файл '{file_path}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    print("Отображение результатов тестов из файла 'agent_test_results.csv'...")
    display_test_results()
    print("Вывод завершен!")

if __name__ == "__main__":
    main()