import pygame
import numpy as np
import random
import math
from collections import deque

# Инициализация pygame
pygame.init()

# Константы
WIDTH, HEIGHT = 1800, 1600
GRID_SIZE = 40
COLS, ROWS = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (150, 150, 150)  # Серый цвет для спор
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (128, 0, 128)

# Создание экрана
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spore Pathfinding Algorithm")

# Представление сетки
# 0: пусто, 1: стена, 2: старт, 3: финиш, 4: спора
grid = np.zeros((ROWS, COLS), dtype=int)

# Инициализация позиций старта и финиша
start_pos = (ROWS // 4, COLS // 4)
finish_pos = (3 * ROWS // 4, 3 * COLS // 4)
grid[start_pos[0]][start_pos[1]] = 2
grid[finish_pos[0]][finish_pos[1]] = 3

# Список спор (строка, столбец, может_достичь_финиша, помечена_для_удаления)
spores = []
# Пути агентов [[(row, col), (row, col), ...], [...], ...]
agent_paths = []
# Пути неуспешных агентов, которые будут удалены при следующем нажатии n
failed_paths = []
# Успешные пути, ведущие к финишу
successful_paths = []
# Клетки, из которых можно достичь финиша
can_reach_finish = set()
# Итоговый путь от старта до финиша
final_path = []
# Текущий режим
mode = "generate_spore"  # "generate_spore", "run_agent"
# Состояние агента
current_spore_index = -1
current_sequence = 0
current_step = 0
current_path = []
# Автоматический режим
auto_mode = False

def draw_grid():
    for row in range(ROWS):
        for col in range(COLS):
            x, y = col * GRID_SIZE, row * GRID_SIZE
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            
            if grid[row][col] == 0:  # Пусто
                pygame.draw.rect(screen, WHITE, rect)
            elif grid[row][col] == 1:  # Стена
                pygame.draw.rect(screen, BLACK, rect)
            elif grid[row][col] == 2:  # Старт
                pygame.draw.rect(screen, GREEN, rect)
            elif grid[row][col] == 3:  # Финиш
                pygame.draw.rect(screen, RED, rect)
            elif grid[row][col] == 4:  # Спора
                pygame.draw.rect(screen, GRAY, rect)
            
            pygame.draw.rect(screen, BLACK, rect, 1)  # Линии сетки
    
    # Рисуем все пути агентов (и успешные, и неуспешные)
    for path in agent_paths:
        for i in range(len(path) - 1):
            start = (path[i][1] * GRID_SIZE + GRID_SIZE // 2, 
                    path[i][0] * GRID_SIZE + GRID_SIZE // 2)
            end = (path[i+1][1] * GRID_SIZE + GRID_SIZE // 2, 
                   path[i+1][0] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.line(screen, LIGHT_BLUE, start, end, 2)
    
    # Рисуем пути неуспешных агентов, которые будут удалены
    for path in failed_paths:
        for i in range(len(path) - 1):
            start = (path[i][1] * GRID_SIZE + GRID_SIZE // 2, 
                    path[i][0] * GRID_SIZE + GRID_SIZE // 2)
            end = (path[i+1][1] * GRID_SIZE + GRID_SIZE // 2, 
                   path[i+1][0] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.line(screen, (255, 200, 200), start, end, 2)  # Светло-красный для неуспешных путей
    
    # Рисуем успешные пути отдельным цветом
    for path in successful_paths:
        for i in range(len(path) - 1):
            start = (path[i][1] * GRID_SIZE + GRID_SIZE // 2, 
                    path[i][0] * GRID_SIZE + GRID_SIZE // 2)
            end = (path[i+1][1] * GRID_SIZE + GRID_SIZE // 2, 
                   path[i+1][0] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.line(screen, BLUE, start, end, 2)

    # Рисуем кружки для спор, которые могут достичь финиша
    for spore in spores:
        if spore[2]:  # Если can_reach_finish равно True
            x, y = spore[1] * GRID_SIZE + GRID_SIZE // 2, spore[0] * GRID_SIZE + GRID_SIZE // 2
            pygame.draw.circle(screen, PURPLE, (x, y), GRID_SIZE // 4)
    
    # Рисуем итоговый путь, если он существует
    if final_path:
        for i in range(len(final_path) - 1):
            start = (final_path[i][1] * GRID_SIZE + GRID_SIZE // 2, 
                    final_path[i][0] * GRID_SIZE + GRID_SIZE // 2)
            end = (final_path[i+1][1] * GRID_SIZE + GRID_SIZE // 2, 
                   final_path[i+1][0] * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.line(screen, BLUE, start, end, 3)

def is_valid_position(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS and grid[row][col] != 1  # Не стена

def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def is_valid_spore_position(row, col):
    # Проверяем, является ли позиция допустимой и не слишком близкой к другим спорам
    if not is_valid_position(row, col) or grid[row][col] in [2, 3, 4]:
        return False
    
    # Проверяем расстояние от других спор
    for spore in spores:
        if distance((row, col), (spore[0], spore[1])) < 5:
            return False
    
    return True

def generate_spore_near_finish():
    # Находим допустимую позицию рядом с финишем
    attempts = 0
    max_attempts = 100
    max_distance = 10
    
    while attempts < max_attempts:
        # Случайная позиция около финиша
        offset_row = int(np.random.normal(0, max_distance / 3))
        offset_col = int(np.random.normal(0, max_distance / 3))
        
        row = finish_pos[0] + offset_row
        col = finish_pos[1] + offset_col
        
        if is_valid_spore_position(row, col):
            grid[row][col] = 4
            spores.append([row, col, False, False])  # строка, столбец, может_достичь_финиша, помечена_для_удаления
            return True
        
        attempts += 1
    
    return False

def generate_spore_near_existing():
    # Находим допустимую позицию рядом с существующей спорой
    if not spores:
        return generate_spore_near_finish()
    
    # С вероятностью 10% создаем спору рядом со стартом
    if random.random() < 0.1:
        # Пытаемся разместить спору рядом со стартом
        start_row, start_col = start_pos
        attempts = 0
        while attempts < 50:
            offset_row = random.randint(-3, 3)
            offset_col = random.randint(-3, 3)
            row = start_row + offset_row
            col = start_col + offset_col
            if is_valid_spore_position(row, col):
                grid[row][col] = 4
                spores.append([row, col, False, False])  # row, col, can_reach_finish, marked_for_death
                return True
            attempts += 1
    
    # Проверяем, есть ли путь агента рядом со стартом
    start_row, start_col = start_pos
    
    has_path_near_start = False
    # Проверяем, есть ли точка любого пути агента рядом со стартом
    for path in agent_paths:
        for row, col in path:
            if abs(row - start_row) <= 1 and abs(col - start_col) <= 1:
                has_path_near_start = True
                break
        if has_path_near_start:
            break
    
    if has_path_near_start:
        # Пытаемся разместить спору рядом со стартом
        attempts = 0
        while attempts < 50:
            offset_row = random.randint(-3, 3)
            offset_col = random.randint(-3, 3)
            row = start_row + offset_row
            col = start_col + offset_col
            if is_valid_spore_position(row, col):
                grid[row][col] = 4
                spores.append([row, col, False, False])
                return True
            attempts += 1
    
    # Размещаем рядом с существующими достижимыми спорами, если возможно
    reachable_spores = [s for s in spores if s[2] and not s[3]]
    if reachable_spores:
        attempts = 0
        while attempts < 50:
            spore = random.choice(reachable_spores)
            offset_row = int(np.random.normal(0, 5))
            offset_col = int(np.random.normal(0, 5))
            row = spore[0] + offset_row
            col = spore[1] + offset_col
            if is_valid_spore_position(row, col):
                grid[row][col] = 4
                spores.append([row, col, False, False])
                return True
            attempts += 1
    
    # В противном случае размещаем рядом с любой существующей спорой, не помеченной на удаление
    active_spores = [s for s in spores if not s[3]]
    if active_spores:
        attempts = 0
        while attempts < 50:
            spore = random.choice(active_spores)
            offset_row = int(np.random.normal(0, 5))
            offset_col = int(np.random.normal(0, 5))
            row = spore[0] + offset_row
            col = spore[1] + offset_col
            if is_valid_spore_position(row, col):
                grid[row][col] = 4
                spores.append([row, col, False, False])
                return True
            attempts += 1
    
    return False

def find_nearest_reachable_point(row, col):
    # Находим ближайшую точку, из которой можно достичь финиша
    nearest = None
    min_dist = float('inf')
    
    # Проверяем, ближе ли финиш, чем любая достижимая спора
    dist_to_finish = distance((row, col), finish_pos)
    if dist_to_finish < min_dist:
        min_dist = dist_to_finish
        nearest = finish_pos
    
    # Проверяем расстояние до достижимых спор
    for spore in spores:
        if spore[2]:  # Если может достичь финиша
            dist = distance((row, col), (spore[0], spore[1]))
            if dist < min_dist:
                min_dist = dist
                nearest = (spore[0], spore[1])
    
    # Если нет достижимой точки, используем финиш
    if nearest is None:
        return finish_pos
    
    return nearest

def run_agent():
    global current_spore_index, mode, agent_paths, failed_paths
    
    if current_spore_index < 0 or current_spore_index >= len(spores):
        mode = "generate_spore"
        return
    
    spore = spores[current_spore_index]
    reached_finish_or_reachable_spore = False
    current_spore_paths = []  # Пути для текущей споры
    
    # Для каждой споры пробуем 5 разных последовательностей
    for sequence in range(5):
        # Начинаем путь от споры
        current_path = [(spore[0], spore[1])]
        
        # Делаем 10 шагов в каждой последовательности
        for step in range(40):
            # Получаем текущую позицию
            current_row, current_col = current_path[-1]
            
            # Находим цель (финиш или ближайшую достижимую точку)
            target = find_nearest_reachable_point(current_row, current_col)
            
            # Вектор направления
            dir_row = target[0] - current_row
            dir_col = target[1] - current_col
            
            # Нормализация
            length = max(1, math.sqrt(dir_row**2 + dir_col**2))
            dir_row /= length
            dir_col /= length
            
            # Добавляем шум (нормальное распределение)
            dir_row += np.random.normal(0, 0.5)
            dir_col += np.random.normal(0, 0.5)
            
            # Выбираем направление (вверх, вниз, влево, вправо)
            if abs(dir_row) > abs(dir_col):
                if dir_row > 0:
                    next_row, next_col = current_row + 1, current_col  # Вниз
                else:
                    next_row, next_col = current_row - 1, current_col  # Вверх
            else:
                if dir_col > 0:
                    next_row, next_col = current_row, current_col + 1  # Вправо
                else:
                    next_row, next_col = current_row, current_col - 1  # Влево
            
            # Проверяем, допустима ли позиция
            if is_valid_position(next_row, next_col):
                current_path.append((next_row, next_col))
                
                # Проверяем, достигнут ли финиш
                if (next_row, next_col) == finish_pos:
                    # Отмечаем спору как достижимую
                    spores[current_spore_index][2] = True
                    can_reach_finish.add((spore[0], spore[1]))
                    successful_paths.append(current_path.copy())
                    reached_finish_or_reachable_spore = True
                    break
                
                # Проверяем, не дошли ли до другой споры, из которой можно достичь финиш
                for other_spore in spores:
                    if other_spore[2] and not other_spore[3] and (next_row, next_col) == (other_spore[0], other_spore[1]):
                        # Отмечаем текущую спору как достижимую через другую спору
                        spores[current_spore_index][2] = True
                        can_reach_finish.add((spore[0], spore[1]))
                        successful_paths.append(current_path.copy())
                        reached_finish_or_reachable_spore = True
                        break
                
                if reached_finish_or_reachable_spore:
                    break
            else:
                # Наткнулись на стену, завершаем эту последовательность
                break
        
        # Сохраняем путь текущего агента
        if len(current_path) > 1:  # Только если сделали хотя бы один шаг
            current_spore_paths.append(current_path.copy())
        
        # Если уже достигли финиша или достижимой споры, прекращаем попытки
        if reached_finish_or_reachable_spore:
            break
    
    # Добавляем пути этой споры в общий список, если спора успешна
    if reached_finish_or_reachable_spore:
        agent_paths.extend(current_spore_paths)
    else:
        # Если спора неуспешна, добавляем пути в список неуспешных путей и помечаем спору для удаления
        failed_paths.extend(current_spore_paths)
        spores[current_spore_index][3] = True  # Помечаем спору для удаления
    
    # Переходим к следующей споре
    current_spore_index += 1
    if current_spore_index >= len(spores):
        mode = "generate_spore"

def update_reachable_spores():
    """Обновляет статус 'может_достичь_финиша' для всех спор на основе связей между ними"""
    # Изначально помечаем как достижимые только те споры, которые напрямую достигли финиша
    reachable_spores = [i for i, spore in enumerate(spores) if spore[2] and not spore[3]]
    newly_reachable = True
    
    # Продолжаем, пока находим новые достижимые споры
    while newly_reachable:
        newly_reachable = False
        # Создаем копию списка для итерации
        current_reachable = reachable_spores.copy()
        
        for i, spore in enumerate(spores):
            if i not in reachable_spores and not spore[3]:  # Проверяем только недостижимые и не помеченные для удаления споры
                for path in agent_paths:
                    # Если путь начинается от этой споры
                    if path[0] == (spore[0], spore[1]):
                        # Проверяем, не ведет ли путь к достижимой споре
                        for j in current_reachable:
                            reachable_spore = spores[j]
                            if any(pos == (reachable_spore[0], reachable_spore[1]) for pos in path):
                                # Нашли путь к достижимой споре, помечаем текущую как достижимую
                                spores[i][2] = True
                                can_reach_finish.add((spore[0], spore[1]))
                                reachable_spores.append(i)
                                newly_reachable = True
                                break
                        if i in reachable_spores:
                            break

def check_for_path():
    # Обновляем статус всех спор
    update_reachable_spores()
    
    # Проверяем, есть ли достижимые споры
    reachable_spores = [s for s in spores if s[2] and not s[3]]
    if not reachable_spores:
        return False
    
    # Проверяем, есть ли пути агентов рядом со стартом
    start_row, start_col = start_pos
    
    has_path_near_start = False
    # Проверяем, есть ли точка любого пути агента рядом со стартом
    for path in agent_paths:
        for row, col in path:
            if abs(row - start_row) <= 1 and abs(col - start_col) <= 1:
                has_path_near_start = True
                break
        if has_path_near_start:
            break
    
    if not has_path_near_start:
        return False
    
    # Строим граф из всех путей агентов
    graph = {}
    for path in agent_paths + successful_paths:
        for i in range(len(path) - 1):
            if path[i] not in graph:
                graph[path[i]] = []
            if path[i+1] not in graph[path[i]]:
                graph[path[i]].append(path[i+1])
    
    # Добавляем связь между стартом и его соседями
    start_neighbors = []
    for path in agent_paths:
        if abs(path[0][0] - start_row) <= 1 and abs(path[0][1] - start_col) <= 1:
            start_neighbors.append(path[0])
    
    # Добавляем соседей старта в граф
    if start_pos not in graph:
        graph[start_pos] = []
    for neighbor in start_neighbors:
        if neighbor not in graph[start_pos]:
            graph[start_pos].append(neighbor)
    
    # Используем BFS для поиска пути от старта до финиша через достижимую спору
    queue = deque([start_pos])
    visited = {start_pos: None}
    path_goes_through_spore = False
    
    while queue:
        current = queue.popleft()
        
        # Проверяем, не находимся ли мы в достижимой споре
        for spore in reachable_spores:
            if current == (spore[0], spore[1]):
                path_goes_through_spore = True
                break
        
        # Если это финиш и мы прошли через достижимую спору
        if current == finish_pos and path_goes_through_spore:
            # Восстанавливаем путь
            path = []
            temp = current
            while temp:
                path.append(temp)
                temp = visited[temp]
            return list(reversed(path))
        
        # Добавляем соседей из графа
        if current in graph:
            for next_pos in graph[current]:
                if next_pos not in visited:
                    queue.append(next_pos)
                    visited[next_pos] = current
    
    # Если не нашли путь, возвращаем False
    return False

def reset_simulation():
    global grid, spores, agent_paths, successful_paths, failed_paths, can_reach_finish, final_path, current_spore_index, mode
    
    grid = np.zeros((ROWS, COLS), dtype=int)
    grid[start_pos[0]][start_pos[1]] = 2
    grid[finish_pos[0]][finish_pos[1]] = 3
    spores = []
    agent_paths = []
    successful_paths = []
    failed_paths = []
    can_reach_finish = set()
    final_path = []
    current_spore_index = -1
    mode = "generate_spore"

def place_walls():
    # Размещаем стены
    for _ in range(100):  # Уменьшено количество стен до 100
        row = random.randint(0, ROWS-1)
        col = random.randint(0, COLS-1)
        
        if grid[row][col] == 0:  # Размещаем только на пустых клетках
            grid[row][col] = 1
    
    # Добавляем несколько линейных стен
    for _ in range(15):  # Немного уменьшено количество линейных стен
        start_row = random.randint(0, ROWS-1)
        start_col = random.randint(0, COLS-1)
        length = random.randint(5, 10)
        direction = random.choice([(0, 1), (1, 0), (1, 1), (1, -1)])
        
        for i in range(length):
            row = start_row + direction[0] * i
            col = start_col + direction[1] * i
            if 0 <= row < ROWS and 0 <= col < COLS and grid[row][col] == 0:
                grid[row][col] = 1
    
    # Удостоверяемся, что старт и финиш имеют хотя бы одну свободную соседнюю клетку
    for pos in [start_pos, finish_pos]:
        row, col = pos
        neighbors = [(row+1, col), (row-1, col), (row, col+1), (row, col-1)]
        
        all_walls = True
        for r, c in neighbors:
            if is_valid_position(r, c) and grid[r][c] != 1:
                all_walls = False
                break
        
        if all_walls:
            # Очищаем одну соседнюю клетку
            for r, c in neighbors:
                if 0 <= r < ROWS and 0 <= c < COLS:
                    grid[r][c] = 0
                    break

def main():
    global final_path, current_spore_index, mode, failed_paths, auto_mode
    
    # Размещаем несколько стен
    place_walls()
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    reset_simulation()
                    place_walls()
                elif event.key == pygame.K_j:
                    # Переключение автоматического режима
                    auto_mode = not auto_mode
                elif event.key == pygame.K_n:
                    # Сначала удаляем споры, помеченные для удаления
                    i = 0
                    while i < len(spores):
                        if spores[i][3]:  # Если спора помечена для удаления
                            # Удаляем спору
                            grid[spores[i][0]][spores[i][1]] = 0
                            spores.pop(i)
                        else:
                            i += 1
                    
                    # Очищаем пути неуспешных агентов
                    failed_paths = []
                    
                    if mode == "generate_spore":
                        # Генерируем новую спору
                        if not spores:
                            generate_spore_near_finish()
                        else:
                            # Проверяем, можем ли найти путь
                            if not final_path:
                                path = check_for_path()
                                if path:
                                    final_path = path
                                    continue
                            
                            generate_spore_near_existing()
                        
                        # Переключаемся в режим агента и сразу запускаем его
                        mode = "run_agent"
                        current_spore_index = len(spores) - 1
                        run_agent()  # Запускаем агента сразу после создания споры
                        
                        # Обновляем статусы всех спор (проверяем косвенную достижимость)
                        update_reachable_spores()
                        
                        # После завершения работы агента проверяем наличие пути
                        if not final_path:
                            path = check_for_path()
                            if path:
                                final_path = path
        
        # Рисуем всё
        screen.fill(WHITE)
        draw_grid()
        
        # Отображаем текущий режим
        font = pygame.font.SysFont(None, 24)
        mode_text = f"Режим: {'Генерация споры' if mode == 'generate_spore' else 'Движение агента'}"
        text_surface = font.render(mode_text, True, BLACK)
        screen.blit(text_surface, (WIDTH - 250, 10))
        
        # Отображаем количество живых спор
        alive_spores = len([s for s in spores if not s[3]])
        spores_text = f"Живых спор: {alive_spores}"
        spores_surface = font.render(spores_text, True, BLACK)
        screen.blit(spores_surface, (WIDTH - 250, 40))
        
        # Отображаем инструкции и статус в нижней части экрана, с большим отступом от левого края
        instructions = "n - создать спору, r - перезапустить, j - автоматический режим, q - выйти"
        instructions_surface = font.render(instructions, True, BLACK)
        screen.blit(instructions_surface, (110, HEIGHT - 30))
        
        # Отображаем статус автоматического режима в нижней части экрана, с большим отступом от левого края
        auto_status = f"Автоматический режим: {'Вкл' if auto_mode else 'Выкл'}"
        auto_surface = font.render(auto_status, True, BLUE if auto_mode else BLACK)
        screen.blit(auto_surface, (110, HEIGHT - 60))
        
        # В автоматическом режиме выполняем действия n
        if auto_mode:
            # Сначала удаляем споры, помеченные для удаления
            i = 0
            while i < len(spores):
                if spores[i][3]:  # Если спора помечена для удаления
                    # Удаляем спору
                    grid[spores[i][0]][spores[i][1]] = 0
                    spores.pop(i)
                else:
                    i += 1
            
            # Очищаем пути неуспешных агентов
            failed_paths = []
            
            if mode == "generate_spore":
                # Генерируем новую спору
                if not spores:
                    generate_spore_near_finish()
                else:
                    # Проверяем, можем ли найти путь
                    if not final_path:
                        path = check_for_path()
                        if path:
                            final_path = path
                            auto_mode = False  # Выключаем автоматический режим при нахождении пути
                            pygame.time.delay(500)  # Небольшая задержка, чтобы увидеть найденный путь
                            continue
                    
                    generate_spore_near_existing()
                
                # Переключаемся в режим агента и сразу запускаем его
                mode = "run_agent"
                current_spore_index = len(spores) - 1
                run_agent()  # Запускаем агента сразу после создания споры
                
                # Обновляем статусы всех спор (проверяем косвенную достижимость)
                update_reachable_spores()
                
                # После завершения работы агента проверяем наличие пути
                if not final_path:
                    path = check_for_path()
                    if path:
                        final_path = path
                        auto_mode = False  # Выключаем автоматический режим при нахождении пути
                
                # Небольшая задержка между итерациями, чтобы можно было видеть процесс
                pygame.time.delay(300)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()