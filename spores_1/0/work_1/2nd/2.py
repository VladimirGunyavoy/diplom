import pygame
import random
import math
from heapq import heappop, heappush

# Константы
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 30
CELL_SIZE = WIDTH // GRID_SIZE

# Цвета
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)  # Цвет для неактивных спор

class Cell:
    def __init__(self):
        self.color = WHITE
        self.is_spore = False
        self.reachable = False
        self.parent = None
        self.visited = False  # Флаг посещения агентами

class Spore:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.agents = [Agent(self) for _ in range(5)]  # 5 агентов на спору
        self.reachable = False  # Может ли эта спора достичь финиша
        self.active = True  # Активна ли спора

class Agent:
    def __init__(self, spore):
        self.spore = spore
        self.path = []  # История движения агента

class Simulation:
    def __init__(self):
        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.spores = []  # Список всех спор
        self.reachable_spores = []  # Споры, из которых можно достичь финиша
        self.start = (1, 1)  # Начальная позиция
        self.finish = (GRID_SIZE-2, GRID_SIZE-2)  # Финишная позиция
        self.final_path = []  # Финальный найденный путь
        self.walls = []  # Список позиций стен

        # Инициализация старта и финиша
        self.grid[self.start[1]][self.start[0]].color = GREEN
        self.grid[self.finish[1]][self.finish[0]].color = RED
        
        # Генерация стен (препятствий)
        self.generate_walls()
        
        # Убедимся, что старт и финиш имеют свободный путь хотя бы с одной стороны
        self.ensure_path_exists()
        
    def generate_walls(self):
        # Генерируем стены с вероятностью 20%
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Пропускаем старт и финиш и клетки возле них
                if ((x, y) == self.start or (x, y) == self.finish or
                    (abs(x - self.start[0]) <= 1 and abs(y - self.start[1]) <= 1) or
                    (abs(x - self.finish[0]) <= 1 and abs(y - self.finish[1]) <= 1)):
                    continue
                    
                if random.random() < 0.2:  # 20% вероятность стены
                    self.grid[y][x].color = BLACK
                    self.walls.append((x, y))
    
    def ensure_path_exists(self):
        # Убеждаемся, что у старта и финиша есть хотя бы один свободный соседний путь
        for pos in [self.start, self.finish]:
            has_free_neighbor = False
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx].color == WHITE:
                    has_free_neighbor = True
                    break
            
            # Если нет свободного соседа, освобождаем случайного соседа
            if not has_free_neighbor:
                neighbors = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        neighbors.append((nx, ny))
                
                if neighbors:
                    nx, ny = random.choice(neighbors)
                    self.grid[ny][nx].color = WHITE
                    if (nx, ny) in self.walls:
                        self.walls.remove((nx, ny))
    
    def create_spore(self):
        # Находим доступные клетки для создания споры
        available_cells = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Спора создается только там, где еще не было спор и агенты не проходили
                if (self.grid[y][x].color == WHITE and 
                    not self.grid[y][x].is_spore and 
                    not self.grid[y][x].visited):
                    available_cells.append((x, y))
        
        # Если нет доступных клеток, пробуем использовать любые белые клетки
        if not available_cells:
            available_cells = [(x, y) for y in range(GRID_SIZE) for x in range(GRID_SIZE) 
                              if self.grid[y][x].color == WHITE and not self.grid[y][x].is_spore]
        
        # Если все еще нет доступных клеток, выходим
        if not available_cells:
            print("Нет доступных клеток для создания споры!")
            return
        
        # Выбираем случайную доступную клетку
        x, y = random.choice(available_cells)
        
        # Создаем спору
        self.grid[y][x].is_spore = True
        self.grid[y][x].color = YELLOW
        new_spore = Spore(x, y)
        self.spores.append(new_spore)
        
        # Запускаем агентов из новой споры
        self.run_agents_from_spore(new_spore)
        
        # Проверяем, может ли новая спора достичь финиша или другой достижимой споры
        self.check_spore_reachability(new_spore)
        
        # Обновляем статус всех спор
        self.update_all_spores_status()
        
        # Проверяем, можно ли построить путь от старта до финиша
        if self.can_reach_finish_from_start():
            self.compute_final_path()
    
    def run_agents_from_spore(self, spore):
        # Запускаем 5 агентов из споры
        for agent in spore.agents:
            agent.path = [(spore.x, spore.y)]  # Начальная позиция
            
            # Каждый агент делает 5 шагов
            current_x, current_y = spore.x, spore.y
            for _ in range(5):
                # Определяем направление движения (с учетом нормального распределения в сторону цели)
                next_x, next_y = self.get_next_position(current_x, current_y, spore)
                
                # Проверяем, что новая позиция допустима
                if 0 <= next_x < GRID_SIZE and 0 <= next_y < GRID_SIZE:
                    # Не ходим через стены
                    if self.grid[next_y][next_x].color == BLACK:
                        break
                    
                    # Отмечаем клетку как посещенную агентом
                    if self.grid[next_y][next_x].color == WHITE:
                        self.grid[next_y][next_x].color = BLUE
                    
                    self.grid[next_y][next_x].visited = True
                    
                    # Перемещаем агента
                    current_x, current_y = next_x, next_y
                    agent.path.append((current_x, current_y))
                    
                    # Если достигли финиша, помечаем спору как достижимую
                    if (current_x, current_y) == self.finish:
                        spore.reachable = True
                        if spore not in self.reachable_spores:
                            self.reachable_spores.append(spore)
                        break
    
    def get_next_position(self, x, y, spore):
        # Определяем цель для агента
        target_x, target_y = self.get_target_for_spore(spore)
        
        # Вычисляем вектор направления к цели
        dir_x = target_x - x
        dir_y = target_y - y
        
        # Определяем возможные направления движения
        directions = []
        
        # Предпочитаем направление ближе к цели
        if abs(dir_x) > abs(dir_y):
            if dir_x > 0:
                directions.append((1, 0))  # Вправо
            elif dir_x < 0:
                directions.append((-1, 0))  # Влево
            
            if dir_y > 0:
                directions.append((0, 1))  # Вниз
            elif dir_y < 0:
                directions.append((0, -1))  # Вверх
        else:
            if dir_y > 0:
                directions.append((0, 1))  # Вниз
            elif dir_y < 0:
                directions.append((0, -1))  # Вверх
                
            if dir_x > 0:
                directions.append((1, 0))  # Вправо
            elif dir_x < 0:
                directions.append((-1, 0))  # Влево
        
        # Добавляем все возможные направления
        all_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for d in all_directions:
            if d not in directions:
                directions.append(d)
        
        # С вероятностью 70% выбираем направление ближе к цели, иначе - случайное
        if directions and random.random() < 0.7:
            return x + directions[0][0], y + directions[0][1]
        else:
            dx, dy = random.choice(all_directions)
            return x + dx, y + dy
    
    def get_target_for_spore(self, spore):
        # Если уже есть споры, из которых можно достичь финиша
        if self.reachable_spores and not spore.reachable:
            # Находим ближайшую достижимую спору
            min_dist = float('inf')
            closest_spore = None
            
            for reach_spore in self.reachable_spores:
                dist = abs(spore.x - reach_spore.x) + abs(spore.y - reach_spore.y)
                if dist < min_dist:
                    min_dist = dist
                    closest_spore = reach_spore
            
            # Если расстояние до финиша меньше, чем до ближайшей споры, выбираем финиш
            dist_to_finish = abs(spore.x - self.finish[0]) + abs(spore.y - self.finish[1])
            if dist_to_finish < min_dist:
                return self.finish
            else:
                return closest_spore.x, closest_spore.y
        else:
            # Иначе направляемся к финишу
            return self.finish
    
    def check_spore_reachability(self, spore):
        # Проверяем, может ли спора достичь финиша
        for agent in spore.agents:
            for x, y in agent.path:
                # Если агент достиг финиша
                if (x, y) == self.finish:
                    spore.reachable = True
                    if spore not in self.reachable_spores:
                        self.reachable_spores.append(spore)
                    return
                
                # Если агент достиг другой достижимой споры
                for reach_spore in self.reachable_spores:
                    if (x, y) == (reach_spore.x, reach_spore.y):
                        spore.reachable = True
                        if spore not in self.reachable_spores:
                            self.reachable_spores.append(spore)
                        return
    
    def update_all_spores_status(self):
        # Пересматриваем статус всех спор
        changed = True
        while changed:
            changed = False
            
            for spore in self.spores:
                if spore.reachable:
                    continue
                    
                # Проверяем, достигает ли какой-то агент споры другой достижимой споры
                for agent in spore.agents:
                    for x, y in agent.path:
                        for reach_spore in self.reachable_spores:
                            if (x, y) == (reach_spore.x, reach_spore.y):
                                spore.reachable = True
                                if spore not in self.reachable_spores:
                                    self.reachable_spores.append(spore)
                                    changed = True
                                break
                        if spore.reachable:
                            break
                    if spore.reachable:
                        break
        
        # Обновляем отображение достижимых спор
        for spore in self.spores:
            x, y = spore.x, spore.y
            if spore.reachable and self.grid[y][x].color == YELLOW:
                self.grid[y][x].color = ORANGE
    
    def can_reach_finish_from_start(self):
        # Проверяем, можно ли из старта достичь какой-либо достижимой споры
        for spore in self.reachable_spores:
            # Проверяем, находится ли спора рядом со стартом
            if abs(spore.x - self.start[0]) + abs(spore.y - self.start[1]) == 1:
                return True
        
        return False
    
    def compute_final_path(self):
        # Алгоритм A* для поиска пути от старта до финиша через достижимые споры
        open_set = []
        heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.finish)}
        
        while open_set:
            _, current = heappop(open_set)
            
            if current == self.finish:
                # Путь найден
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                self.final_path = path
                return
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Проверка границ и проходимости
                if not(0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                    continue
                
                # Нельзя идти через стены
                if self.grid[neighbor[1]][neighbor[0]].color == BLACK:
                    continue
                
                # Можно идти только через клетки, помеченные как проходимые агентами,
                # достижимые споры, старт и финиш
                if (self.grid[neighbor[1]][neighbor[0]].color not in 
                    [BLUE, ORANGE, GREEN, RED] and not self.grid[neighbor[1]][neighbor[0]].visited):
                    continue
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, self.finish)
                    
                    # Проверяем, есть ли уже этот узел в очереди
                    in_open_set = False
                    for i, (_, pos) in enumerate(open_set):
                        if pos == neighbor:
                            in_open_set = True
                            # Если текущая оценка лучше, обновляем
                            if f_score < open_set[i][0]:
                                open_set[i] = (f_score, neighbor)
                            break
                    
                    if not in_open_set:
                        heappush(open_set, (f_score, neighbor))
    
    def heuristic(self, a, b):
        # Манхэттенское расстояние в качестве эвристики
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def reset(self):
        # Сброс симуляции
        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.spores = []
        self.reachable_spores = []
        self.final_path = []
        self.walls = []
        
        # Инициализация старта и финиша
        self.grid[self.start[1]][self.start[0]].color = GREEN
        self.grid[self.finish[1]][self.finish[0]].color = RED
        
        # Генерация стен и обеспечение доступности старта и финиша
        self.generate_walls()
        self.ensure_path_exists()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Алгоритм планирования пути Spore")
    
    # Создаем симуляцию
    sim = Simulation()
    
    # Шрифт для отображения информации
    font = pygame.font.SysFont('Arial', 16)
    
    running = True
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    # Создание новой споры
                    sim.create_spore()
                if event.key == pygame.K_r:
                    # Сброс симуляции
                    sim.reset()
        
        # Отрисовка сетки
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, sim.grid[y][x].color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
        
        # Отрисовка финального пути, если он найден
        if sim.final_path:
            for i in range(len(sim.final_path) - 1):
                start_pos = sim.final_path[i]
                end_pos = sim.final_path[i + 1]
                
                # Рисуем линию между точками пути
                start_screen_pos = (start_pos[0] * CELL_SIZE + CELL_SIZE // 2, start_pos[1] * CELL_SIZE + CELL_SIZE // 2)
                end_screen_pos = (end_pos[0] * CELL_SIZE + CELL_SIZE // 2, end_pos[1] * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.line(screen, GREEN, start_screen_pos, end_screen_pos, 3)
        
        # Отображение информации
        info_text = [
            f"Спор: {len(sim.spores)}, Достижимых: {len(sim.reachable_spores)}",
            f"Найден путь: {'Да' if sim.final_path else 'Нет'}",
            "N - создать спору, R - сбросить симуляцию"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 20))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()