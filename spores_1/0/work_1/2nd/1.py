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

class Cell:
    def __init__(self):
        self.color = WHITE
        self.is_spore = False
        self.reachable = False
        self.parent = None

class Spore:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.agents = [Agent(self) for _ in range(5)]
        self.reachable = False

class Agent:
    def __init__(self, spore):
        self.spore = spore
        self.path = []

class Simulation:
    def __init__(self):
        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.spores = []
        self.reachable_spores = []
        self.start = (1, 1)
        self.finish = (GRID_SIZE-2, GRID_SIZE-2)
        self.final_path = []
        
        # Инициализация старта и финиша
        self.grid[self.start[1]][self.start[0]].color = GREEN
        self.grid[self.finish[1]][self.finish[0]].color = RED
        
    def create_spore(self):
        empty_cells = [
            (x, y) 
            for y in range(GRID_SIZE) 
            for x in range(GRID_SIZE) 
            if self.grid[y][x].color == WHITE
        ]
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.grid[y][x].is_spore = True
            self.grid[y][x].color = YELLOW
            self.spores.append(Spore(x, y))
            
    def reset(self):
        self.grid = [[Cell() for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.spores = []
        self.reachable_spores = []
        self.final_path = []
        self.grid[self.start[1]][self.start[0]].color = GREEN
        self.grid[self.finish[1]][self.finish[0]].color = RED

    def move_agents(self):
        for spore in self.spores:
            if spore.reachable:
                continue
            for agent in spore.agents:
                x, y = spore.x, spore.y
                for _ in range(5):
                    dx, dy = self.get_direction(x, y)
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                        if self.grid[new_y][new_x].color == RED:
                            spore.reachable = True
                            self.grid[spore.y][spore.x].color = ORANGE
                            self.reachable_spores.append(spore)
                            self.update_final_path()
                            return
                        if self.grid[new_y][new_x].color == WHITE:
                            self.grid[new_y][new_x].color = BLUE
                        x, y = new_x, new_y

    def get_direction(self, x, y):
        closest = self.find_closest_reachable(x, y)
        target = closest if closest else self.finish
        
        directions = []
        if target[0] > x:
            directions.append((1, 0))
        elif target[0] < x:
            directions.append((-1, 0))
        if target[1] > y:
            directions.append((0, 1))
        elif target[1] < y:
            directions.append((0, -1))
        
        if not directions:
            return random.choice([(0,1), (0,-1), (1,0), (-1,0)])
        
        main_dir = random.choice(directions)
        if random.random() < 0.7:
            return main_dir
        else:
            return random.choice([(0,1), (0,-1), (1,0), (-1,0)])

    def find_closest_reachable(self, x, y):
        min_dist = float('inf')
        closest = None
        for spore in self.reachable_spores:
            dist = abs(spore.x - x) + abs(spore.y - y)
            if dist < min_dist:
                min_dist = dist
                closest = (spore.x, spore.y)
        return closest

    def update_final_path(self):
        if not self.reachable_spores:
            return
        
        # Поиск пути A*
        start = self.start
        finish = self.finish
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = { (x,y): float('inf') for y in range(GRID_SIZE) for x in range(GRID_SIZE)}
        g_score[start] = 0

        while open_set:
            current = heappop(open_set)[1]
            if current == finish:
                break

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                neighbor = (current[0]+dx, current[1]+dy)
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                    if self.grid[neighbor[1]][neighbor[0]].color in [BLUE, ORANGE, RED]:
                        tentative_g = g_score[current] + 1
                        if tentative_g < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g
                            f_score = tentative_g + self.heuristic(neighbor, finish)
                            heappush(open_set, (f_score, neighbor))

        # Восстановление пути
        path = []
        current = finish
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        self.final_path = path[::-1]

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    sim = Simulation()

    running = True
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    sim.create_spore()
                    sim.move_agents()
                if event.key == pygame.K_r:
                    sim.reset()

        # Отрисовка сетки
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, sim.grid[y][x].color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

        # Отрисовка финального пути
        for (x, y) in sim.final_path:
            rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
