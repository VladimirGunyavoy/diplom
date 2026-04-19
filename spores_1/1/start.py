from ursina import *
from src.scene_setup import SceneSetup
from src.fame import Frame
import numpy as np

class StatePoint:
    """Класс для представления точки состояния системы"""
    def __init__(self, position, is_start=False, is_goal=False):
        self.position = position
        self.is_start = is_start
        self.is_goal = is_goal
        self.connections = []  # Список связей с другими точками
        
        # Создаем визуальное представление точки
        self.entity = Entity(
            model='sphere',
            color=color.green if is_start else (color.red if is_goal else color.blue),
            scale=0.1,
            position=position
        )
        
        self.label = Text(
            text='Старт' if is_start else 'Цель',
            position=(position[0] - 0.1, position[1] - 0.3),
            scale=2,
            color=color.green if is_start else color.red,
            background=True,
            parent=self.entity
        )

class PendulumSystem:
    """Класс, описывающий систему маятника"""
    def __init__(self):
        # Параметры маятника
        self.g = 9.81  # ускорение свободного падения
        self.l = 2.0   # длина маятника
        self.m = 1.0   # масса
        
    def dynamics(self, state, control):
        """
        Динамика маятника: θ'' = -g/l * sin(θ) + u
        state = [θ, θ']
        control = u (управление)
        """
        theta, theta_dot = state
        theta_ddot = -self.g/self.l * np.sin(theta) + control
        return np.array([theta_dot, theta_ddot])

class ReachabilityGraph:
    """Класс для хранения графа достижимости"""
    def __init__(self):
        self.points = []  # список точек состояния
        self.connections = []  # список связей между точками
        
    def add_point(self, point):
        """Добавление новой точки в граф"""
        self.points.append(point)
        
    def add_connection(self, point1, point2, control):
        """Добавление связи между точками"""
        connection = (point1, point2, control)
        self.connections.append(connection)
        point1.connections.append(connection)
        point2.connections.append(connection)
        
        # Визуализация связи
        Entity(
            model='cylinder',
            color=color.white,
            scale=(0.02, distance(point1.position, point2.position), 0.02),
            position=lerp(point1.position, point2.position, 0.5),
            rotation=Vec3(0, 0, 0).look_at_2d(point2.position - point1.position)
        )

class PendulumScene:
    def __init__(self):
        # Инициализация сцены
        self.scene = SceneSetup()
        
        # Создаем систему координат
        self.frame = Frame(position=(0, 0, 0))

        self.x_label = Text(
            text='θ',
            parent=self.frame.x_axis,
            position=(-1/10, 1, 0),
            scale=5,
            color=color.white,
            billboard=True,
            origin=(0, 0) 
        )

        
        self.z_label = Text(
            text='ω',
            parent=self.frame.z_axis,
            position=(0, 1, -1/10),
            scale=5,
            color=color.white,
            billboard=True,
            origin=(0, 0) 
        )


        
        # Создаем систему маятника
        self.system = PendulumSystem()
        
        # Создаем граф достижимости
        self.graph = ReachabilityGraph()
        
        # Создаем стартовую и целевую точки
        self.start_point = StatePoint(
            position=(0, 0, 0),
            is_start=True
        )

        self.start_label = Text(
            text='start',
            parent=self.start_point.entity,
            position=(0, 3, 0),
            scale=50,
            color=color.white,
            billboard=True,
            origin=(0, 0),
            background=True
        )
        self.start_label.background.alpha = 0.75


        self.goal_point = StatePoint(
            position=(np.pi, 0, 0),
            is_goal=True
        )
        
        # Добавляем точки в граф
        self.graph.add_point(self.start_point)
        self.graph.add_point(self.goal_point)
        
        # Добавляем ось маятника
        self.theta_lims = (-2*np.pi, 2*np.pi)
        self.omega_lims = (-5, 5)

        self.draw_lims()

        


    def draw_lims(self):
        lim_color = color.black
        lim_alpha = 0.75

        scales = [(1/20,
                  self.omega_lims[1] - self.omega_lims[0],
                  1/20),
                  (self.theta_lims[1] - self.theta_lims[0],
                  1/20,
                  1/20),
                  ]
        poses = [(self.theta_lims[0], 0, 0),
                 (self.theta_lims[1], 0, 0),
                 (0, 0, self.omega_lims[0]),
                 (0, 0, self.omega_lims[1]),
        ]
        rots = [(90, 0, 0), (0, 0, 0)]
        bords = []

        for i in range(4):
            print(i, i//2)
            bord = Entity(
                model='cube',
                scale=scales[i//2],
                position=poses[i],
                rotation=rots[i//2]
            )
            bords.append(bord)

    def update(self, dt):
        """Обновление сцены"""
        # Здесь будет логика обновления графа достижимости
        self.scene.update(dt)

# Создаем приложение
app = Ursina()

# Создаем сцену
scene = PendulumScene()


def update():
    scene.update(time.dt)

def input(key):
    scene.scene.input_handler(key)

# Запускаем приложение
app.run()
