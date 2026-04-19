import unittest
import numpy as np
import sys
import os

# Добавляем корневую папку проекта в sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.simulation_engine import SimulationEngine
from src.logic.pendulum import PendulumSystem
from src.logic.spawn_area import SpawnArea
from src.logic.ghost_processor import GhostProcessor

class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        """Настройка тестового окружения перед каждым тестом."""
        self.pendulum = PendulumSystem()
        self.spawn_area = SpawnArea(focus1=np.array([-5, 0]),
                                      focus2=np.array([5, 0]),
                                      eccentricity=0.8)
        self.dt = 0.1
        self.ghost_processor = GhostProcessor(pendulum_system=self.pendulum, dt=self.dt)
        self.goal_pos = np.array([0.0, 0.0])

        self.engine = SimulationEngine(
            pendulum_system=self.pendulum,
            spawn_area=self.spawn_area,
            ghost_processor=self.ghost_processor,
            dt=self.dt,
            goal_position=self.goal_pos
        )

    def test_initialization(self):
        """Тест: движок должен быть инициализирован без спор."""
        self.assertEqual(len(self.engine.spores), 0, "Вначале не должно быть спор")

    def test_spawn_spore(self):
        """Тест: спавн споры должен увеличивать их количество."""
        self.assertEqual(len(self.engine.spores), 0)
        
        spore_id = self.engine.spawn_spore()
        
        self.assertEqual(len(self.engine.spores), 1, "Должна быть одна спора")
        self.assertIn(spore_id, self.engine.spores, "ID споры должен быть в словаре")
        
        # Проверим, что позиция споры находится внутри области спавна
        pos = self.engine.get_spore_state(spore_id)['position']
        self.assertTrue(self.spawn_area.is_inside(pos), "Спора должна спавниться внутри области")

    def test_step_evolution(self):
        """Тест: шаг симуляции должен изменять состояние споры."""
        spore_id = self.engine.spawn_spore()
        initial_state = self.engine.get_spore_state(spore_id)
        initial_pos = initial_state['position'].copy()

        # Выполняем шаг с ненулевым управлением, чтобы гарантировать изменение
        self.engine.step(controls={spore_id: 0.5})
        
        evolved_state = self.engine.get_spore_state(spore_id)
        evolved_pos = evolved_state['position']

        # Для эллипса с фокусами не в центре, is_inside может быть сложным.
        # Просто убедимся, что позиция изменилась.
        self.assertFalse(np.array_equal(initial_pos, evolved_pos),
                         "Позиция споры должна измениться после шага")

    def test_remove_spore(self):
        """Тест: удаление споры должно уменьшать их количество."""
        spore_id_1 = self.engine.spawn_spore()
        spore_id_2 = self.engine.spawn_spore()
        self.assertEqual(len(self.engine.spores), 2)
        
        self.engine.remove_spore(spore_id_1)
        
        self.assertEqual(len(self.engine.spores), 1, "Должна остаться одна спора")
        self.assertNotIn(spore_id_1, self.engine.spores)
        self.assertIn(spore_id_2, self.engine.spores)

    def test_get_ghosts(self):
        """Тест: получение данных о призраках."""
        spore_id = self.engine.spawn_spore()
        num_ghosts = 7
        
        ghost_data = self.engine.get_ghosts(spore_id, num_ghosts=num_ghosts, control_method='mesh')
        
        self.assertIn('states', ghost_data)
        self.assertIn('costs', ghost_data)
        self.assertIn('controls', ghost_data)
        
        self.assertEqual(len(ghost_data['states']), num_ghosts, f"Должно быть {num_ghosts} состояний призраков")
        self.assertEqual(len(ghost_data['costs']), num_ghosts, f"Должно быть {num_ghosts} стоимостей призраков")
        self.assertEqual(len(ghost_data['controls']), num_ghosts, f"Должно быть {num_ghosts} управлений призраков")

if __name__ == '__main__':
    # Создаем директорию для отчета, если ее нет
    reports_dir = os.path.join(project_root, 'tests', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Запускаем тесты
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSimulationEngine))
    runner = unittest.TextTestRunner()
    runner.run(suite) 