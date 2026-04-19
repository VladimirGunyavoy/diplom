import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Добавляем корневую папку проекта в sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Мокаем Ursina перед импортом, так как нам не нужна реальная графика
sys.modules['ursina'] = MagicMock()

from src.visual.visual_manager import VisualManager

class TestVisualManager(unittest.TestCase):

    def setUp(self):
        """Настройка с использованием моков для зависимостей."""
        self.mock_engine = MagicMock()
        self.mock_cost_viz = MagicMock()
        self.mock_spawn_viz = MagicMock()
        self.mock_ghost_viz = MagicMock()

        self.visual_manager = VisualManager(
            engine=self.mock_engine,
            cost_visualizer=self.mock_cost_viz,
            spawn_visualizer=self.mock_spawn_viz,
            ghost_visualizer=self.mock_ghost_viz
        )

    def test_initialization_and_parenting(self):
        """
        Тест: Проверяет, что VisualManager правильно инициализирует
        и устанавливает родительские связи для своих компонентов.
        """
        self.assertEqual(self.visual_manager.engine, self.mock_engine)
        self.assertEqual(self.visual_manager.cost_visualizer, self.mock_cost_viz)
        self.assertEqual(self.visual_manager.spawn_visualizer, self.mock_spawn_viz)
        self.assertEqual(self.visual_manager.ghost_visualizer, self.mock_ghost_viz)
        
        self.assertEqual(self.mock_cost_viz.parent, self.visual_manager)
        self.assertEqual(self.mock_spawn_viz.parent, self.visual_manager)
        self.assertEqual(self.mock_ghost_viz.parent, self.visual_manager)

    def test_toggle_methods(self):
        """
        Тест: Проверяет, что toggle-методы корректно изменяют
        свойство 'enabled' у дочерних визуализаторов.
        """
        self.visual_manager.toggle_cost_surface(True)
        self.visual_manager.toggle_spawn_area(True)
        self.visual_manager.toggle_ghosts(True)
        
        self.assertTrue(self.mock_cost_viz.enabled)
        self.assertTrue(self.mock_spawn_viz.enabled)
        self.assertTrue(self.mock_ghost_viz.enabled)
        
        self.visual_manager.toggle_cost_surface(False)
        self.visual_manager.toggle_spawn_area(False)
        self.visual_manager.toggle_ghosts(False)
        
        self.assertFalse(self.mock_cost_viz.enabled)
        self.assertFalse(self.mock_spawn_viz.enabled)
        self.assertFalse(self.mock_ghost_viz.enabled)
        
    @patch('src.visual.visual_manager.SporeVisual') 
    @patch('src.visual.visual_manager.destroy') 
    def test_sync_with_engine(self, mock_destroy, mock_spore_visual):
        """
        Тест: Проверяет логику синхронизации с движком.
        """
        # --- Сценарий 1: Движок имеет 2 споры, у нас 0. Должны создаться 2. ---
        self.mock_engine.get_all_spore_states.return_value = {
            'id1': {'position': [1, 1]},
            'id2': {'position': [2, 2]}
        }
        self.visual_manager.sync_with_engine()
        
        self.assertEqual(mock_spore_visual.call_count, 2, "Должны быть созданы 2 визуальные споры")
        self.assertEqual(len(self.visual_manager.visual_spores), 2)
        
        # --- Сценарий 2: Движок имеет 1 спору, у нас 2. Должна удалиться 1. ---
        self.mock_engine.get_all_spore_states.return_value = {
            'id1': {'position': [3, 3]}
        }
        self.visual_manager.sync_with_engine()
        
        mock_destroy.assert_called_once() 
        self.assertEqual(len(self.visual_manager.visual_spores), 1)
        self.assertIn('id1', self.visual_manager.visual_spores)
        
        # --- Сценарий 3: Позиция существующей споры должна обновиться ---
        spore_to_update = self.visual_manager.visual_spores['id1']
        self.visual_manager.sync_with_engine()
        
        self.assertEqual(spore_to_update.position, (3, 0, 3))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 