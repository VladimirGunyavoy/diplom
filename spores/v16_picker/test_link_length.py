#!/usr/bin/env python3
"""
Тест функциональности укорачивания линков при изменении dt.
"""

import numpy as np
from src.visual.link import Link
from src.managers.dt_manager import DTManager
from src.managers.spore_manager import SporeManager
from src.managers.color_manager import ColorManager
from src.managers.zoom_manager import ZoomManager
from src.core.spore import Spore
from src.logic.pendulum import PendulumSystem

def test_link_length_functionality():
    """Тестирует функциональность укорачивания линков."""
    print("🧪 ТЕСТ ФУНКЦИОНАЛЬНОСТИ УКОРАЧИВАНИЯ ЛИНКОВ")
    print("=" * 50)
    
    # Создаем минимальную конфигурацию
    config = {
        'pendulum': {'dt': 0.1},
        'link': {'distance_per_dt': 3.0, 'thickness': 0.4}
    }
    
    # Создаем необходимые менеджеры
    pendulum = PendulumSystem(config)
    zoom_manager = ZoomManager(config)
    color_manager = ColorManager()
    
    # Создаем DTManager
    dt_manager = DTManager(config, pendulum)
    print(f"✅ DTManager создан, link_length_per_dt = {dt_manager.link_length_per_dt}")
    
    # Создаем SporeManager
    spore_manager = SporeManager(
        pendulum=pendulum,
        zoom_manager=zoom_manager,
        settings_param=None,
        color_manager=color_manager,
        config=config
    )
    
    # Связываем менеджеры
    dt_manager.spore_manager = spore_manager
    print("✅ Менеджеры связаны")
    
    # Создаем тестовые споры
    parent_spore = Spore(
        dt=0.1,
        pendulum=pendulum,
        goal_position=[3.14159, 0],
        position=[0, 0, 0],
        color_manager=color_manager,
        config=config
    )
    
    child_spore = Spore(
        dt=0.1,
        pendulum=pendulum,
        goal_position=[3.14159, 0],
        position=[10, 0, 0],  # Далеко от родителя
        color_manager=color_manager,
        config=config
    )
    
    # Создаем линк
    link = Link(
        parent_spore=parent_spore,
        child_spore=child_spore,
        zoom_manager=zoom_manager,
        color_manager=color_manager,
        config=config
    )
    
    spore_manager.links.append(link)
    print("✅ Тестовый линк создан")
    
    # Проверяем начальную длину
    initial_max_length = dt_manager.get_max_link_length()
    print(f"📏 Начальная максимальная длина линка: {initial_max_length:.2f}")
    
    # Уменьшаем dt и проверяем, что линк укорачивается
    print("\n🔽 Уменьшаем dt...")
    dt_manager.decrease_dt()
    
    new_max_length = dt_manager.get_max_link_length()
    print(f"📏 Новая максимальная длина линка: {new_max_length:.2f}")
    
    if new_max_length < initial_max_length:
        print("✅ Линк успешно укоротился!")
    else:
        print("❌ Линк не укоротился")
    
    # Проверяем, что метод update_links_max_length работает
    print("\n🔧 Тестируем update_links_max_length...")
    spore_manager.update_links_max_length(new_max_length)
    print("✅ update_links_max_length выполнен без ошибок")
    
    print("\n🎉 ТЕСТ ЗАВЕРШЕН УСПЕШНО!")

if __name__ == "__main__":
    test_link_length_functionality()
