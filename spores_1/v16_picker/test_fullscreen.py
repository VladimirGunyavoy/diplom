"""
Тестовый скрипт для проверки полноэкранного режима
=================================================

Этот скрипт демонстрирует работу полноэкранного режима в WindowManager.
"""

import sys
import os
from ursina import *

# --- Настройка путей для импорта ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.managers.window_manager import WindowManager

print("=== ТЕСТ ПОЛНОЭКРАННОГО РЕЖИМА ===")
print("🖥️ Тестирование WindowManager с поддержкой полноэкранного режима")
print("=" * 50)

# ===== НАСТРОЙКИ =====
FULLSCREEN_MODE = True  # Измените на False для оконного режима

# ===== ИНИЦИАЛИЗАЦИЯ =====
app = Ursina()

# ===== СОЗДАНИЕ WINDOW MANAGER =====
window_manager = WindowManager(
    title="Тест полноэкранного режима", 
    monitor='main', 
    fullscreen=FULLSCREEN_MODE
)

print(f"✓ WindowManager создан")
print(f"✓ Монитор: {window_manager.get_current_monitor()}")
print(f"✓ Полноэкранный режим: {'включен' if window_manager.is_fullscreen() else 'выключен'}")

# ===== СОЗДАНИЕ ПРОСТОЙ СЦЕНЫ =====
# Создаем простой куб для визуализации
cube = Entity(
    model='cube',
    color=color.blue,
    scale=2,
    position=(0, 0, 0)
)

# Добавляем текст с инструкциями
instructions = Text(
    text="F11 - переключить полноэкранный режим\nQ или Escape - выход",
    position=(-0.8, 0.4),
    scale=1.5,
    color=color.white
)

print("\n🎮 Управление:")
print("   F11 - переключить полноэкранный режим")
print("   Q или Escape - выход")
print("   Мышь - вращение камеры")

# ===== ОБРАБОТКА ВВОДА =====
def input(key):
    """Обработчик ввода."""
    if key == 'q' or key == 'escape':
        print("👋 Выход из приложения")
        application.quit()
        return
    
    if key == 'f11':
        window_manager.toggle_fullscreen()
        status = 'включен' if window_manager.is_fullscreen() else 'выключен'
        print(f"🖥️ Полноэкранный режим: {status}")
        
        # Обновляем текст инструкций
        instructions.text = f"F11 - переключить полноэкранный режим\nQ или Escape - выход\n\nРежим: {status}"

# ===== ЗАПУСК =====
print("\n🚀 Тест запущен!")
print("=" * 50)

if __name__ == '__main__':
    app.run()






