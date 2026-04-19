import json
import os
from ursina import color

class ColorManager:
    def __init__(self, colors_file_path=None):
        if colors_file_path is None:
            # Путь к файлу colors.json относительно корневой папки проекта
            # ../.. -> managers/ -> src/ -> spores/9/
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            colors_file_path = os.path.join(project_root, 'config', 'json', 'colors.json')
        
        self.colors_file_path = colors_file_path
        self.colors = self._load_colors()
    
    def _load_colors(self):
        """Загружает цвета из JSON файла"""
        try:
            with open(self.colors_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Файл цветов {self.colors_file_path} не найден. Используются цвета по умолчанию.")
            return self._get_default_colors()
        except json.JSONDecodeError as e:
            print(f"Ошибка при разборе JSON файла цветов: {e}. Используются цвета по умолчанию.")
            return self._get_default_colors()
    
    def _get_default_colors(self):
        """Возвращает цвета по умолчанию в случае проблем с загрузкой"""
        return {
            "frame": {
                "origin": [1.0, 1.0, 1.0, 1.0],
                "x_axis": [0.863, 0.196, 0.184, 1.0],
                "y_axis": [0.2, 0.7, 0.25, 1.0],
                "z_axis": [0.149, 0.545, 0.824, 1.0]
            },
            "scene": {
                "floor": [0.3, 0.3, 0.4, 0.5],
                "window_background": [0.08, 0.1, 0.2, 1.0],
                "ambient_light": [0.6, 0.6, 0.65, 1.0],
                "directional_light": [1.0, 1.0, 1.0, 1.0]
            },
            "spore": {
                "default": [0.6, 0.4, 0.9, 1.0],
                "ghost": [1.0, 1.0, 1.0, 0.47]
            },
            "link": {
                "default": [0.78, 0.78, 0.78, 0.59]
            },
            "ui": {
                "text_primary": [1.0, 1.0, 1.0, 1.0],
                "text_secondary": [1.0, 1.0, 1.0, 0.7],
                "background_transparent": [0.0, 0.0, 0.0, 0.7],
                "background_solid": [0.0, 0.0, 0.0, 1.0]
            }
        }
    
    def get_color(self, category, color_name):
        """
        Получает цвет в формате Ursina
        
        Args:
            category (str): Категория цвета (frame, scene, spore, link, ui)
            color_name (str): Название цвета в категории
            
        Returns:
            ursina.color: Цвет в формате Ursina
        """
        try:
            rgba = self.colors[category][color_name]
            return color.rgba(*rgba)
        except KeyError:
            print(f"Цвет {category}.{color_name} не найден. Используется белый цвет.")
            return color.white
    
    def get_rgba(self, category, color_name):
        """
        Получает цвет в формате RGBA tuple
        
        Args:
            category (str): Категория цвета
            color_name (str): Название цвета в категории
            
        Returns:
            tuple: RGBA значения (r, g, b, a)
        """
        try:
            return tuple(self.colors[category][color_name])
        except KeyError:
            print(f"Цвет {category}.{color_name} не найден. Используется белый цвет.")
            return (1.0, 1.0, 1.0, 1.0)
    
    def reload_colors(self):
        """Перезагружает цвета из файла"""
        self.colors = self._load_colors()
        print("Цвета перезагружены из файла.")
    
    def save_colors(self):
        """Сохраняет текущие цвета в файл"""
        try:
            with open(self.colors_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.colors, f, ensure_ascii=False, indent=2)
            print(f"Цвета сохранены в {self.colors_file_path}")
        except Exception as e:
            print(f"Ошибка при сохранении цветов: {e}")
    
    def set_color(self, category, color_name, rgba):
        """
        Устанавливает новый цвет
        
        Args:
            category (str): Категория цвета
            color_name (str): Название цвета
            rgba (list): RGBA значения [r, g, b, a]
        """
        if category not in self.colors:
            self.colors[category] = {}
        self.colors[category][color_name] = rgba
        print(f"Цвет {category}.{color_name} установлен на {rgba}")

    def get_value(self, section, name):
        """Возвращает отдельное значение (не цвет) из конфига."""
        try:
            return self.colors[section][name]
        except KeyError:
            print(f"Warning: Value for '{section}' -> '{name}' not found. Returning default (1).")
            return 1 