from pathlib import Path

# Определяем корневую директорию проекта spores/10
# __file__ -> .../spores/10/src/config/paths.py
# .parent -> .../spores/10/src/config
# .parent.parent -> .../spores/10/src
# .parent.parent.parent -> .../spores/10
SPORES_10_ROOT = Path(__file__).parent.parent.parent

# Путь к директории с ассетами (моделями, текстурами и т.д.)
ASSETS_PATH = SPORES_10_ROOT / 'src' / 'assets'

# Путь к модели стрелки в формате, понятном для Ursina.
# str() преобразует объект Path в строку с разделителями ОС (e.g., '\' на Windows),
# .replace() гарантирует, что мы используем '/' для совместимости.
ARROW_MODEL_PATH = str(ASSETS_PATH / 'arrow.obj').replace('\\', '/')

# Пример использования:
# from src.config.paths import MODELS_PATH
# arrow_model_path = MODELS_PATH / 'arrow.obj' 