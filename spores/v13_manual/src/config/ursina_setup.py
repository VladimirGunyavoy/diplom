from ursina import application
from pathlib import Path

# Определяем корневую директорию проекта spores/10
SPORES_10_ROOT = Path(__file__).parent.parent.parent

# Путь к нашей основной папке с ассетами
CUSTOM_ASSETS_PATH = SPORES_10_ROOT / 'src' / 'assets'

# Преобразуем текущую папку с ассетами в строку, так как application.asset_folder может быть объектом Path.
current_asset_folder_str = str(application.asset_folder)
custom_assets_path_str = str(CUSTOM_ASSETS_PATH)

# Ursina использует ';' как разделитель для нескольких папок.
# Проверяем, не добавлена ли уже наша папка.
if custom_assets_path_str not in current_asset_folder_str.split(';'):
    # Добавляем нашу папку, формируя новую строку.
    application.asset_folder = f"{current_asset_folder_str};{custom_assets_path_str}"

print(f"INFO: Ursina asset folder: {application.asset_folder}")