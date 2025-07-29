import ursina
from ursina import application, entity, mesh_importer
from pathlib import Path

# Сохраняем оригинальную, нетронутую функцию
_original_load_model = mesh_importer.load_model

def _patched_load_model(name, path=None, filetype=('.bam', '.obj', '.ursinamesh'), use_cache=True):
    """
    Переопределенная функция загрузки, которая ищет модель последовательно:
    1. В пользовательских папках, указанных через ';' в application.asset_folder.
    2. В стандартных внутренних папках Ursina.
    """
    # Если путь не указан, используем глобальный
    if path is None:
        path = application.asset_folder

    # Список всех путей для поиска
    search_paths = []

    # Добавляем пользовательские пути, если они заданы строкой
    if isinstance(path, str) and ';' in path:
        search_paths.extend([Path(p) for p in path.split(';') if p])
    elif path is not None:
        # Добавляем одиночный путь, если он есть
        search_paths.append(Path(path))

    # Добавляем внутренние папки Ursina в конец списка для поиска
    search_paths.append(application.internal_models_compressed_folder)
    search_paths.append(application.internal_models_folder)
    
    # Ищем модель в каждом из путей
    for p in search_paths:
        # Вызываем оригинальный загрузчик с каждым отдельным путем.
        # ВАЖНО: Мы не передаем 'filetype' и 'use_cache', так как внутренняя
        # реализация в этом контексте вызова их не принимает.
        model = _original_load_model(name, path=p)
        # Если модель найдена (возвращен не None), возвращаем ее.
        # Проверка на .vertices была неверной, так как возвращаться может NodePath.
        if model:
            return model
    
    # Если модель не найдена ни в одной из папок
    ursina.logger.warning(f'could not find model: "{name}" in any of the search paths: {search_paths}')
    return None


def patch_ursina_loader():
    """
    Применяет monkey-patch к загрузчику моделей в том месте, где он используется (ursina.entity),
    чтобы поддерживать несколько папок с ресурсами.
    """
    if entity.load_model.__name__ != '_patched_load_model':
        print("Patching Ursina's model loader (at ursina.entity) to support multiple asset folders...")
        entity.load_model = _patched_load_model
        print("Ursina's model loader has been patched.") 