# Этот файл выполняется при импорте любого модуля из пакета 'utils'.
# Мы используем это, чтобы применить наш патч как можно раньше.

from .ursina_patcher import patch_ursina_loader

patch_ursina_loader()
