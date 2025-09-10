# v15_5_tree_opt/src/logic/tree/tree_area_bridge.py
from __future__ import annotations
import importlib
import json
import os
from typing import Any, Dict

def _load_optimization_config() -> Dict[str, Any]:
    """Загружаем конфигурацию оптимизации из JSON файла"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'json', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('tree', {}).get('area_optimization', {})
    except Exception as e:
        print(f"[tree_area_bridge] ⚠️  Не удалось загрузить конфиг: {e}, используем значения по умолчанию")
        return {}

def _load_pairing_config() -> Dict[str, Any]:
    """Загружаем конфигурацию спаривания из JSON файла"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'json', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('tree', {}).get('pairing', {})
    except Exception as e:
        print(f"[tree_area_bridge] ⚠️  Не удалось загрузить конфиг спаривания: {e}, используем значения по умолчанию")
        return {}

def _get_current_dt_from_manager(dt_manager: Any) -> float:
    """Аккуратно достаём текущий dt из dt-manager с разными возможными API."""
    # Попытки в порядке ожидаемой вероятности
    for attr in ("current_dt", "dt", "value"):
        if hasattr(dt_manager, attr):
            return float(getattr(dt_manager, attr))
    # Методы
    for meth in ("get_current_dt", "get_dt"):
        if hasattr(dt_manager, meth) and callable(getattr(dt_manager, meth)):
            return float(getattr(dt_manager, meth)())
    raise AttributeError("[tree_area_bridge] Не удалось получить текущее dt из dt-manager")

def _find_optimizer():
    """
    Ищем реальную функцию оптимизации в папке v15_5_tree_opt/src/logic/tree.
    Поддерживаем несколько возможных имён модулей и функций.
    """
    module_candidates = [
        ".area_opt.optimize_tree_area",  # правильный путь!
        ".optimizer",
        ".area_optimizer",
        ".tree_area",
        ".optimize",
        ".",  # текущий модуль
    ]
    func_candidates = [
        "optimize_tree_area",
        "optimize_area",
        "maximize_area_with_constraints",
    ]

    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name, package="src.logic.tree")
        except Exception:
            continue
        for fn in func_candidates:
            if hasattr(mod, fn):
                return getattr(mod, fn)
    raise RuntimeError("[tree_area_bridge] Не найден оптимизатор в v15_5_tree_opt/src/logic/tree. "
                       "Добавь функцию maximize_area_with_constraints / optimize_area(...) или экспортируй её.")

def run_area_optimization(*,
                          tree: Any,
                          pairs: Any,
                          pendulum: Any,
                          dt_manager: Any,
                          dt_bounds=(0.001, 0.2),
                          optimization_method="SLSQP",
                          max_iterations=1500,
                          show=False) -> Dict[str, Any] | None:
    """
    Унифицированный вызов оптимизации площади.
    Параметры загружаются из конфигурационного файла config/json/config.json
    """
    # Загружаем конфигурацию
    config = _load_optimization_config()
    pairing_config = _load_pairing_config()
    
    # Применяем параметры из конфига (с fallback на переданные параметры)
    constraint_distance = config.get('constraint_distance', 1e-3)
    dt_bounds = tuple(config.get('dt_bounds', dt_bounds))
    optimization_method = config.get('method', optimization_method)
    max_iterations = config.get('max_iterations', max_iterations)
    
    # Параметры спаривания
    show_pairing_debug = pairing_config.get('show_debug', True)
    dt_grandchildren_factor = pairing_config.get('dt_grandchildren_factor', 0.2)
    max_pairs = pairing_config.get('max_pairs', 4)
    
    # Получаем максимальный dt из dt-manager для ограничения границ
    max_dt_from_manager = _get_current_dt_from_manager(dt_manager)
    
    # Ограничиваем границы dt максимальным значением из dt-manager
    dt_lo = max(dt_bounds[0], 0.001)  # минимум 0.001
    dt_hi = min(dt_bounds[1], max_dt_from_manager)  # максимум из dt-manager
    adjusted_dt_bounds = (dt_lo, dt_hi)
    
    optimizer = _find_optimizer()

    # Сокращенный дебаг - только результаты
    if show:
        print(f"[AreaOpt] 🔧 Оптимизация площади дерева...")
        print(f"   Метод: {optimization_method}, Итераций: {max_iterations}")
        print(f"   Пар для оптимизации: {len(pairs)}")

    # Пробуем гибкий интерфейс — большинство твоих функций принимали именно такие аргументы
    kwargs = dict(
        tree=tree,
        pairs=pairs,
        optimization_method=optimization_method,
        pendulum=pendulum,
        dt_bounds=adjusted_dt_bounds,  # скорректированные границы
        max_iterations=max_iterations,
        show=show,
        constraint_distance=constraint_distance,  # передаем constraint_distance из конфига
    )

    result = optimizer(**kwargs)  # пусть реальная функция вернёт dict / объект — просто возвращаем как есть
    
    # Проверяем, что результат не None
    if result is None:
        print(f"[AreaOpt] ❌ Оптимизатор вернул None - возможно, произошла ошибка")
        return None
    
    # ==== ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ====
    print(f"\n[AreaOpt] 📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
    print(f"   {'='*50}")
    
    # Основные метрики
    try:
        success = result.get("success", False)
        print(f"   ✅ Успех оптимизации: {success}")
        
        # Площади
        original_area = result.get("original_area", result.get("start_area", result.get("initial_area", None)))
        optimized_area = result.get("optimized_area", result.get("best_area", None))
        
        if original_area is not None and optimized_area is not None:
            delta = optimized_area - original_area
            rel = (delta / max(abs(original_area), 1e-12)) * 100.0
            print(f"   📈 Площадь: {original_area:.6e} → {optimized_area:.6e}")
            print(f"   📊 Изменение: Δ={delta:+.3e} ({rel:+.2f}%)")
        
        # Улучшение
        improvement = result.get("improvement", None)
        improvement_percent = result.get("improvement_percent", None)
        if improvement is not None:
            print(f"   🎯 Абсолютное улучшение: {improvement:.6e}")
        if improvement_percent is not None:
            print(f"   📊 Процентное улучшение: {improvement_percent:+.2f}%")
        
        # Статистика пар и констрейнтов
        pairs_count = result.get("pairs_count", None)
        constraints_count = result.get("constraints_count", None)
        if pairs_count is not None:
            print(f"   🔗 Количество пар: {pairs_count}")
        if constraints_count is not None:
            print(f"   ⚡ Количество констрейнтов: {constraints_count}")
        
        # Нарушения констрейнтов
        constraint_violations = result.get("constraint_violations", {})
        if constraint_violations:
            violations = constraint_violations.get("violations", [])
            if violations:
                print(f"   ⚠️  Нарушения констрейнтов:")
                for i, violation in enumerate(violations):
                    print(f"      Констрейнт {i+1}: {violation:.6e}")
                max_violation = max(violations)
                print(f"   🚨 МАКСИМАЛЬНОЕ НАРУШЕНИЕ: {max_violation:.6e}")
                print(f"   📏 Ожидаемый constraint_distance: 1e-3")
                if max_violation > 1e-3:
                    print(f"   ❌ КРИТИЧЕСКАЯ ОШИБКА: Нарушение больше ожидаемого constraint_distance!")
            else:
                print(f"   ✅ Нарушений констрейнтов нет")
        else:
            print(f"   ✅ Нарушений констрейнтов нет")
        
        # Дистанции по парам (из constraint_violations)
        if constraint_violations:
            print(f"   📏 Дистанции по парам:")
            pair_distances = []
            for i in range(len(constraint_violations)):
                if isinstance(constraint_violations.get(i), dict):
                    distance = constraint_violations[i].get('distance', None)
                    if distance is not None:
                        pair_distances.append(distance)
                        print(f"      Пара {i+1}: {distance:.6e}")
            
            if pair_distances:
                min_distance = min(pair_distances)
                max_distance = max(pair_distances)
                print(f"   📊 Минимальная дистанция: {min_distance:.6e}")
                print(f"   📊 Максимальная дистанция: {max_distance:.6e}")
                print(f"   📊 Средняя дистанция: {sum(pair_distances)/len(pair_distances):.6e}")
            else:
                print(f"   📏 Дистанции по парам: не найдены в constraint_violations")
        else:
            print(f"   📏 Дистанции по парам: constraint_violations пуст")
        
        # Оптимизированные dt
        optimized_dt_vector = result.get("optimized_dt_vector", None)
        if optimized_dt_vector is not None:
            print(f"   ⏱️  Оптимизированные dt:")
            print(f"      Дети (0:4): {optimized_dt_vector[:4]}")
            print(f"      Внуки (4:12): {optimized_dt_vector[4:12]}")
            
            # Проверяем, что dt не превышают max_dt из dt-manager
            dt_children = optimized_dt_vector[:4]
            dt_grandchildren = optimized_dt_vector[4:12]
            
            max_dt_children = max(abs(dt) for dt in dt_children)
            max_dt_grandchildren = max(abs(dt) for dt in dt_grandchildren)
            
            print(f"   🔍 Анализ dt относительно max_dt из dt-manager:")
            print(f"      Макс. dt детей: {max_dt_children:.6e} (max_dt: {max_dt_from_manager:.6e})")
            print(f"      Макс. dt внуков: {max_dt_grandchildren:.6e} (max_dt: {max_dt_from_manager:.6e})")
            
            if max_dt_children > max_dt_from_manager:
                print(f"      ⚠️  dt детей превышают max_dt из dt-manager!")
            if max_dt_grandchildren > max_dt_from_manager:
                print(f"      ⚠️  dt внуков превышают max_dt из dt-manager!")
        
        # Информация о методе оптимизации
        optimization_result = result.get("optimization_result", None)
        if optimization_result is not None:
            print(f"   🔧 Метод оптимизации: {optimization_method}")
            print(f"   🔄 Итераций выполнено: {getattr(optimization_result, 'nit', 'N/A')}")
            print(f"   🎯 Статус: {getattr(optimization_result, 'success', 'N/A')}")
            if hasattr(optimization_result, 'message'):
                print(f"   💬 Сообщение: {optimization_result.message}")
        
    except Exception as e:
        print(f"   ❌ Ошибка при анализе результатов: {e}")
        import traceback
        print(f"   🔍 Подробный трейсбэк:")
        traceback.print_exc()
    
    print(f"   {'='*50}\n")
    
    # ВАЖНО: Возвращаем результат только если оптимизация прошла успешно
    if result is not None and result.get("success", False):
        print(f"[AreaOpt] ✅ Возвращаем успешный результат")
        return result
    else:
        print(f"[AreaOpt] ❌ Оптимизация не удалась, возвращаем None")
        return None
