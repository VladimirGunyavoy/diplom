"""
Оптимизатор площади дерева с поддержкой валентности
====================================================

Новая версия оптимизатора, которая работает с валентной структурой:
- Принимает объект SporeValence с зафиксированными слотами
- Оптимизирует только свободные слоты
- Учитывает ограничения на расстояния между парами

Ключевые отличия от старого оптимизатора:
1. Принимает valence: SporeValence вместо простого дерева
2. Извлекает зафиксированные dt из валентности и не трогает их
3. Оптимизирует только свободные слоты
4. Возвращает результат в формате, совместимом с валентной системой

Использование:
    from src.logic.valence import SporeValence
    from src.managers.valence_manager import ValenceManager

    # Получаем валентность споры
    valence = valence_manager.analyze_spore_valence(spore_id)

    # Запускаем оптимизацию с учетом валентности
    result = optimize_tree_area_with_valence(
        tree=temp_tree,
        pairs=pairs,
        valence=valence,
        pendulum=pendulum,
        constraint_distance=1e-3,
        dt_bounds=(0.001, 0.1),
        show=True
    )
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Dict, Any, List
from .create_distance_constraints import create_distance_constraints, test_constraints
from .tree_area_evaluator import TreeAreaEvaluator
from ...valence import SporeValence


def optimize_tree_area_with_valence(
    tree,
    pairs,
    valence: SporeValence,
    pendulum,
    constraint_distance: float = 1e-5,
    dt_bounds: tuple = (0.001, 0.1),
    max_iterations: int = 1000,
    optimization_method: str = 'SLSQP',
    show: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Оптимизирует площадь дерева с учетом валентности и зафиксированных слотов.

    Главная фича: принимает SporeValence и оптимизирует только свободные слоты,
    сохраняя зафиксированные dt из занятых валентных слотов.

    Args:
        tree: исходное дерево SporeTree
        pairs: список пар [(gc_i, gc_j, meeting_info), ...] от find_optimal_pairs()
        valence: объект SporeValence с информацией о занятых/свободных слотах
        pendulum: объект маятника
        constraint_distance: максимально допустимое расстояние в парах
        dt_bounds: границы для всех dt (min_dt, max_dt)
        max_iterations: максимальное количество итераций оптимизации
        optimization_method: метод оптимизации ('SLSQP', 'L-BFGS-B', etc.)
        show: вывод отладочной информации

    Returns:
        dict: {
            'success': bool - успех оптимизации,
            'optimized_area': float - итоговая площадь,
            'original_area': float - исходная площадь,
            'improvement': float - улучшение площади,
            'optimized_dt_vector': np.array - оптимальные времена [12],
            'optimized_tree': SporeTree - оптимизированное дерево,
            'optimization_result': scipy result - полный результат scipy,
            'constraint_violations': dict - нарушения констрейнтов,
            'valence_info': dict - информация о валентности,
            'fixed_slots': dict - зафиксированные слоты,
            'optimized_slots': list - оптимизированные слоты
        }
        None при ошибке
    """

    try:
        if show:
            print("🔧 Оптимизация площади дерева С ВАЛЕНТНОСТЬЮ...")
            print("="*60)

        # ================================================================
        # ПРОВЕРКИ ВХОДНЫХ ДАННЫХ
        # ================================================================

        if not pairs:
            if show:
                print("Ошибка: Список пар пуст")
            return None

        if not hasattr(tree, 'grandchildren') or len(tree.grandchildren) == 0:
            if show:
                print("Ошибка: В дереве нет внуков")
            return None

        if valence is None:
            if show:
                print("Ошибка: Валентность не передана")
            return None

        # ================================================================
        # АНАЛИЗ ВАЛЕНТНОСТИ И ИЗВЛЕЧЕНИЕ ЗАФИКСИРОВАННЫХ DT
        # ================================================================

        if show:
            print("\n📊 АНАЛИЗ ВАЛЕНТНОСТИ:")
            print(f"   Спора: {valence.spore_id}")
            print(f"   Всего слотов: {valence.total_slots}")
            print(f"   Занято: {len(valence.get_occupied_slots())}")
            print(f"   Свободно: {len(valence.get_free_slots())}")

        # Получаем зафиксированные dt из валентности
        fixed_dt_dict = valence.get_fixed_dt_values()
        free_slot_names = valence.get_free_slot_names()

        if show:
            print(f"\n🔒 Зафиксированные слоты ({len(fixed_dt_dict)}):")
            for slot_name, dt_value in fixed_dt_dict.items():
                print(f"      {slot_name}: {dt_value:+.6f}")

            print(f"\n🔓 Слоты для оптимизации ({len(free_slot_names)}):")
            for slot_name in free_slot_names:
                print(f"      {slot_name}")

        # ================================================================
        # СОЗДАНИЕ МАСКИ ФИКСАЦИИ ДЛЯ DT_VECTOR
        # ================================================================

        # Создаем маску: True = фиксирован, False = оптимизируется
        # Вектор dt: [4 детей, 8 внуков]
        fixed_mask = np.zeros(12, dtype=bool)

        # Маппинг индексов в dt_vector на имена слотов
        slot_names_map = _create_slot_names_map()

        # Заполняем маску
        for idx, slot_name in slot_names_map.items():
            if slot_name in fixed_dt_dict:
                fixed_mask[idx] = True

        if show:
            print(f"\n🎯 Маска фиксации dt_vector:")
            print(f"   Дети (0:4): {fixed_mask[:4]}")
            print(f"   Внуки (4:12): {fixed_mask[4:12]}")
            print(f"   Зафиксировано позиций: {np.sum(fixed_mask)}")
            print(f"   Для оптимизации: {12 - np.sum(fixed_mask)}")

        # ================================================================
        # СОЗДАНИЕ JIT-ОПТИМИЗИРОВАННОГО AREA EVALUATOR
        # ================================================================

        try:
            # Убеждаемся что исходное дерево имеет детей и внуков
            if not tree._children_created:
                tree.create_children(show=False)
            if not tree._grandchildren_created:
                tree.create_grandchildren(show=False)

            # Создаем TreeAreaEvaluator
            area_evaluator = TreeAreaEvaluator(tree, show=show and False)

            # Получаем исходный dt_vector
            original_dt_children = np.abs([child['dt'] for child in tree.children])
            original_dt_grandchildren = np.abs([gc['dt'] for gc in tree.grandchildren])
            original_dt_vector = np.hstack([original_dt_children, original_dt_grandchildren])

            # Применяем зафиксированные значения из валентности
            for idx, slot_name in slot_names_map.items():
                if slot_name in fixed_dt_dict:
                    original_dt_vector[idx] = abs(fixed_dt_dict[slot_name])

            # Вычисляем исходную площадь
            original_area = area_evaluator.area(original_dt_vector)

            if show:
                print(f"\nTreeAreaEvaluator создан")
                print(f"Исходная площадь дерева: {original_area:.6f}")
                print(f"Исходный dt_vector:")
                print(f"  Дети: {original_dt_vector[:4]}")
                print(f"  Внуки: {original_dt_vector[4:12]}")

        except Exception as e:
            if show:
                print(f"Ошибка создания TreeAreaEvaluator: {e}")
            return None

        # ================================================================
        # СОЗДАНИЕ КОНСТРЕЙНТОВ
        # ================================================================

        if show:
            print("\nСоздание констрейнтов расстояний...")

        constraint_functions, constraint_info = create_distance_constraints(
            pairs, tree, pendulum, constraint_distance, show=show and False
        )

        if not constraint_functions:
            if show:
                print("Ошибка: Не удалось создать констрейнты")
            return None

        # Преобразуем в формат scipy
        scipy_constraints = [{'type': 'ineq', 'fun': func} for func in constraint_functions]

        if show:
            print(f"Создано {len(scipy_constraints)} констрейнтов")

        # ================================================================
        # ПОДГОТОВКА JIT-ОПТИМИЗИРОВАННОЙ ЦЕЛЕВОЙ ФУНКЦИИ С ФИКСАЦИЕЙ
        # ================================================================

        def objective_function(dt_vector_free):
            """
            JIT-оптимизированная целевая функция с учетом фиксации.

            Принимает только СВОБОДНЫЕ параметры, подставляет зафиксированные.

            Args:
                dt_vector_free: np.array свободных dt (размер < 12)

            Returns:
                float: -площадь дерева (для минимизации)
            """
            try:
                # Восстанавливаем полный вектор из свободных параметров
                dt_vector_full = original_dt_vector.copy()
                dt_vector_full[~fixed_mask] = np.abs(dt_vector_free)

                # Быстрое вычисление площади через JIT-оптимизированный area_evaluator
                area = area_evaluator.area(dt_vector_full)

                # Возвращаем отрицательную площадь для минимизации
                return -area

            except Exception as e:
                # При ошибке возвращаем большое положительное число (плохая площадь)
                if show:
                    print(f"Ошибка в целевой функции: {e}")
                return 1e6

        # ================================================================
        # СОЗДАНИЕ CONSTRAINT WRAPPERS ДЛЯ СВОБОДНЫХ ПАРАМЕТРОВ
        # ================================================================

        def create_constraint_wrapper(constraint_func):
            """Оборачивает constraint функцию для работы со свободными параметрами"""
            def wrapper(dt_vector_free):
                # Восстанавливаем полный вектор
                dt_vector_full = original_dt_vector.copy()
                dt_vector_full[~fixed_mask] = np.abs(dt_vector_free)
                # Вызываем исходный constraint
                return constraint_func(dt_vector_full)
            return wrapper

        # Оборачиваем все констрейнты
        scipy_constraints_wrapped = [
            {'type': 'ineq', 'fun': create_constraint_wrapper(c['fun'])}
            for c in scipy_constraints
        ]

        # ================================================================
        # НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ И ГРАНИЦЫ ДЛЯ СВОБОДНЫХ ПАРАМЕТРОВ
        # ================================================================

        # Извлекаем свободные параметры из исходного вектора
        x0_free = original_dt_vector[~fixed_mask].copy()

        # Границы: только для свободных времен
        bounds_free = [(dt_bounds[0], dt_bounds[1]) for _ in range(len(x0_free))]

        if show:
            print(f"\nПараметры оптимизации:")
            print(f"  Свободных параметров: {len(x0_free)}")
            print(f"  Зафиксированных параметров: {np.sum(fixed_mask)}")
            print(f"  Начальное приближение (свободные): {x0_free}")
            print(f"  Границы dt: {dt_bounds}")
            print(f"  Метод оптимизации: {optimization_method}")
            print(f"  Максимум итераций: {max_iterations}")

        # ================================================================
        # ТЕСТИРОВАНИЕ НАЧАЛЬНОГО ПРИБЛИЖЕНИЯ
        # ================================================================

        if show:
            print("\nТестирование начального приближения...")

            # Тестируем JIT-оптимизированную целевую функцию
            initial_objective = objective_function(x0_free)
            print(f"Начальная целевая функция: {initial_objective:.6f} (площадь: {-initial_objective:.6f})")

            # Проверяем что площади совпадают
            if abs(-initial_objective - original_area) > 1e-10:
                print(f"ВНИМАНИЕ: Несоответствие площади в evaluator!")
                print(f"Original: {original_area:.10f}, Evaluator: {-initial_objective:.10f}")
            else:
                print(f"Проверка evaluator: OK")

            # Тестируем констрейнты
            constraint_test = test_constraints(constraint_functions, original_dt_vector, constraint_info, show=show and False)
            satisfied_count = constraint_test.get('summary', {}).get('satisfied_count', 0)
            total_count = constraint_test.get('summary', {}).get('total_constraints', 0)
            print(f"Начальные констрейнты: {satisfied_count}/{total_count} выполнено")

        # ================================================================
        # ОПТИМИЗАЦИЯ
        # ================================================================

        if show:
            print(f"\nЗапуск оптимизации (только свободные параметры)...")

        # Настройки оптимизации
        options = {
            'maxiter': max_iterations,
            'ftol': 1e-9,
            'disp': show
        }

        # Запуск оптимизации
        optimization_result = minimize(
            fun=objective_function,
            x0=x0_free,
            method=optimization_method,
            bounds=bounds_free,
            constraints=scipy_constraints_wrapped,
            options=options
        )

        if show:
            print(f"Оптимизация завершена:")
            print(f"  Успех: {optimization_result.success}")
            print(f"  Сообщение: {optimization_result.message}")
            print(f"  Итераций: {optimization_result.get('nit', 'N/A')}")
            print(f"  Вызовов функции: {optimization_result.get('nfev', 'N/A')}")

        # ================================================================
        # АНАЛИЗ РЕЗУЛЬТАТА
        # ================================================================

        if not optimization_result.success:
            if show:
                print(f"Оптимизация не сошлась: {optimization_result.message}")
            # Возвращаем частичный результат
            return {
                'success': False,
                'optimization_result': optimization_result,
                'error': optimization_result.message
            }

        # Восстанавливаем полный вектор из оптимизированных свободных параметров
        optimized_dt_vector_full = original_dt_vector.copy()
        optimized_dt_vector_full[~fixed_mask] = np.abs(optimization_result.x)

        optimized_area = -optimization_result.fun  # Восстанавливаем площадь
        improvement = optimized_area - original_area

        if show:
            print(f"✅ Оптимизация завершена: площадь {original_area:.3e} → {optimized_area:.3e}")
            print(f"   Улучшение: {improvement:+.3e} ({improvement/original_area*100:+.2f}%)")

        # ================================================================
        # СОЗДАНИЕ ОПТИМИЗИРОВАННОГО ДЕРЕВА
        # ================================================================

        try:
            # Извлекаем оптимальные времена
            dt_children_opt = np.abs(optimized_dt_vector_full[0:4])
            dt_grandchildren_opt = np.abs(optimized_dt_vector_full[4:12])

            # Создаем оптимизированное дерево
            optimized_tree = tree.__class__(
                pendulum=pendulum,
                config=tree.config,
                dt_children=dt_children_opt,
                dt_grandchildren=dt_grandchildren_opt,
                show=False
            )

            if show:
                print(f"\nОптимизированное дерево создано")
                print(f"  dt_children: {[f'{dt:.6f}' for dt in dt_children_opt]}")
                print(f"  dt_grandchildren: {[f'{dt:.6f}' for dt in dt_grandchildren_opt]}")

        except Exception as e:
            if show:
                print(f"Ошибка создания оптимизированного дерева: {e}")
            optimized_tree = None

        # ================================================================
        # ПРОВЕРКА КОНСТРЕЙНТОВ В ФИНАЛЬНОМ РЕШЕНИИ
        # ================================================================

        constraint_violations = test_constraints(
            constraint_functions, optimized_dt_vector_full, constraint_info, show=show and False
        )

        if show:
            violated_count = constraint_violations.get('summary', {}).get('total_constraints', 0) - \
                           constraint_violations.get('summary', {}).get('satisfied_count', 0)

            print(f"\nПроверка финальных констрейнтов:")
            print(f"  Нарушено: {violated_count}/{len(constraint_functions)}")

            if violated_count > 0:
                print(f"  ВНИМАНИЕ: Есть нарушения констрейнтов!")

        # ================================================================
        # ФОРМИРОВАНИЕ РЕЗУЛЬТАТА
        # ================================================================

        # Информация о валентности для результата
        valence_info = {
            'spore_id': valence.spore_id,
            'total_slots': valence.total_slots,
            'occupied_slots': len(valence.get_occupied_slots()),
            'free_slots': len(valence.get_free_slots()),
            'fixed_dt_count': len(fixed_dt_dict)
        }

        # Какие слоты были оптимизированы
        optimized_slots = []
        for idx in range(12):
            if not fixed_mask[idx]:
                slot_name = slot_names_map[idx]
                optimized_slots.append({
                    'slot_name': slot_name,
                    'index': idx,
                    'original_dt': original_dt_vector[idx],
                    'optimized_dt': optimized_dt_vector_full[idx]
                })

        return {
            'success': True,
            'optimized_area': optimized_area,
            'original_area': original_area,
            'improvement': improvement,
            'improvement_percent': improvement / original_area * 100,
            'optimized_dt_vector': optimized_dt_vector_full,
            'optimized_dt_children': dt_children_opt,
            'optimized_dt_grandchildren': dt_grandchildren_opt,
            'optimized_tree': optimized_tree,
            'optimization_result': optimization_result,
            'constraint_violations': constraint_violations,
            'pairs_count': len(pairs),
            'constraints_count': len(constraint_functions),
            'valence_info': valence_info,
            'fixed_slots': fixed_dt_dict,
            'optimized_slots': optimized_slots
        }

    except Exception as e:
        if show:
            print(f"Критическая ошибка оптимизации с валентностью: {e}")
            import traceback
            traceback.print_exc()
        return None


def _create_slot_names_map() -> Dict[int, str]:
    """
    Создает маппинг индексов dt_vector на имена слотов.

    dt_vector: [4 детей, 8 внуков]
    Индексы 0-3: дети
    Индексы 4-11: внуки

    Returns:
        Словарь {индекс: имя_слота}
    """
    mapping = {}

    # Дети (4 слота) - индексы 0-3
    children_slots = [
        'forward_max',
        'forward_min',
        'backward_max',
        'backward_min'
    ]

    for i, slot_name in enumerate(children_slots):
        mapping[i] = slot_name

    # Внуки (8 слотов) - индексы 4-11
    grandchildren_slots = [
        'forward_max_forward_min',
        'forward_max_backward_min',
        'forward_min_forward_max',
        'forward_min_backward_max',
        'backward_max_forward_min',
        'backward_max_backward_min',
        'backward_min_forward_max',
        'backward_min_backward_max'
    ]

    for i, slot_name in enumerate(grandchildren_slots):
        mapping[4 + i] = slot_name

    return mapping


def print_optimization_comparison(result: Dict[str, Any], show_details: bool = True) -> None:
    """
    Выводит красивое сравнение результатов оптимизации с валентностью.

    Args:
        result: Результат от optimize_tree_area_with_valence()
        show_details: Показывать ли детальную информацию о слотах
    """
    if not result or not result.get('success'):
        print("❌ Оптимизация не удалась")
        return

    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ С ВАЛЕНТНОСТЬЮ")
    print("="*60)

    # Основные метрики
    print(f"\n🎯 ПЛОЩАДЬ:")
    print(f"   Исходная: {result['original_area']:.6e}")
    print(f"   Оптимизированная: {result['optimized_area']:.6e}")
    print(f"   Улучшение: {result['improvement']:+.6e} ({result['improvement_percent']:+.2f}%)")

    # Информация о валентности
    valence_info = result.get('valence_info', {})
    print(f"\n🔬 ВАЛЕНТНОСТЬ:")
    print(f"   Спора: {valence_info.get('spore_id')}")
    print(f"   Всего слотов: {valence_info.get('total_slots')}")
    print(f"   Занято: {valence_info.get('occupied_slots')}")
    print(f"   Свободно: {valence_info.get('free_slots')}")
    print(f"   Зафиксировано dt: {valence_info.get('fixed_dt_count')}")

    # Оптимизация
    print(f"\n⚙️  ОПТИМИЗАЦИЯ:")
    print(f"   Пар для оптимизации: {result['pairs_count']}")
    print(f"   Констрейнтов: {result['constraints_count']}")

    # Детали слотов
    if show_details:
        fixed_slots = result.get('fixed_slots', {})
        optimized_slots = result.get('optimized_slots', [])

        if fixed_slots:
            print(f"\n🔒 ЗАФИКСИРОВАННЫЕ СЛОТЫ ({len(fixed_slots)}):")
            for slot_name, dt_value in fixed_slots.items():
                print(f"   {slot_name}: {dt_value:+.6f}")

        if optimized_slots:
            print(f"\n🔓 ОПТИМИЗИРОВАННЫЕ СЛОТЫ ({len(optimized_slots)}):")
            for slot in optimized_slots:
                orig = slot['original_dt']
                opt = slot['optimized_dt']
                delta = opt - orig
                print(f"   {slot['slot_name']}: {orig:.6f} → {opt:.6f} (Δ={delta:+.6f})")

    # Статус констрейнтов
    violations = result.get('constraint_violations', {})
    summary = violations.get('summary', {})
    satisfied = summary.get('satisfied_count', 0)
    total = summary.get('total_constraints', 0)

    print(f"\n✅ КОНСТРЕЙНТЫ:")
    print(f"   Выполнено: {satisfied}/{total}")
    if satisfied < total:
        print(f"   ⚠️  Нарушено: {total - satisfied}")

    print("="*60)
