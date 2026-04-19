import numpy as np
from .find_optimal_pairs import find_optimal_pairs
from .extract_optimal_times_from_pairs import extract_optimal_times_from_pairs
from ..area_opt.optimize_tree_area import optimize_tree_area


def create_tree_from_pairs(tree, pendulum, config, show=False):
    """
    Создает оптимизированное дерево из найденных пар внуков.
    
    Выполняет полный пайплайн:
    1. Находит оптимальные пары внуков
    2. Извлекает оптимальные времена из пар  
    3. Создает новое дерево с этими временами
    
    Args:
        tree: исходное дерево SporeTree
        pendulum: объект маятника
        config: конфигурация для нового дерева
        show: bool - вывод промежуточных результатов
        
    Returns:
        dict: {
            'success': bool - успех создания,
            'optimized_tree': SporeTree - новое дерево или None,
            'pairs': list - найденные пары,
            'optimization_results': dict - результаты извлечения времен,
            'stats': dict - статистика оптимизации
        }
        None при критической ошибке
    """
    
    try:
        if show:
            print("СОЗДАНИЕ ДЕРЕВА ИЗ НАЙДЕННЫХ ПАР")
            print("="*50)
        
        # ================================================================
        # ЭТАП 1: ПОИСК ОПТИМАЛЬНЫХ ПАР
        # ================================================================
        
        if show:
            print("Поиск оптимальных пар...")
        
        pairs = find_optimal_pairs(tree, show=show and False)  # Детальный дебаг только при необходимости
        
        if pairs is None:
            if show:
                print("ОШИБКА: Не удалось найти оптимальные пары!")
                print("Возможные причины:")
                print("- Нет сближающихся пар в дереве")
                print("- Оптимизация пар не сошлась") 
                print("- Пары не прошли distance constraint")
            return {
                'success': False,
                'error': 'pairs_not_found',
                'optimized_tree': None,
                'pairs': None,
                'optimization_results': None,
                'stats': None
            }
        
        if show:
            print(f"Найдено {len(pairs)} оптимальных пар")
            print("Найденные пары:")
            for i, (gc_i, gc_j, meeting_info) in enumerate(pairs):
                gc_i_info = tree.grandchildren[gc_i]
                gc_j_info = tree.grandchildren[gc_j]
                direction_i = "F" if gc_i_info['dt'] > 0 else "B"
                direction_j = "F" if gc_j_info['dt'] > 0 else "B"
                print(f"  {i+1}. gc_{gc_i}({direction_i}) ↔ gc_{gc_j}({direction_j}): "
                      f"расст={meeting_info['distance']:.6f}, t={meeting_info['meeting_time']:.6f}с")
        
        # ================================================================
        # ЭТАП 2: ИЗВЛЕЧЕНИЕ ОПТИМАЛЬНЫХ ВРЕМЕН
        # ================================================================
        
        if show:
            print(f"\nИзвлечение оптимальных времен из {len(pairs)} пар...")
        
        optimization_results = extract_optimal_times_from_pairs(pairs, tree, show=show and False)
        
        if optimization_results is None:
            if show:
                print("ОШИБКА: Не удалось извлечь оптимальные времена!")
            return {
                'success': False,
                'error': 'times_extraction_failed',
                'optimized_tree': None,
                'pairs': pairs,
                'optimization_results': None,
                'stats': None
            }
        
        if show:
            stats = optimization_results['stats']
            print(f"Успешно извлечены времена:")
            print(f"  Спаренных внуков: {stats['paired_count']}/{stats['total_grandchildren']}")
            print(f"  Изменено времен: {stats['changed_count']}/{stats['total_grandchildren']}")
            print(f"  Нарушений направления времени: {stats['direction_violations']}")
            if stats['changed_count'] > 0:
                print(f"  Среднее изменение |новый|/|старый|: {stats['avg_change_ratio']:.3f}")
        
        # ================================================================
        # ЭТАП 3: СОЗДАНИЕ НОВОГО ДЕРЕВА
        # ================================================================
        
        if show:
            print(f"\nСоздание оптимизированного дерева...")
        
        try:
            # ВАЖНО: Берем модули времен (SporeTree сам определяет знак времени)
            dt_children_abs = np.abs(optimization_results['dt_children'])
            dt_grandchildren_abs = np.abs(optimization_results['dt_grandchildren'])
            
            if show:
                print(f"Используем модули времен:")
                print(f"  dt_children: {[f'{dt:.6f}' for dt in dt_children_abs]}")
                print(f"  dt_grandchildren: {[f'{dt:.6f}' for dt in dt_grandchildren_abs]}")
            
            # Создаем оптимизированное дерево
            optimized_tree = tree.__class__(
                pendulum=pendulum,
                config=config,
                dt_children=dt_children_abs,
                dt_grandchildren=dt_grandchildren_abs,
                show=False  # Без дебага при создании
            )
            
            if show:
                print(f"\nОптимизированное дерево создано:")
                print(f"  Детей: {len(optimized_tree.children)}")
                print(f"  Внуков: {len(optimized_tree.grandchildren)}")
                print(f"  Дерево готово к использованию!")
            
            # ===== (НОВОЕ) ДОП. ОПТИМИЗАЦИЯ ПО ПЛОЩАДИ С ОГРАНИЧЕНИЯМИ ПАР =====
            area_cfg = {}
            if isinstance(config, dict):
                # ожидаем (если есть):
                # config['tree']['area_optimization'] = {
                #   "enabled": true, "constraint_distance": 1e-4,
                #   "dt_bounds": [0.001, 0.2], "max_iterations": 1500, "method": "SLSQP"
                # }
                area_cfg = config.get('tree', {}).get('area_optimization', {}) or {}

            area_result = None
            if area_cfg.get('enabled', False):
                constraint_distance = float(area_cfg.get('constraint_distance', 1e-4))
                dt_bounds = tuple(area_cfg.get('dt_bounds', (0.001, 0.2)))
                max_iterations = int(area_cfg.get('max_iterations', 1500))
                method = str(area_cfg.get('method', 'SLSQP'))

                if show:
                    print("\n🔷 Запуск оптимизации площади дерева с констрейнтами:")
                    print(f"   constraint_distance={constraint_distance}, dt_bounds={dt_bounds}, "
                          f"max_iterations={max_iterations}, method={method}")

                area_result = optimize_tree_area(
                    optimized_tree,         # текущее дерево из пар
                    pairs,                  # те самые пары (gc_i, gc_j, meeting_info)
                    pendulum,               # маятник
                    constraint_distance=constraint_distance,
                    dt_bounds=dt_bounds,
                    max_iterations=max_iterations,
                    optimization_method=method,
                    show=bool(show)
                )

                if area_result and area_result.get('success'):
                    if show:
                        print("   ✅ Площадь оптимизирована, подменяем дерево на area-оптимизированное")
                    optimized_tree = area_result['optimized_tree']
                else:
                    if show:
                        print(f"   ⚠️ Оптимизация площади не удалась или выключена: {area_result}")
            
        except Exception as e:
            if show:
                print(f"ОШИБКА создания дерева: {e}")
            return {
                'success': False,
                'error': f'tree_creation_failed: {e}',
                'optimized_tree': None,
                'pairs': pairs,
                'optimization_results': optimization_results,
                'stats': optimization_results['stats']
            }
        
        # ================================================================
        # ФИНАЛЬНАЯ СТАТИСТИКА
        # ================================================================
        
        stats_summary = {
            'pairs_found': len(pairs),
            'paired_grandchildren': optimization_results['stats']['paired_count'],
            'total_grandchildren': optimization_results['stats']['total_grandchildren'],
            'times_changed': optimization_results['stats']['changed_count'],
            'direction_violations': optimization_results['stats']['direction_violations'],
            'avg_change_ratio': optimization_results['stats']['avg_change_ratio'],
            'tree_created': True
        }
        
        area_summary = None
        if area_cfg.get('enabled', False):
            if area_result and area_result.get('success'):
                area_summary = {
                    'success': True,
                    'original_area': area_result['original_area'],
                    'optimized_area': area_result['optimized_area'],
                    'area_improvement': area_result['area_improvement'],
                    'area_improvement_percent': area_result['area_improvement_percent'],
                    'constraints_satisfied': area_result['constraints_satisfied'],
                    'constraints_count': area_result['constraints_count'],
                    'pairs_count': area_result['pairs_count'],
                }
            else:
                area_summary = {'success': False}
        
        if show:
            print(f"\n" + "="*50)
            print(f"ДЕРЕВО ИЗ ПАР СОЗДАНО УСПЕШНО!")
            print(f"="*50)
            print(f"Статистика:")
            print(f"  Найдено пар: {stats_summary['pairs_found']}")
            print(f"  Спарено внуков: {stats_summary['paired_grandchildren']}/{stats_summary['total_grandchildren']}")
            print(f"  Изменено времен: {stats_summary['times_changed']}")
            print(f"  Готово к дальнейшей оптимизации площади")
        
        return {
            'success': True,
            'optimized_tree': optimized_tree,
            'pairs': pairs,
            'optimization_results': optimization_results,
            'stats': stats_summary,
            'area_optimization': area_summary
        }
        
    except Exception as e:
        if show:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        return None


def compare_trees_from_pairs(original_tree, result, show=False):
    """
    Сравнивает исходное дерево с деревом созданным из пар.
    
    Args:
        original_tree: исходное дерево
        result: результат от create_tree_from_pairs()
        show: вывод сравнения
        
    Returns:
        dict: статистика сравнения или None
    """
    
    if not result or not result['success']:
        if show:
            print("Создание дерева из пар не удалось, сравнение невозможно")
        return None
    
    try:
        optimized_tree = result['optimized_tree']
        stats = result['stats']
        
        # Можно добавить сравнение площадей если нужно
        comparison = {
            'pairs_used': stats['pairs_found'],
            'grandchildren_optimized': stats['paired_grandchildren'],
            'grandchildren_total': stats['total_grandchildren'],
            'optimization_coverage': stats['paired_grandchildren'] / stats['total_grandchildren'] * 100,
            'times_changed': stats['times_changed'],
            'direction_violations': stats['direction_violations']
        }
        
        if show:
            print("СРАВНЕНИЕ ДЕРЕВЬЕВ")
            print("="*30)
            print(f"Использовано пар: {comparison['pairs_used']}")
            print(f"Оптимизировано внуков: {comparison['grandchildren_optimized']}/{comparison['grandchildren_total']}")
            print(f"Покрытие оптимизацией: {comparison['optimization_coverage']:.1f}%")
            print(f"Изменено времен: {comparison['times_changed']}")
            print(f"Нарушений направления времени: {comparison['direction_violations']}")
            
            if comparison['direction_violations'] == 0:
                print("Все направления времени сохранены")
            else:
                print("ВНИМАНИЕ: Есть нарушения направления времени!")
        
        return comparison
        
    except Exception as e:
        if show:
            print(f"Ошибка сравнения: {e}")
        return None