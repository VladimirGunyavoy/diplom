"""
Тестовый скрипт для ControlTreeBuilder с визуализацией через matplotlib.
Файл: spores/v14_back/scripts/tests/test_control_tree.py

Для запуска из корня проекта:
    python scripts/tests/test_control_tree.py
    
Для Jupyter notebook - скопировать содержимое в ячейки.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
# import networkx as nx  # Не нужен для новой структуры

# Добавляем путь к src для импортов

script_dir = os.path.dirname(os.path.abspath(__file__))
# Поднимаемся от spores/9/scripts/run до spores/9
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импортируем необходимые модули
from src.logic.pendulum import PendulumSystem
from src.logic.control_tree import ControlTreeBuilder


def visualize_tree(tree_data, title="Control Tree Visualization"):
    """
    Визуализирует 8 траекторий управления с помощью matplotlib.
    
    Args:
        tree_data: Словарь с данными от ControlTreeBuilder.build_tree()
        title: Заголовок графика
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    trajectories = tree_data['trajectories']
    convergence_info = tree_data['convergence_info']
    
    # === Левый график: Фазовое пространство ===
    ax1.set_title(f"{title}\nPhase Space - 8 Trajectories", fontsize=14, fontweight='bold')
    ax1.set_xlabel('θ (angle)', fontsize=12)
    ax1.set_ylabel('θ̇ (angular velocity)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    
    # Цветовая схема для групп схождения
    group_colors = ['red', 'blue', 'green', 'purple']
    
    # Рисуем начальную точку
    initial_point = trajectories[0].points[0]
    ax1.scatter(initial_point[0], initial_point[1], 
               c='black', s=200, marker='o',
               edgecolors='white', linewidth=2,
               label='Start', zorder=5)
    
    # Рисуем каждую траекторию
    for traj in trajectories:
        color = group_colors[traj.convergence_group]
        
        # Рисуем первый сегмент (начало → промежуточная)
        p0, p1 = traj.points[0], traj.points[1]
        u1, dt1 = traj.sequence[0]
        
        # Стиль линии: сплошная для dt>0, пунктир для dt<0
        linestyle = '-' if dt1 > 0 else '--'
        # Толщина: толще для u>0
        linewidth = 2 if u1 > 0 else 1
        
        ax1.annotate('', xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle='->', 
                                  color=color, alpha=0.5,
                                  lw=linewidth, ls=linestyle))
        
        # Промежуточная точка
        ax1.scatter(p1[0], p1[1], c=color, s=50, alpha=0.6, marker='o')
        
        # Рисуем второй сегмент (промежуточная → конечная)
        p2 = traj.points[2]
        u2, dt2 = traj.sequence[1]
        
        linestyle = '-' if dt2 > 0 else '--'
        linewidth = 2 if u2 > 0 else 1
        
        ax1.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', 
                                  color=color, alpha=0.7,
                                  lw=linewidth, ls=linestyle))
    
    # Отмечаем точки схождения
    for group in convergence_info['groups']:
        mean_point = group['mean_point']
        converged = group['converged']
        
        # Конечная точка группы
        marker = '*' if converged else 's'
        size = 150 if converged else 100
        
        ax1.scatter(mean_point[0], mean_point[1], 
                   c=group_colors[group['group_id']], s=size,
                   marker=marker, edgecolors='black', linewidth=1.5,
                   label=f"Group {group['group_id']}", zorder=4)
        
        # Круг вокруг точки схождения
        if converged:
            circle = plt.Circle(mean_point, 0.02, fill=False,
                              edgecolor=group_colors[group['group_id']], 
                              linewidth=2, linestyle=':', alpha=0.8)
            ax1.add_patch(circle)
    
    ax1.legend(loc='best', fontsize=9)
    
    # === Правый график: Диаграмма схождения ===
    ax2.set_title("Convergence Analysis", fontsize=14, fontweight='bold')
    ax2.set_xlabel('Convergence Group', fontsize=12)
    ax2.set_ylabel('Max Deviation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # График отклонений для каждой группы
    group_ids = []
    deviations = []
    colors = []
    
    for group in convergence_info['groups']:
        group_ids.append(group['group_id'])
        deviations.append(group['max_deviation'])
        colors.append(group_colors[group['group_id']])
    
    bars = ax2.bar(group_ids, deviations, color=colors, alpha=0.7, edgecolor='black')
    
    # Линия порога схождения
    ax2.axhline(y=1e-4, color='red', linestyle='--', alpha=0.5, label='Convergence threshold')
    
    # Подписи на барах
    for i, (bar, group) in enumerate(zip(bars, convergence_info['groups'])):
        height = bar.get_height()
        status = '✓' if group['converged'] else '✗'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{status}\n{height:.2e}',
                ha='center', va='bottom', fontsize=9)
    
    ax2.set_xticks(group_ids)
    ax2.set_xticklabels([f'Group {i}' for i in group_ids])
    ax2.set_yscale('log')
    ax2.legend()
    
    # Добавляем текстовую информацию
    info_text = f"Converged: {convergence_info['num_converged']}/{convergence_info['total_groups']}"
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def test_basic_tree():
    """Базовый тест построения траекторий с визуализацией."""
    
    print("=" * 60)
    print("🌳 ТЕСТ ПОСТРОЕНИЯ 8 ТРАЕКТОРИЙ УПРАВЛЕНИЯ")
    print("=" * 60)
    
    # 1. Создаем систему маятника
    pendulum_params = {
        'g': 9.81,
        'l': 2.0,
        'm': 1.0,
        'damping': 0.1,
        'max_control': 2.0
    }
    pendulum = PendulumSystem(**pendulum_params)
    print(f"✅ Создана система маятника с параметрами:")
    print(f"   max_control = ±{pendulum.max_control}")
    
    # 2. Создаем построитель дерева с дефолтными dt
    builder = ControlTreeBuilder(pendulum)
    print(f"\n✅ Создан ControlTreeBuilder")
    print(f"   dt_vector = {builder.dt_vector}")
    
    # 3. Строим траектории из начальной точки
    initial_position = np.array([0.5, 0.0])  # Начальный угол 0.5 рад, нулевая скорость
    print(f"\n🚀 Строим 8 траекторий из точки: ({initial_position[0]:.2f}, {initial_position[1]:.2f})")
    print("-" * 40)
    
    tree_data = builder.build_tree(initial_position)
    
    # 4. Визуализируем результат
    print("\n📊 Визуализация траекторий...")
    fig = visualize_tree(tree_data, title="8 Control Trajectories (Default dt = 0.1)")
    plt.show()
    
    # 5. Выводим сводку
    builder.print_summary()
    
    return builder, tree_data


def test_optimized_dt():
    """Тест с оптимизированными временными шагами для лучшего схождения."""
    
    print("\n" + "=" * 60)
    print("🔧 ТЕСТ С НАСТРОЕННЫМИ ВРЕМЕННЫМИ ШАГАМИ")
    print("=" * 60)
    
    # Создаем систему
    pendulum = PendulumSystem(damping=0.1, max_control=2.0)
    
    # Создаем построитель с кастомными dt для лучшего схождения
    # Идея: малые одинаковые dt должны дать хорошее схождение
    custom_dt = np.array([
        0.05, 0.05,  # Группа 0: forward-forward
        0.05, 0.05,  # Группа 1: forward-backward
        0.05, 0.05,  # Группа 2: backward-backward  
        0.05, 0.05   # Группа 3: mixed
    ])
    
    builder = ControlTreeBuilder(pendulum, dt_vector=custom_dt)
    print(f"✅ Создан ControlTreeBuilder с кастомными dt:")
    print(f"   dt_vector = {builder.dt_vector}")
    
    # Строим дерево
    initial_position = np.array([0.3, -0.2])
    print(f"\n🚀 Строим 8 траекторий из точки: ({initial_position[0]:.2f}, {initial_position[1]:.2f})")
    print("-" * 40)
    
    tree_data = builder.build_tree(initial_position)
    
    # Визуализируем
    print("\n📊 Визуализация траекторий с настроенными dt...")
    fig = visualize_tree(tree_data, title="Trajectories with Optimized dt (all = 0.05)")
    plt.show()
    
    return builder, tree_data


def analyze_convergence_quality(builder, tree_data):
    """Анализ качества схождения траекторий."""
    
    print("\n" + "=" * 60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ СХОЖДЕНИЯ")
    print("=" * 60)
    
    convergence_groups = tree_data['convergence_info']['groups']
    
    if not convergence_groups:
        print("⚠️ Нет групп схождения")
        return
    
    # Для каждой группы анализируем качество
    for group in convergence_groups:
        print(f"\n📍 Группа схождения {group['group_id']}:")
        print(f"   Траектории: {group['trajectories']}")
        
        # Показываем последовательности управлений для траекторий в группе
        for traj_id in group['trajectories']:
            traj = builder.get_trajectory_by_id(traj_id)
            if traj:
                seq_str = " → ".join([f"({u:+.1f}, {dt:+.3f})" for u, dt in traj.sequence])
                print(f"   Траектория {traj_id}: {seq_str}")
        
        # Анализ точности схождения
        print(f"   Средняя точка: ({group['mean_point'][0]:.6f}, {group['mean_point'][1]:.6f})")
        print(f"   Макс. отклонение: {group['max_deviation']:.2e}")
        
        # Визуальный индикатор качества
        if group['max_deviation'] < 1e-6:
            quality = "🟢 Отличное"
        elif group['max_deviation'] < 1e-4:
            quality = "🟡 Хорошее"
        elif group['max_deviation'] < 1e-2:
            quality = "🟠 Среднее"
        else:
            quality = "🔴 Плохое"
        
        print(f"   Качество схождения: {quality}")
    
    # Общая статистика
    total_deviation = sum(g['max_deviation'] for g in convergence_groups)
    avg_deviation = total_deviation / len(convergence_groups)
    
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Групп сошлось: {tree_data['convergence_info']['num_converged']}/{tree_data['convergence_info']['total_groups']}")
    print(f"   Среднее отклонение: {avg_deviation:.2e}")
    print(f"   dt_vector: {tree_data['dt_vector']}")


def main():
    """Основная функция для запуска всех тестов."""
    
    # Тест 1: Базовое дерево
    builder1, tree1 = test_basic_tree()
    
    # Тест 2: Дерево с настроенными dt
    builder2, tree2 = test_optimized_dt()
    
    # Анализ схождения
    analyze_convergence_quality(builder1, tree1)
    analyze_convergence_quality(builder2, tree2)
    
    # Интерактивный режим для экспериментов
    print("\n" + "=" * 60)
    print("💡 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 60)
    print("Объекты доступны для экспериментов:")
    print("  - builder1, tree1: траектории с дефолтными dt")
    print("  - builder2, tree2: траектории с одинаковыми малыми dt")
    print("\nПример обновления dt:")
    print("  new_dt = np.array([0.1, 0.05, 0.15, 0.1, 0.08, 0.12, 0.1, 0.09])")
    print("  builder1.update_dt_vector(new_dt)")
    print("  tree_new = builder1.build_tree(np.array([0.5, 0.0]))")
    print("  fig = visualize_tree(tree_new)")
    print("\nДля анализа конкретной траектории:")
    print("  traj = builder1.get_trajectory_by_id(0)")
    print("  print(traj)")
    
    return builder1, tree1, builder2, tree2


if __name__ == "__main__":
    # Запускаем тесты
    builder1, tree1, builder2, tree2 = main()
    
    # Показываем все графики
    plt.show()