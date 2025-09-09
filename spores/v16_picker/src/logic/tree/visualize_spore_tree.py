import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def visualize_spore_tree(tree_data, title="Дерево спор", ax=None,
                         figsize=None, show_legend=True):
    """
    Упрощенная визуализация: только точки спор + линии четырехугольника.
    
    Args:
        tree_data: dict с данными дерева ('root', 'children', 'grandchildren')
                   или объект SporeTree
        title: заголовок графика
        ax: объект осей matplotlib для рисования. Если None, создается новый.
        figsize: размер полотна (ширина, высота).
        show_legend: показывать ли легенду.
    """
    if ax is None:
        if figsize is None:
            figsize = (12, 9)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Определяем, работаем мы с объектом tree или со словарем
    if isinstance(tree_data, dict):
        root = tree_data['root']
        children = tree_data['children']
        grandchildren = tree_data['grandchildren']
        _children_created = bool(children)
        _grandchildren_created = bool(grandchildren)
        # В словаре нет информации о сортировке, предполагаем, что она не нужна
        grandchildren_to_show = grandchildren
    else: # SporeTree object
        root = tree_data.root
        children = tree_data.children
        grandchildren = tree_data.grandchildren
        _children_created = tree_data._children_created
        _grandchildren_created = tree_data._grandchildren_created
        # 🔧 ИСПРАВЛЕНИЕ: Всегда показываем ВСЕ внуков для полного графа
        grandchildren_to_show = tree_data.grandchildren  # Всегда все внуки!
        
        # Добавляем информацию для отладки
        if (hasattr(tree_data, '_grandchildren_sorted') and
                tree_data._grandchildren_sorted):
            sorted_count = (len(tree_data.sorted_grandchildren)
                           if hasattr(tree_data, 'sorted_grandchildren') else 0)
            total_count = len(tree_data.grandchildren)
            print(f"🔍 tree_debug.png: Показываем ВСЕ {total_count} внуков "
                  f"(было бы {sorted_count} отсортированных)")


    # === ТОЧКИ ===
    
    # Корень
    ax.scatter(root['position'][0], root['position'][1], 
               c='#2C3E50', s=300, alpha=0.9, zorder=5) # Увеличен размер
    
    # Дети + стрелки
    if _children_created:
        child_colors = ['#5DADE2', '#A569BD', '#58D68D', '#F4D03F']
        for i, child in enumerate(children):
            ax.scatter(child['position'][0], child['position'][1],
                      c=child_colors[i], s=300, alpha=1, zorder=4, label=f'{i}') # Увеличен размер
            
            # НАПРАВЛЕНИЕ стрелки зависит от знака dt
            if child['dt'] > 0:  # forward: от корня к ребенку
                arrow_start = root['position']
                arrow_end = child['position']
            else:  # backward: от ребенка к корню
                arrow_start = child['position']
                arrow_end = root['position']
            
            # ЦВЕТ стрелки зависит от знака управления
            if child['control'] > 0:  # u_max 
                arrow_color = '#FF6B6B'
            else:  # u_min
                arrow_color = '#1ABC9C'
            
            arrow = FancyArrowPatch(
                arrow_start, arrow_end, arrowstyle='->', 
                mutation_scale=20, color=arrow_color, alpha=0.7, linewidth=3
            )
            ax.add_patch(arrow)
    
    # Внуки + стрелки
    if _grandchildren_created:
        grandchild_colors = [
            '#FF1744', '#9C27B0', '#2196F3', '#4CAF50',
            '#FF9800', '#795548', '#E91E63', '#607D8B'
        ]
        
        # 🔧 УЛУЧШЕНИЕ: Показываем все линки с улучшенной визуализацией
        grandchild_info = {}  # для отслеживания дублирующихся позиций

        for i, gc in enumerate(grandchildren_to_show):
            gc_color = grandchild_colors[gc['global_idx'] % len(grandchild_colors)]
            
            # Проверяем на дублирующиеся позиции
            pos_key = f"{gc['position'][0]:.6f}_{gc['position'][1]:.6f}"
            if pos_key in grandchild_info:
                # Смещаем дублированную позицию для видимости
                offset = len(grandchild_info[pos_key]) * 0.001
                display_pos = (gc['position'][0] + offset,
                              gc['position'][1] + offset)
                grandchild_info[pos_key].append(gc['global_idx'])
                print(f"⚠️ Дубликат позиции найден для внука "
                      f"{gc['global_idx']}, смещение на {offset}")
            else:
                display_pos = gc['position']
                grandchild_info[pos_key] = [gc['global_idx']]
            
            ax.scatter(display_pos[0], display_pos[1],
                      c=gc_color, s=400, alpha=1, zorder=3)
            
            # Показываем ID внука на точке
            ax.text(display_pos[0], display_pos[1], str(gc['global_idx']),
                    color='white', ha='center', va='center',
                    fontweight='bold', fontsize=10)

            # Создаем стрелку к родителю
            parent = children[gc['parent_idx']]
            
            if gc['dt'] > 0:
                arrow_start, arrow_end = parent['position'], display_pos
            else:
                arrow_start, arrow_end = display_pos, parent['position']
            
            # Цвет стрелки по управлению + увеличенная толщина
            arrow_color = '#FF6B6B' if gc['control'] > 0 else '#1ABC9C'
            
            arrow = FancyArrowPatch(
                arrow_start, arrow_end, arrowstyle='->', 
                mutation_scale=25, color=arrow_color,
                linewidth=4, alpha=0.8  # Увеличенная толщина и прозрачность
            )
            ax.add_patch(arrow)
    
    # === НАСТРОЙКИ ГРАФИКА ===
    
    ax.set_xlabel('θ (угол, рад)')
    ax.set_ylabel('θ̇ (скорость, рад/с)')
    
    # Добавляем статистику в заголовок
    total_links = len(children) + len(grandchildren_to_show)
    ax.set_title(f"{title} - Всего линков: {total_links} "
                 f"(дети: {len(children)}, внуки: {len(grandchildren_to_show)})")
    
    ax.grid(True, alpha=0.3)
    
    # Легенда больше не нужна
    # if show_legend and _grandchildren_created:
    #     handles, labels = ax.get_legend_handles_labels()
    #     grandchild_handles = [h for h, l in zip(handles, labels) if l and l.startswith('Внук')]
    #     grandchild_labels = [l for l in labels if l and l.startswith('Внук')]
        
    #     if grandchild_handles:
    #         legend = ax.legend(grandchild_handles, grandchild_labels, 
    #                          bbox_to_anchor=(1.05, 1), loc='upper left',
    #                          title='Внуки (индексы)', title_fontsize=20,
    #                          fontsize=16)
    #         legend.get_title().set_fontweight('bold')
    
    all_x = [root['position'][0]]
    all_y = [root['position'][1]]
    if _children_created:
        for child in children:
            all_x.append(child['position'][0])
            all_y.append(child['position'][1])
    if _grandchildren_created:
        for gc in grandchildren_to_show:
            all_x.append(gc['position'][0])
            all_y.append(gc['position'][1])
            
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_margin = max(x_range * 0.05, 1e-6)
    y_margin = max(y_range * 0.1, 0.001)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # if ax is None:
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])
    #     plt.show()