import json, glob, os
names = [('di', 'двойной интегратор 2D'), ('pend', 'маятник 2D'), ('dd', 'дифдрайв ромб 3D'), ('dd_rect', 'дифдрайв RECT 3D (слои 0/1 те же, что ромб)'), ('m2', '2 звена 4D'), ('m2g', '2 звена 4D, g=.3 (двойной маятник)'), ('m3', '3 звена 6D'), ('m3g', '3 звена 6D, g=.3'), ('m4g', '4 звена 8D, g=.3')]
print('| система | клеток/слой | покрыто пробами | ядра пересек. | гало пересек. | выход в ядро соседа | ср. r | ср. τ | τ=τmax | ош. модели | сек/слой |\n|---|---|---|---|---|---|---|---|---|---|---|')
for k, t in names:
    p = 'reports/cn_%s.json' % k
    if not os.path.exists(p): print('| %s | — не досчитано — |||||||||' % t); continue
    d = json.load(open(p)); a = d['layer0']; b = d.get('layer1', a)
    f = lambda key, fm='%.3f': fm % ((a[key] + b[key]) / 2)
    print('| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %.0f |' % (t, a['cells'], f('covered', '%.1f%%') .replace(f('covered','%.1f%%'), '%.1f%%' % (50 * (a['covered'] + b['covered']) * 100 / 100 * 1)), f('kernel_overlap'), f('halo_overlap'), f('exit_in_other_kernel'), f('r_mean'), f('tau_mean'), f('tau_max_frac'), f('err_mean', '%.4f'), (a['sec'] + b['sec']) / 2))
