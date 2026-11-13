"""блок 4 (д): раскладка веса обратного дерева m8 (deep_weight) по массивам — на малом NB той же конструкции (периодика 4 углов ⇒ KD-дерево по 3^4=81 образам)."""
import sys, os, runpy, numpy as np
NBn = sys.argv[1]; sys.argv = ['spores_8d.py', 'm8', NBn, '200', '3', 'w']
G = runpy.run_path(os.path.expanduser('~/spore_v5/a31/spores_2/bench/spores_8d.py'), run_name='lib')
back, common, S = G['back'], G['common'], G['S']; Pb, Gb, parb, layb, seqs, tree = back; n = len(Pb)
tot = common.deep_weight(back); print('n=%d, deep_weight(back): elems %d (%.1f на узел), bytes %d' % (n, tot['elems'], tot['elems'] / n, tot['bytes']))
for nm, o in (('Pb', Pb), ('Gb', Gb), ('parb', parb), ('layb', layb), ('seqs', seqs), ('tree', tree)): w = common.deep_weight(o); print('  %-5s elems %9d (%.1f/узел) bytes %d' % (nm, w['elems'], w['elems'] / n, w['bytes']))
print('  tree.data %s dtype %s, tree.indices %s; len(images)=%d (=%d·n)' % (tree.data.shape, tree.data.dtype, tree.indices.shape, len(tree.data), len(tree.data) // n))
sl = np.array([len(s) for s in seqs]); print('  seqs: средняя длина %.1f, макс %d, сумма %d (python-списки: deep_weight их НЕ считает; ~8 Б/ссылка + объекты int)' % (sl.mean(), sl.max(), sl.sum()))
print('  python-объекты вне deep_weight: Gb list[float] %d шт, parb/layb list[int], seqs — суммарно ≈ %.1f МБ (оценка 8 Б на элемент + 28–32 Б на не-малый int/float)' % (n, (n * 3 + sl.sum()) * 8 / 1e6))
