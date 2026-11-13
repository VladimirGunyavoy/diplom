import sys, os, runpy
ENV, NB, STARTS = sys.argv[1:4]
sys.path.insert(0, os.path.expanduser('~/spore_v5/c13/spd6'))
sys.argv = ['spores_di.py', ENV, NB, '100', '3', 'wt', STARTS]
G = runpy.run_path(os.path.expanduser('~/spore_v5/c13/spd6/spores_di.py'), run_name='lib')
print('WEIGHT', G['common'].deep_weight(G['back']))
