import sys, os, cProfile, pstats
sys.argv = ['x', 'm8', os.environ.get('NBP', '2000'), os.environ.get('NFP', '1500'), '3', '_', 'wlim']
__file__ = os.path.abspath('spores_8d.py'); exec(open('spores_8d.py').read().split("if __name__ == '__main__':")[0])
cProfile.run('r = work(int(os.environ.get("QI", 4)))', 'p8.out'); print(r)
pstats.Stats('p8.out').sort_stats('tottime').print_stats(14)
