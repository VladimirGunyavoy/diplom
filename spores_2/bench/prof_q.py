import sys, os, cProfile, pstats
sys.argv = ['x', os.environ.get('ENVP', 'm3d'), '12000', '3000', '3'] + (['_', 'wlim'] if os.environ.get('WLP') else [])
__file__ = os.path.abspath('spores_6d.py'); exec(open('spores_6d.py').read().split("if __name__ == '__main__':")[0])
i = int(os.environ.get('QI', 4)); cProfile.run('r = work(i)', 'prof.out'); print(r)
pstats.Stats('prof.out').sort_stats('cumtime').print_stats(22)
