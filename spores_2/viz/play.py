"""python viz/play.py [система] [--png DIR] [--start K] [--method ИМЯ] [--frames N] [--gif FILE]
Без аргумента — список готовых систем (viz/data/*.json). Зависимости: pygame, numpy (json — стандартный)."""
import os, sys, json, glob, importlib, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
MODS = ['draw_links', 'draw_mobile']
def systems():
    tab = {}
    for m in MODS:
        try: tab.update({k: v for k, v in importlib.import_module(m).SYSTEMS.items()})
        except ImportError: pass
    return tab
def main():
    p = argparse.ArgumentParser(); p.add_argument('system', nargs='?'); p.add_argument('--png'); p.add_argument('--start'); p.add_argument('--method'); p.add_argument('--frames', type=int, default=8); p.add_argument('--gif'); p.add_argument('--data', help='json вне viz/data (для проверок)'); a = p.parse_args()
    have = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(HERE, 'data', '*.json')))
    if not a.system: print('готовые системы:', ', '.join(have) or '(нет)'); return
    if a.system not in have and not a.data: print('нет данных viz/data/%s.json; готовые: %s' % (a.system, ', '.join(have))); sys.exit(1)
    tab = systems()
    if a.system not in tab: print('нет рисовалки для', a.system); sys.exit(1)
    data = json.load(open(a.data or os.path.join(HERE, 'data', a.system + '.json'))); import viz_core
    st = a.start
    if st is not None:
        try: st = int(st)
        except ValueError: pass
    outs = viz_core.run(a.system, tab[a.system], data, png=a.png, start=st, method=a.method, nframes=a.frames, gif=a.gif)
    if outs: print('\n'.join(outs))
if __name__ == '__main__': main()
