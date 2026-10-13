"""record_backup.py — бэкап-видео для презентации (№22, 2026-10-13): MP4 (H.264) + GIF каждой системы в режиме --present на медианном старте.
Запуск (Windows-venv): PYTHONUTF8=1 python record_backup.py [система ...] [--out ДИР] [--start K] [--fps 25]. Код viz не правит — вызывает play.systems()/viz_present.run(png=...) и собирает кадры."""
import os, sys, json, shutil, argparse, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, imageio.v2 as iio
import play, viz_core, viz_present as vp
ALL = ['car', 'pend', 'pend15', 'dd', 'carfree', 'dp1', 'm3d', 'm3gu']
def record(name, out, fps, start=None, tag='backup'):
    tab = play.systems(); data = json.load(open(os.path.join(HERE, 'data', name + '.json'), encoding='utf-8'))
    eps = [viz_core.Episode(e, data['params'], tab[name].get('trail')) for e in data['episodes']]
    st = start if start is not None else vp.pick_start(eps); sel = [e for e in eps if e.start == st]
    tmax = max(e.tmax for e in sel); play_s = min(max(tmax, 5.), 11.); n = int(play_s * fps); tmp = tempfile.mkdtemp(prefix='rec_')
    fr = (tab[name].get('present') or vp.run)(name, tab[name], data, png=tmp, start=st, nframes=n, hold=1.0)
    imgs = [iio.imread(f) for f in fr]; imgs += [imgs[-1]] * int(fps * 1.0)                      # пауза на последнем кадре
    base = os.path.join(out, 'anim_%s_%s_s%s' % (name, tag, st)); iio.mimsave(base + '.mp4', imgs, fps=fps, codec='libx264', quality=8, macro_block_size=None, pixelformat='yuv420p')
    iio.mimsave(base + '.gif', [im[::2, ::2] for im in imgs[::2]], duration=2. / fps, loop=0)  # GIF: 630x450, каждый 2-й кадр
    shutil.rmtree(tmp, ignore_errors=True)
    rows = ['  - %s: T %s, T/T_ref %s%s' % (vp.en_method(e.method), ('%.2f' % e.T) if e.ok else 'failed', ('%.3f' % (e.T / e.Tref)) if (e.ok and e.Tref) else '—', '' if e.ok else ' (провал)') for e in sorted(sel, key=lambda e: not vp.is_spore(e.method))]
    return '- `%s.mp4`/`.gif` → %s → старт %s → %.1f с (play %.1f с)\n%s\n' % (os.path.basename(base), name, st, play_s, play_s, '\n'.join(rows))
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('systems', nargs='*'); p.add_argument('--out', default='../../img/anim_2026-10-13'); p.add_argument('--start'); p.add_argument('--tag', default='backup'); p.add_argument('--fps', type=int, default=25); a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); lst = os.path.join(a.out, 'list.md')
    if not os.path.exists(lst): open(lst, 'w', encoding='utf-8').write('# anim_2026-10-13 — бэкап-видео (файл → система → старт → T и T/T_ref)\n\n')
    for s in (a.systems or ALL):
        try: r = record(s, a.out, a.fps, int(a.start) if a.start else None, a.tag)
        except Exception as ex: r = '- %s: ОШИБКА %r\n' % (s, ex)
        print(r); open(lst, 'a', encoding='utf-8').write(r)
