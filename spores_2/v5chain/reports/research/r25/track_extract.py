"""research-25 (2026-10-12): трасса «Картленд» (синяя) из картинки пользователя → маска (синий, крупнейшая компонента, стрелки-дыры < 800 пкс закрыты) → знаковое расстояние до края (пкс) = gfree. Запуск: python3 track_extract.py kartland_blue.png"""
import sys, numpy as np, scipy.ndimage as nd; from PIL import Image
im = np.asarray(Image.open(sys.argv[1]).convert('RGB')).astype(int); r, g, b = im[..., 0], im[..., 1], im[..., 2]
m = (b > 120) & (b - r > 60) & (b - g > 30); lab, n = nd.label(m); m = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
hl, hn = nd.label(~m); sz = np.bincount(hl.ravel())
for k in range(1, hn + 1):
    if sz[k] < 800: m |= hl == k
sdf = np.where(m, nd.distance_transform_edt(m), -nd.distance_transform_edt(~m)); np.savez_compressed('kartland_sdf_px.npz', sdf=sdf.astype(np.float32))
