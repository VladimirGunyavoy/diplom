import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
import time
def shot(name, n=2):
    for _ in range(n): M.app.step(); G.poll(); G.live_update()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), name))), flush=True)
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 9, -8); P.camera_pivot.rotation_x = 45; P.rotation_y = 0
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot)
G.toggle_common() if False else None
t0 = time.time()
while (G._traj is None or G._tprog) and time.time() - t0 < 200: M.app.step(); G.poll(); time.sleep(.05)
print('traj ready', G._traj is not None, '%.0f s' % (time.time() - t0), flush=True)
G.live_toggle(); ts = []
for i in range(120):
    t = time.perf_counter(); M.app.step(); G.poll(); G.live_update(); ts.append(time.perf_counter() - t)
    if i in (20, 60): shot('live_%d.png' % i, 0)
import numpy as np; ts = np.array(ts[5:]) * 1e3; print('frame ms (step+update, 300 agents): mean %.1f p95 %.1f max %.1f -> fps %.0f' % (ts.mean(), np.percentile(ts, 95), ts.max(), 1e3 / ts.mean()), flush=True)
G.live_launch((1.2, -0.7)); shot('live_O.png', 5); print('active', G._live.active, 'agents', len(G._live.A))
G.live_toggle(); print('SMOKE OK')
