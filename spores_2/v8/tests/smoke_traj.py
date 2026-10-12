import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step(); G.poll()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), name))), flush=True)
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; P.rotation_y = 0
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot)
import time
G.toggle_common(); G.toggle_traj(); print(G.vlabel()); shot('traj_computing.png')
t0 = time.time()
while G._traj is None and time.time() - t0 < 120: M.app.step(); G.poll(); time.sleep(.05)
print(G.vlabel()); P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; shot('traj_both.png'); P.position = (0.5, 3.2, -3.6); P.camera_pivot.rotation_x = 40; shot('traj_both_close.png')
G.toggle_traj(); print(G.vlabel()); shot('traj_floor.png'); G.toggle_traj(); print(G.vlabel()); shot('traj_surface.png'); G.toggle_traj(); print('off', G.vlabel()); G.toggle_common(); print('SMOKE OK')
