import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), name))), flush=True)
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; P.rotation_y = 0
M.stepper.key('C'); G = M.growview; G.draw(M.stepper.snapshot)
G.toggle_traj(); print(G.vlabel()); shot('traj_only.png')
G.toggle_common(); print(G.vlabel()); shot('traj_common.png'); P.position = (0.5, 3.2, -3.6); P.camera_pivot.rotation_x = 40; shot('traj_common_close.png')
G.toggle_common(); G.toggle_vsurf(); P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; shot('traj_layers.png')
G.toggle_vsurf(); G.toggle_traj(); print('off ok', G.vlabel() == ''); print('SMOKE OK')
