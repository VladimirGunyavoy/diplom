import os, sys, time
os.environ["V8_WINDOW"] = "offscreen"; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main_pend as M, numpy as np
from panda3d.core import Filename
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot); t0 = time.time()
while (G._traj is None or G._tprog) and time.time() - t0 < 250: M.app.step(); G.poll(); time.sleep(.05)
G.tmode = True; G._draw_traj()
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 16, -8); P.camera_pivot.rotation_x = 55
for _ in range(4): M.app.step()
print('shot', ursina.application.base.win.save_screenshot(Filename.from_os_specific(r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports\pend_traj.png')))
res, m = G._traj; st = np.asarray(res['status'], object); print('n', len(st), 'reached', int((st == 'reached').sum()), 'path lens', [len(p) for p in res['paths']][:40]); print('TRAJ DONE')
