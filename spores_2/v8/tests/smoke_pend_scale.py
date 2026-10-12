import os, sys, time
os.environ["V8_WINDOW"] = "offscreen"; os.environ['V8_MAXCELLS'] = '1'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main_pend as M
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot)
t0 = time.time()
while (G._traj is None or G._tprog) and time.time() - t0 < 250: M.app.step(); G.poll(); time.sleep(.05)
print('traj ready', G._traj is not None, flush=True)
G.tmode = True; G._draw_traj(); M.app.step()
def info(): return {n: (round(e.tk, 2), e._thick, (len(e._gn.node().get_geom(0).get_vertex_data().get_array(0).get_handle().get_data()) if e._gn is not None else 0)) for n, e in G.layers.items() if n.startswith('traj_')}
print('size', M.spore_manager.size, info())
M.spore_manager.size *= 3; G.refresh(); M.app.step(); print('size', M.spore_manager.size, info())
G.live_toggle(); 
for _ in range(60): M.app.step(); G.poll(); G.live_update()
M.spore_manager.size *= 2; G.refresh(); G.live_update(); print({n: (round(e.tk, 2), e._thick) for n, e in G.layers.items() if n.startswith('lv_')}); print('SCALE DONE')
