#!/bin/bash
# запуск скрипта в r31/venv + mujoco/safety_gymnasium из mj-venv (в конце sys.path)
SP=$(ls -d $HOME/spore_v5/mj/venv/lib/python3*/site-packages); export PYTHONPATH=$SP
exec $HOME/spore_v5/r31/venv/bin/python -c "
import sys,runpy; sys.path=[p for p in sys.path if p!='$SP']+['$SP']; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')" "$@"
