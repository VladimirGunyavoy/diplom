#!/bin/bash
cd ~/spore_v5/c1/spores_2/bench; P=~/venv-ompl/bin/python
for k in A B; do $P manip_goals.py ref $k --procs 16 2>&1 | tail -2 | cut -c1-300; done
echo MGREFDONE
