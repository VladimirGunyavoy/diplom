#!/bin/bash
# OCP-CasADi на тихой машине, по одной среде (16 процессов). Лог ~/spore_v5/c1/ocpq.log
cd ~/spore_v5/c1/spores_2/bench
for e in pend dd manip; do ~/venv-ompl/bin/python ocp_casadi.py $e --procs 16 2>&1 | tail -1 | cut -c1-300; done
echo OCPQDONE
