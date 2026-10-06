cd /home/rl/claude-work/projects/spore/spores_2/v7/reports/bdp/w1
export PYTHONPATH=../../..
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 env WIN=1.0 TR=1 FR=1 FWD=1500 CHUNK=800 python3 -m src.cells7.butterfly_dp grow 1500 > base.log 2>&1
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 env WIN=1.0 TR=1 CHUNK=800 NEW=500 ROUNDS=8 python3 -m src.cells7.butterfly_dp ellipse base_research.npz > ell.log 2>&1
