cd /home/rl/claude-work/projects/spore/spores_2/v7/reports/bdp/w1
export PYTHONPATH=../../..
systemd-run --user --scope -p MemoryMax=1500M -p MemorySwapMax=0 env WIN=1.0 TR=1 CHUNK=800 NEW=500 ROUNDS=8 NPROC=2 DMIN0=.05 python3 -m src.cells7.butterfly_dp ellipse e8.npz > ell2.log 2>&1
