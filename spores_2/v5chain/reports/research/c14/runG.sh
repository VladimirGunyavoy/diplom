export PROCS=4
S8=$HOME/spore_v5/c13/out/di8_ref_100_rho35_s9.npy
cd ../b4
KP=20 MACC=5 GT=4 VEC=64 /usr/bin/time -v -o ../v1/G1.time nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh ../v1/b4_run_vec.py di8 400000 3000 3 nb400k_best5 $S8 local > ../v1/G1.log 2>&1
