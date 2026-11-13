export PROCS=4
S8=$HOME/spore_v5/c13/out_blind/di8_ref_100_rho35_s11.npy
cd ../b4
KP=20 MACC=5 GT=4 VEC=64 /usr/bin/time -v -o ../v1/H3.time nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh ../v1/b4_run_vec.py di8 200000 10000 3 blind11_nb200k_solo $S8 local > ../v1/H3.log 2>&1
