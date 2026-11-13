export PROCS=4 NSTART=20
S6=$HOME/spore_v5/r37/di6_ref_200_rho35_s8.npy; S8=$HOME/spore_v5/c13/out/di8_ref_100_rho35_s9.npy
R="nice -n 10 bash $HOME/spore_v5/a38/spores_2/bench/b22/runpc.sh"
cd ../b4
for v in 64; do
 if [ $v = 0 ]; then B=../b4/b4_run.py; else B=../v1/b4_run_vec.py; fi
 VEC=$([ $v = 0 ] || echo $v) /usr/bin/time -v -o ../v1/V6_$v.time $R $B di6 100000 3000 3 v6_$v $S6 base > ../v1/V6_$v.log 2>&1
 KP=20 VEC=$([ $v = 0 ] || echo $v) /usr/bin/time -v -o ../v1/V8_$v.time $R $B di8 100000 10000 3 v8_$v $S8 local > ../v1/V8_$v.log 2>&1
done
