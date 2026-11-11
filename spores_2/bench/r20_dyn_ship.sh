# ship.sh k : копия атласа s<k>.pkl aida1 -> aida2 c8/atlas с проверкой размера
k=$1; src=/home/random/spore_v5/a36/r20car/s$k.pkl
n=$(ssh -p 2222 random@127.0.0.1 "stat -c %s $src") || exit 1
ssh -p 2222 random@127.0.0.1 "cat $src" | ssh -p 2223 aida@127.0.0.1 "cat > ~/spore_v5/c8/atlas/s$k.pkl"
m=$(ssh -p 2223 aida@127.0.0.1 "stat -c %s ~/spore_v5/c8/atlas/s$k.pkl"); [ "$n" = "$m" ] && echo "s$k ok $n" || echo "s$k SIZE MISMATCH $n $m"
