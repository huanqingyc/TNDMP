#!/bin/sh

epidemic='-l 0.1 -r 0.05 --patient_zeros 0 --tau 0.1'
p_MC='-n_mp 50 --MC_repeats 1000000 --n_threads 4'
methods='--method MC DMP PA TNDMP'

# Fig.2(a-d)

python3 main.py -s $epidemic -g "Loop star" -T 200 $methods $p_MC -N 3 4 5 6 7 8 9 --save

# Fig.2(e-h)

python3 main.py -s $epidemic -g "Random tree" -n 200 --seed 1 --add_clique -T 500 $methods $p_MC -N 3 4 5 6 --save

# Fig.3
common_args='-L 0 -N 9'

python3 main.py -s $epidemic -g "494bus" -T 300 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "sandi_auths" -T 150 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "network_science" -T 100 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "contiguous_usa" -T 100 $methods $p_MC $common_args --save