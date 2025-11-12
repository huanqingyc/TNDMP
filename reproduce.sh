#!/bin/sh

epidemic='-l 0.1 -r 0.05 --patient_zeros 0 --tau 0.1'
p_MC='-n_mp 50 --MC_repeats 1000000 --n_threads 4'
methods='--method MC DMP PA TNDMP'

# # Fig.5(b-d)

# python3 main.py -s $epidemic -g "Loop star" -T 200 $methods $p_MC -N 3 4 5 6 7 8 9 --save

# # Fig.6

# python3 main.py -s $epidemic -g "Random tree" -n 200 --seed 1 --add_clique -T 500 $methods $p_MC -N 3 4 5 6 --save

# # Fig.7
common_args='-L 0 -N 0 9'

# python3 main.py -p $epidemic -g "494bus" -T 300 $methods $p_MC $common_args --save
# python3 main.py -p $epidemic -g "network_science" -T 200 $methods $p_MC $common_args --save

# # Other data in Table 1  

# methods='--method PA'
# python3 main.py -p $epidemic -g "sandi_auths" -T 400 $methods $p_MC $common_args --save
python3 main.py -p $epidemic -g "contiguous_usa" -T 200 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "karate_club" -T 200 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "dolphins" -T 300 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "interactome_pdz" -T 300 $methods $p_MC $common_args --save
    