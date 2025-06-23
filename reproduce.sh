#!/bin/sh

epidemic='-l 0.1 -r 0.05 --patient_zeros 0 --tau 0.1'
p_MC='-n_mp 4 --MC_repeats 1000000 --n_threads 4'
methods='--method MC DMP PA ARM'

# Fig.5(b-d)

python3 main.py -s $epidemic -g "Loop star" -T 200 $methods $p_MC -L 3 4 5 6 7 8 9 --save

# Fig.6

python3 main.py -s $epidemic -g "Random tree" -n 200 --seed 1 --add_clique -T 500 $methods $p_MC -N 3 4 5 6 --save

# Fig.7

python3 main.py -s $epidemic -g "494bus" -T 300 $methods $p_MC -L 0 -N 3 6 9 --save

python3 main.py -s $epidemic -g "science" -T 200 $methods $p_MC -L 0 -N 3 6 9 --save

# Other data in Table 1  
common_args='-L 0 -N 9'

# only reproduce data in Tab.1
# python3 main.py -s $epidemic -g "Loop star" -T 200 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "Random tree" -n 200 --seed 1 --add_clique -T 500 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "494bus" -T 300 $methods $p_MC $common_args --save
# python3 main.py -s $epidemic -g "science" -T 200 $methods $p_MC $common_args --save

python3 main.py -s $epidemic -g "auth" -T 400 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "contiguous usa" -T 200 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "Karate club" -T 200 $methods $p_MC $common_args --save
python3 main.py -s $epidemic -g "dolphins" -T 300 $methods $p_MC $common_args --save
    