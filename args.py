import argparse

parser = argparse.ArgumentParser()

# Program parameters
group = parser.add_argument_group('program parameters')
group.add_argument(
    '-p', '--show_partition',
    action='store_true',
    help='Show partitioning results'
)
group.add_argument(
    '-s', '--simulation',
    action='store_true',
    help='Run simulation'
)
group.add_argument(
    '--precision',
    type=int,
    default=5,
    help='Precision for output data'
)
group.add_argument(
    '--print_edges',
    action='store_true',
    help='Print edges of regions or not'
)

# Graph parameters
group = parser.add_argument_group('graph parameters')
group.add_argument(
    '-g', '--graph',
    type=str,
    choices=[
        'random regular graph', 'contiguous usa', 'Karate club',
        'Loop star', 'Random tree', 'Regular tree', 'dolphins',
        'science', 'euroroad', '494bus', 'auth'
    ],
    required=True,
    help='Graph type'
)
group.add_argument(
    '-n', '--n_graph',
    type=int,
    default=100,
    help='Initial number of nodes for generated networks'
)
group.add_argument(
    '-c',
    type=int,
    default=4,
    help='Degree for generated networks'
)
group.add_argument(
    '--seed',
    type=int,
    default=1,
    help='Random seed for network generation and adding cliques'
)
group.add_argument(
    '--add_clique',
    action='store_true',
    help='Add clique to the network'
)

# Initial state mutually exclusive group
mutex_group = parser.add_mutually_exclusive_group()
mutex_group.add_argument(
    '--patient_zeros',
    type=int,
    nargs='+',
    default=[],
    help='List of initially infected node indices'
)
mutex_group.add_argument(
    '-z',
    type=float,
    default=0.0,
    help='Identical initial susceptible probability for all nodes'
)
mutex_group.add_argument(
    '-zi',
    type=float,
    nargs='+',
    default=[],
    help='Individual initial susceptible probability for each node (list of float)'
)

# Model parameters
group = parser.add_argument_group('model parameters')
group.add_argument(
    '-l', '--lamda',
    type=float,
    help='Rate of infection'
)
group.add_argument(
    '-r', '--rho',
    type=float,
    help='Rate of recovery'
)
group.add_argument(
    '-T', '--t_max',
    type=int,
    default=200,
    help='Maximum time'
)
group.add_argument(
    '--tau',
    type=float,
    default=0.1,
    help='Time step'
)

# Simulation method parameters
group = parser.add_argument_group('simulation method')
group.add_argument(
    '--method',
    nargs='+',
    choices=['ARM', 'MC', 'PA', 'DMP'],
    help='One or more methods, among ARM, MC, PA, DMP'
)
group.add_argument(
    '-L',
    type=int,
    nargs='+',
    default=[0],
    help='List of L (loop length limit for partitioning, 0 for no limit)'
)
group.add_argument(
    '-N',
    type=int,
    nargs='+',
    default=[0],
    help='List of N (max region size for partitioning, 0 for no limit)'
)
group.add_argument(
    '--MC_repeats',
    type=int,
    default=1000000,
    help='Number of repeats for MC'
)
group.add_argument(
    '-n_mp', '--n_multiprocess',
    type=int,
    default=40,
    help='Number of processes for MC'
)
group.add_argument(
    '--n_threads',
    type=int,
    default=4,
    help='Max threads for one process'
)

group = parser.add_argument_group('Save and plotting')
group.add_argument(
    '--save',
    action='store_true',
    help='Save simulation results'
)
group.add_argument(
    '--save_path',
    type=str,
    default='data',
    help='Path to save simulation results'
)

args = parser.parse_args()