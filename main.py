import time
from Graph import *
from Simulation import *
from args import args

if __name__ == '__main__':
    G, g_name = graph(
        args.graph,
        [args.n_graph, args.c, args.seed],
        args.add_clique
    )

    if args.show_partition:
        print_region_diction(G, args.L, args.N, args.print_edges)

    if args.simulation:
        set_environment_variables(args.n_threads)
        n = len(G)
        # Initialize susceptible probability
        if args.z > 0:
            init_s = np.ones(n) * args.z
        elif len(args.zi) > 0:
            init_s = np.array(args.zi)
        elif len(args.patient_zeros) > 0:
            init_s = np.ones(n)
            init_s[args.patient_zeros] = 0
        else:
            print('Wrong initialization')

        epar = [args.lamda, args.rho]
        if args.save:
            ensure_dir(f'{args.save_path}/{g_name}/')
            path = os.path.join(f'{args.save_path}/{g_name}/({args.lamda},{args.rho}),({args.t_max},{args.tau})_')
            log_dir = os.path.dirname(path)
        else:
            log_dir = None

        if 'MC' in args.method:
            s_mc = MC_mp(G, epar, args.tau, init_s)
            s_mc.evolution(args.t_max, repeats=args.MC_repeats, mp_num=args.n_multiprocess, log_dir=log_dir)
            if args.save:
                s_mc.save_data(args.precision, path + 'MC.npy')
        if 'DMP' in args.method:
            s = DMP(G, epar, args.tau, init_s)
            s.evolution(args.t_max, log_dir=log_dir)
            if args.save:
                s.save_data(args.precision, path + 'DMP.npy')
        if 'PA' in args.method:
            s = PA(G, epar, args.tau, init_s)
            s.evolution(args.t_max, log_dir=log_dir)
            if args.save:
                s.save_data(args.precision, path + 'PA.npy')
        if 'TNDMP' in args.method:
            if not args.show_partition:
                t0_partition = time.process_time()
                region_dict = get_partition(G, args.L, args.N)
                partition_time = time.process_time() - t0_partition
            else:
                region_dict = get_partition(G, args.L, args.N)
                partition_time = None
            for ln, partition in region_dict.items():
                l,n = ln
                label ='_'
                if l != 0:
                    label += 'L' + str(l)
                if n != 0:
                    label += 'N' + str(n)
                if l+n == 0:
                    label += '_exact'
                s = TNDMP(G, epar, args.tau, init_s, partition, label)
                s.evolution(args.t_max, log_dir=log_dir, partition_time=partition_time)
                if args.save:
                    s.save_data(args.precision, path + 'TNDMP' + label + '.npy')