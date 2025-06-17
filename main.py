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
        region_dict = get_partition(G, args.L, args.N)
        print_region_diction(region_dict, args.print_edges)

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

        if 'MC' in args.method:
            s_mc = MC_mp(G, epar, args.tau, init_s)
            s_mc.evolution(args.t_max, repeats=args.MC_repeats, mp_num=args.n_multiprocess)
            s_mc.save_data(args.precision,path + 'MC.npy')
        if 'DMP' in args.method:
            s = DMP(G, epar, args.tau, init_s)
            s.evolution(args.t_max)
            s.save_data(args.precision,path + 'DMP.npy')
        if 'PA' in args.method:
            s = PA(G, epar, args.tau, init_s)
            s.evolution(args.t_max)
            s.save_data(args.precision,path + 'PA.npy')
        if 'RA' in args.method:
            if not args.show_partition:
                region_dict = get_partition(G, args.L, args.N)
            for ln, partition in region_dict.items():
                l,n = ln
                label ='_'
                if l != 0:
                    label += 'L' + str(l)
                if n != 0:
                    label += 'N' + str(n)
                if l+n == 0:
                    label += '_exact'
                s = TNPA(G, epar, args.tau, init_s, partition, label)
                s.evolution(args.t_max)
                s.save_data(args.precision,path + 'TNPA' + label + '.npy')