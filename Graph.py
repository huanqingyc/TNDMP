import networkx as nx
import numpy as np
import time

def print_region(regions, print_edges):
    """
    Print nodes and (optionally) edges of each region.
    """
    all_nodes = []
    for region in regions:
        all_nodes += list(region)
        print('Nodes:', end=' ')
        print(list(region))
        if print_edges:
            print('Edges:', end=' ')
            print(list(region.edges()))
    print('Size of regions: ' + str(len(set(all_nodes))))

def print_region_diction(G, L, N, print_edges: bool):
    """
    Print information of each region in the partition dictionary.
    """
    regions_dict = get_partition(G, L, N)
    for n in N:
        for l in L:
            print_region(regions_dict[(l, n)], print_edges)

def get_biconnected_subgraph(G):
    """Get the list of non-trivial biconnected subgraphs."""
    return [nx.Graph(G.subgraph(comps)) for comps in nx.biconnected_components(G) if len(comps) > 2]

def get_l(G, e):
    """Get the length of the shortest loop containing edge e in G."""
    g = G.copy()
    g.remove_edge(*e)
    return nx.shortest_path_length(g, e[0], e[1]) + 1

def add_l(G):
    for e in G.edges():
        G[e[0]][e[1]]['l'] = get_l(G, e)

def get_partition(G, L, N):
    """
    Get the partition of graph G by approximate partitioning with parameter L and N,
    where L is the length of the longest loop in consideration, and N is the maximal number of nodes in a region.
    In which given 0 means unlimited, hence L=0 return a list, L>0 return a dict

    Parameters
    ----------
    G : nx.Graph
        The complete graph.
    L : list of int
    N : list of int

    Returns
    -------
    Regions_dict : dict of list of nx.Graph
    """
    exact_regions = get_biconnected_subgraph(G)
    for region in exact_regions:
        add_l(region)

    Regions_dict = dict()
    for l in L:
        for n in N:
            if l == 0 and n == 0:
                Regions_dict[(0, 0)] = exact_regions
            else:
                regions = []
                for region in exact_regions:
                    regions += approximation_regions(region, l, n)
                Regions_dict[(l, n)] = regions

    return Regions_dict

def approximation_regions(G, l, n):
    if n == 0:
        regions = remove_edge_L(G, l)
    else:
        if l > n or l == 0:
            l = n
        regions = []
        for region in remove_edge_L(G, l):
            regions += remove_edge_N(region, n)
    return regions

def remove_edge_L(G, L):
    g = G.copy()
    g.remove_edges_from([(e[0], e[1]) for e in g.edges() if g[e[0]][e[1]]['l'] > L])
    return get_biconnected_subgraph(g)

def get_n_loop(G, e, cutoff):
    """Get the number of simple loops containing edge e with length up to cutoff."""
    g = G.copy()
    g.remove_edge(*e)
    return sum(1 for _ in nx.all_simple_paths(g, e[0], e[1], cutoff=cutoff))

def remove_edge_N(G, N):
    if len(G) <= N:
        return [G]

    edges = list(G.edges())
    labels_of_edges = [G[u][v]['l'] for u, v in edges]

    Regions = []
    G_left = G.copy()

    while G_left:
        # Remove the edges by the order of shortest loop length, and the number of shortest loops it consists in.
        # print(G.edges())
        max_value = max(labels_of_edges)
        removing_edges = []
        for i, value in enumerate(labels_of_edges):
            if value == max_value:
                removing_edges.append(edges[i])
                labels_of_edges[i] = -1
        sorted_edges = sorted(removing_edges, key=lambda edge: get_n_loop(G_left, edge, cutoff=max_value + 1))

        for edge in sorted_edges:
            if not G_left.has_edge(edge[0], edge[1]):
                continue

            G_left.remove_edge(edge[0], edge[1])
            components = get_biconnected_subgraph(G_left)

            if not components:  # a single giant loop. With one edge removed, it degenerates to tree.
                G_left = nx.Graph()
                break

            if len(components) > 1:  # divided into multiple components
                G_left = nx.Graph()
                for comp in components:
                    if len(comp) <= N:
                        Regions.append(comp)
                    else:
                        Regions += remove_edge_N(comp, N)
                break

            G_left = components[0]

            if len(G_left) <= N:
                Regions.append(G_left)
                G_left = nx.Graph()
                break
            else:# continue removing edges
                edges = list(G_left.edges())
                labels_of_edges = [G_left[u][v]['l'] for u, v in edges]
                break

    if Regions:  # At least one region is split out, give the left part a try
        G_removed = G.copy()
        for region in Regions:
            G_removed.remove_edges_from(region.edges())

        for comp in get_biconnected_subgraph(G_removed):
            if len(comp) <= N:
                Regions.append(comp)
            else:
                Regions += remove_edge_N(comp, N)
    return Regions

def graph(g_name, g_parameter = None, add_clique = False):
    '''
    g_name: str
        The name of the graph.
    '''
    if g_name == 'contiguous usa':
        g_name = 'contiguous_usa'
        g = nx.read_edgelist('./networks/contiguous_usa.txt',nodetype=int)
        n = len(g)
    elif g_name == 'dolphins':
        g = nx.read_edgelist('./networks/dolphins.txt',nodetype=int)
        n = len(g)
    elif g_name == 'science':
        g = nx.read_edgelist('./networks/network_science.txt',nodetype=int)
        n = len(g)
    elif g_name == 'euroroad':
        g = nx.read_edgelist('./networks/euroroad.txt',nodetype=int)
        n = len(g)
    elif g_name == '494bus':
        g = nx.read_edgelist('./networks/494bus.txt',nodetype=int)
        n = len(g)
    elif g_name == 'auth':
        g = nx.read_edgelist('./networks/sandi_auths.txt',nodetype=int)
        n = len(g)
    elif g_name == 'Karate club':
        g_name = 'karate_club'
        g = nx.karate_club_graph()
        n = len(list(g))
    elif g_name == 'Loop star':
        g_name = 'loop_star'
        g = nx.empty_graph()
        g.add_edges_from([(0,l) for l in range(1,8)])
        n_next = 8
        for l in range(3,10):
            g.add_edges_from([(n_next+i,n_next+i+1) for i in range(l-2)])
            g.add_edges_from([(l-2,n_next),(l-2,n_next+l-2)])
            n_next = n_next+l-1
        n = len(g)
    elif g_name == 'Random tree':
        g_name = 'random_tree'
        [n,_,seed] = g_parameter
        g = nx.random_tree(n,seed=seed)
    
    if add_clique:
        g = add_cliques(g,seed)
        g_name = 'cliques_added_' + g_name

    g_name += '_n=' + str(len(g))

    return g, g_name

def add_cliques(G, seed):
    """
    Replace nodes with degree > 2 with cliques.
    """
    np.random.seed(seed)
    present = len(G)
    degrees = dict(G.degree())
    replace_list = [node for node, degree in degrees.items() if degree > 2]
    for node in replace_list:
        neighs = list(G[node])
        c = len(neighs)
        G.remove_edges_from([(node, neighs[i]) for i in range(1, c)])
        G.add_edges_from([(present + i, neighs[i + 1]) for i in range(c - 1)] + [(node, present + i) for i in range(c - 1)])
        G.add_edges_from([(present + i, present + j) for i in range(c - 1) for j in range(i + 1, c - 1)])
        present += (c - 1)
    return G


