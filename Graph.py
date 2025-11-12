import networkx as nx
import numpy as np
import time
import os

def print_region(regions, print_edges):
    """
    Print nodes and (optionally) edges of each region.
    """
    all_nodes = []
    for region in regions:
        all_nodes += list(region)
        print('Nodes:', list(region))
        if print_edges:
            print('Edges:', list(region.edges()))
    print('Size of regions:', len(set(all_nodes)))

def print_region_diction(G, L, N, print_edges: bool):
    """
    Print information of each region in the partition dictionary.
    Args:
        G (nx.Graph): The graph.
        L (list): List of loop length limits.
        N (list): List of node number limits.
        print_edges (bool): Whether to print edges.
    """
    regions_dict = get_partition(G, L, N)
    for n in N:
        for l in L:
            print_region(regions_dict[(l, n)], print_edges)

def get_biconnected_subgraph(G):
    """
    Get the list of non-trivial biconnected subgraphs (size > 2).
    Args:
        G (nx.Graph): The graph.
    Returns:
        list: List of nx.Graph subgraphs.
    """
    return [nx.Graph(G.subgraph(comps)) for comps in nx.biconnected_components(G) if len(comps) > 2]

def get_max_region_size(g_name):
    """
    Get the maximum size of regions in the graph.
    """
    g, _ = graph(g_name)
    return max(len(region) for region in get_partition(g, [0], [0])[(0, 0)])

def get_l(G, e):
    """
    Get the length of the shortest loop containing edge e in G.
    """
    g = G.copy()
    g.remove_edge(*e)
    return nx.shortest_path_length(g, e[0], e[1]) + 1

def add_l(G):
    """
    Add shortest loop length attribute 'l' to each edge in G.
    """
    for e in G.edges():
        G[e[0]][e[1]]['l'] = get_l(G, e)

def get_partition(G, L, N):
    """
    Get the partition of graph G by approximate partitioning with parameter L and N.
    L: list of loop length limits; N: list of node number limits.
    Returns a dict mapping (l, n) to region lists.
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
    """
    Approximate partitioning of G by loop length l and node number n.
    """
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
    """
    Remove edges with loop length > L and return biconnected subgraphs.
    """
    g = G.copy()
    g.remove_edges_from([(e[0], e[1]) for e in g.edges() if g[e[0]][e[1]]['l'] > L])
    return get_biconnected_subgraph(g)

def get_n_loop(G, e, cutoff):
    """
    Get the number of simple loops containing edge e with length up to cutoff.
    """
    g = G.copy()
    g.remove_edge(*e)
    return sum(1 for _ in nx.all_simple_paths(g, e[0], e[1], cutoff=cutoff))

def remove_edge_N(G, N):
    """
    Recursively split G into regions with at most N nodes by removing edges.
    """
    if len(G) <= N:
        return [G]
    edges = list(G.edges())
    labels_of_edges = [G[u][v]['l'] for u, v in edges]
    Regions = []
    G_left = G.copy()
    while G_left:
        # Remove the edges by the order of shortest loop length, and the number of shortest loops it consists in.
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
            if not components:
                G_left = nx.Graph()
                break
            if len(components) > 1:
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
            else:
                edges = list(G_left.edges())
                labels_of_edges = [G_left[u][v]['l'] for u, v in edges]
                break
    if Regions:
        G_removed = G.copy()
        for region in Regions:
            G_removed.remove_edges_from(region.edges())
        for comp in get_biconnected_subgraph(G_removed):
            if len(comp) <= N:
                Regions.append(comp)
            else:
                Regions += remove_edge_N(comp, N)
    return Regions

def graph(g_name, g_parameter=None, add_clique=False):
    '''
    Generate a graph by name and parameters.
    Args:
        g_name (str): The name of the graph.
        g_parameter (list): Parameters for graph generation.
        add_clique (bool): Whether to add cliques to nodes with degree > 2.
    Returns:
        tuple: (graph, graph_name)
    '''
    if g_name == 'karate_club':
        g = nx.karate_club_graph()
        n = len(list(g))
    elif g_name == 'Loop star':
        g_name = 'loop_star'
        g = nx.empty_graph()
        g.add_edges_from([(0, l) for l in range(1, 8)])
        n_next = 8
        for l in range(3, 10):
            g.add_edges_from([(n_next + i, n_next + i + 1) for i in range(l - 2)])
            g.add_edges_from([(l - 2, n_next), (l - 2, n_next + l - 2)])
            n_next = n_next + l - 1
        n = len(g)
    elif g_name == 'Random tree':
        g_name = 'random_tree'
        [n, _, seed] = g_parameter
        g = nx.random_tree(n, seed=seed)
    else:
        # 检查networks文件夹中是否存在对应的网络文件
        network_file_path = './networks/' + g_name + '.txt'
        if not os.path.exists(network_file_path):
            print(f"错误：在networks文件夹中找不到网络文件 '{g_name}.txt'")
            print(f"文件路径：{os.path.abspath(network_file_path)}")
            print("可用的网络文件：")
            networks_dir = './networks/'
            if os.path.exists(networks_dir):
                available_files = [f for f in os.listdir(networks_dir) if f.endswith('.txt')]
                for file in available_files:
                    print(f"  - {file}")
            else:
                print("  networks文件夹不存在")
            raise FileNotFoundError(f"网络文件 '{g_name}.txt' 不存在")
        
        g = nx.read_edgelist(network_file_path, nodetype=int)
        
    if add_clique:
        g = add_cliques(g, seed)
        g_name = 'cliques_added_' + g_name
    g_name += '_n=' + str(len(g))
    return g, g_name

def add_cliques(G, seed):
    """
    Replace nodes with degree > 2 with cliques.
    Args:
        G (nx.Graph): The graph.
        seed (int): Random seed.
    Returns:
        nx.Graph: Modified graph with cliques added.
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


