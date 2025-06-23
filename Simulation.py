import multiprocessing as mp
import os
import networkx as nx
import numpy as np

def ensure_dir(filename):
    """
    Ensure that the directory for the given filename exists.
    """
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

def set_environment_variables(threads=4):
    """
    Set environment variables for controlling the number of threads in numpy/scipy.
    """
    threads = str(threads)
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads
    os.environ["OMP_NUM_THREADS"] = threads

class Epidemic:
    def __init__(self, g, epar, tau, init_s):
        """
        Base class for epidemic simulation.
        g: networkx graph
        epar: epidemic parameters (infection, recovery)
        tau: time step
        init_s: initial susceptible probability array
        """
        self.G = nx.Graph(g)
        self.nodes = list(self.G)
        self.tau = tau
        self.spt = int(1 / tau)  # steps per unit time
        self.d = 3
        self.n = len(g)
        self.marginal_all = []
        self.marginal = np.zeros((self.n, self.d))
        self.marginal[:, 0] = init_s
        self.marginal[:, 1] = 1 - init_s
        self.algorithm_label = ''

        self.epar = np.array(epar) * tau
        self.l = self.epar[0]
        self.r = self.epar[1]

    def save_data(self, precision, fullname):
        """
        Save the marginal_all data as a numpy .npy file with the specified precision.
        """
        marginal_all = np.round(self.marginal_all, precision)
        np.save(fullname, marginal_all)

class MC_mp(Epidemic):
    def __init__(self, g, epar, tau, init_s):
        """
        Monte Carlo simulation with multiprocessing.
        """
        super().__init__(g, epar, tau, init_s)
        self.algorithm = 'MC'

    def evolution(self, t, repeats, mp_num=10):
        """
        Run the MC simulation to time t, using multiprocessing.
        """
        self.t = t
        self.marginal_all = np.zeros([t + 1, self.n, self.d])

        adjacency_matrix = nx.to_numpy_array(self.G, list(range(self.n)))
        adjacency_matrix[adjacency_matrix != 0] = 1

        single_repeat = repeats // mp_num

        with mp.Pool(mp_num) as pool:
            arg_list = [
                [single_repeat, self.epar, self.n, self.d, t, seed, self.spt, self.marginal, adjacency_matrix]
                for seed in range(mp_num)
            ]
            results = pool.map(MC, arg_list)
        for result in results:
            self.marginal_all += result
        self.marginal_all /= repeats

class DMP(Epidemic):
    def __init__(self, g, epar, tau, init_i):
        """
        Dynamic Message Passing (DMP) simulation.
        """
        super().__init__(g, epar, tau, init_i)
        self.marginal_all.append(self.marginal.copy())
        self.edges = list(self.G.edges())

        self.algorithm = 'DMP'
        self.H = np.ones((self.n, self.n))
        self.z = self.marginal[:, 0].copy()

    def evolution(self, t):
        """
        Run the DMP simulation to time t.
        """
        self.t = t
        self.pt = 0
        for _ in range(t):
            for __ in range(self.spt):
                self.step()
                self.pt += self.tau
            self.marginal_all.append(self.marginal.copy())
        self.marginal_all = np.array(self.marginal_all)

    def step(self):
        """
        Perform one DMP step.
        """
        new_H = self.H.copy()

        for [a, b] in self.edges:
            # Update messages
            new_H[a, b] += -(self.r + self.l) * self.H[a, b] + self.r + self.l * self.z[a] * np.prod(self.H[:, a]) / self.H[b, a]
            new_H[b, a] += -(self.r + self.l) * self.H[b, a] + self.r + self.l * self.z[b] * np.prod(self.H[:, b]) / self.H[a, b]
        self.H = new_H

        self.marginal[:, 0] = self.z * np.prod(self.H, axis=0)
        self.marginal[:, 2] += self.r * self.marginal[:, 1]
        self.marginal[:, 1] = 1. - self.marginal[:, 0] - self.marginal[:, 2]

class PA(Epidemic):
    def __init__(self, g, epar, tau, init_i):
        """
        Pair Approximation (PA) simulation.
        """
        super().__init__(g, epar, tau, init_i)
        self.marginal_all.append(self.marginal.copy())
        self.edges = list(self.G.edges())
        self.algorithm = 'PA'
        self.IS = np.zeros((self.n, self.n))
        self.SS = np.zeros((self.n, self.n))
        for e in self.edges:
            [i, j] = e
            self.IS[i, j] = self.marginal[i, 1] * self.marginal[j, 0]
            self.IS[j, i] = self.marginal[j, 1] * self.marginal[i, 0]
            self.SS[i, j] = self.marginal[i, 0] * self.marginal[j, 0]
            self.SS[j, i] = self.marginal[i, 0] * self.marginal[j, 0]

    def evolution(self, t):
        """
        Run the PA simulation to time t.
        """
        self.t = t
        for _ in range(t):
            for __ in range(self.spt):
                self.step()
            self.marginal_all.append(self.marginal.copy())
        self.marginal_all = np.array(self.marginal_all)

    def step(self):
        """
        Perform one PA step.
        """
        new_IS, new_SS = self.update_pair()
        self.marginal = self.update_marginal()
        self.IS = new_IS
        self.SS = new_SS

    def update_pair(self):
        """
        Update pairwise probabilities for all edges.
        """
        new_IS = self.IS.copy()
        new_SS = self.SS.copy()
        for [a, b] in self.edges:
            sa = (np.sum(self.IS[:, a]) - self.IS[b, a]) / self.marginal[a, 0] if self.marginal[a, 0] != 0 else 0
            sb = (np.sum(self.IS[:, b]) - self.IS[a, b]) / self.marginal[b, 0] if self.marginal[b, 0] != 0 else 0

            da = self.l * self.SS[a, b] * sa
            db = self.l * self.SS[a, b] * sb
            new_SS[a, b] -= (da + db)
            new_SS[b, a] = new_SS[a, b]

            new_IS[a, b] += (-(self.r + self.l * (1 + sb)) * self.IS[a, b] + da)
            new_IS[b, a] += (-(self.r + self.l * (1 + sa)) * self.IS[b, a] + db)
        return [new_IS, new_SS]

    def update_marginal(self):
        """
        Update marginal probabilities for all nodes.
        """
        IS_sum = np.sum(self.IS, axis=0)
        deltai = np.minimum(self.l * IS_sum, self.marginal[:, 0])
        deltai = np.maximum(deltai, 0)

        deltar = self.r * self.marginal[:, 1]

        marginal = self.marginal.copy()
        marginal[:, 0] -= deltai
        marginal[:, 1] += deltai - deltar
        marginal[:, 2] += deltar

        return marginal

class ARM(PA):
    def __init__(self, g, epar, tau, init_i, partition, label):
        """
        Approximate Region Moment (ARM) simulation.
        """
        super().__init__(g, epar, tau, init_i)

        self.algorithm = 'ARM'
        self.algorithm_label += label

        self.Regions = []
        self.TN = []
        self.neighs_of_region = []  # Index by order of [region, node, neighbor]
        for region in partition:
            self.Regions.append(region)
            self.TN.append(Region(region, self.epar, self.marginal[list(region)], self.d))
            nodes = list(region)
            neighs = []
            for i in nodes:
                neigh = []
                for j in self.G[i]:
                    if j not in nodes:
                        neigh.append(j)
                neighs.append(neigh)
            self.neighs_of_region.append(neighs)

        [self.edges, self.edges_TN] = self.edges_classification()  # Remove other edges inside TN

        self.num_tn = len(partition)

    def edges_classification(self):
        """
        Get edges for PA and TN (region) parts.
        """
        leftg = nx.Graph(self.G)
        edges_TN = []
        for Region_graph in self.Regions:
            leftg.remove_edges_from(Region_graph.edges())
            edges_TN.append([e for e in Region_graph.edges()])
        edges_PA = list(leftg.edges())
        return [edges_PA, edges_TN]

    def evolution(self, t):
        """
        Run the ARM simulation to time t.
        """
        self.t = t
        for _ in range(t):
            for __ in range(self.spt):
                self.ARM_step()
            self.marginal_all.append(self.marginal.copy())
        self.marginal_all = np.array(self.marginal_all)

    def ARM_step(self):
        """
        Perform one ARM step, updating both TN and PA parts.
        """
        # First update TN with previous step info, but do not update marginals in TN
        for i in range(self.num_tn):
            nodes = list(self.Regions[i])
            neighs = self.neighs_of_region[i]
            msgin_all = np.zeros([len(nodes)])
            for j in range(len(nodes)):
                if len(neighs[j]) > 0 and self.marginal[nodes[j], 0] > 0:
                    msgin = np.sum(self.IS[neighs[j], nodes[j]]) / self.marginal[nodes[j], 0]
                    msgin_all[j] = min(msgin, 1. / self.l) # in case the error makes msgin > 1/lambda
            self.TN[i].update(msgin_all)

        new_IS, new_SS = super().update_pair()
        self.marginal = super().update_marginal()
        self.IS = new_IS
        self.SS = new_SS

        # Finally merge TN, update edges inside TN
        for i in range(self.num_tn):
            edges = self.edges_TN[i]
            for e in edges:
                [a, b] = e
                pp = self.TN[i].pair_marginal(a, b)
                self.IS[a, b] = pp[1, 0]
                self.IS[b, a] = pp[0, 1]
                self.SS[a, b] = pp[0, 0]
                self.SS[b, a] = pp[0, 0]

class Region:
    def __init__(self, G, epar, init_state, d):
        """
        Region class for ARM, representing a subgraph.
        """
        self.G = G
        self.nodes = list(self.G)
        self.n = len(self.nodes)
        self.l = epar[0]
        self.r = epar[1]
        self.d = d

        t = init_state[0]
        for node in range(1, self.n):
            t = np.kron(t, init_state[node])
        self.T = t.reshape([self.d for _ in range(self.n)])

        self.get_operators()

    def get_operators(self):
        """
        Precompute operators for infection and recovery.
        """
        infc = np.eye(self.d ** 2).reshape(self.d, self.d, self.d, self.d)
        infc[1, 1, 0, 1] += self.l
        infc[0, 1, 0, 1] -= self.l
        infc[1, 1, 1, 0] += self.l
        infc[1, 0, 1, 0] -= self.l
        self.infc = infc.reshape(self.d * self.d, self.d * self.d)
        getinfc = np.zeros([self.d, self.d])
        getinfc[1, 0] += self.l
        getinfc[0, 0] -= self.l
        self.getinfc = getinfc

        local = np.eye(self.d)
        local[2, 1] += self.r
        local[1, 1] -= self.r

        self.local = local

    def update(self, msgin):
        """
        Update the region's tensor T using incoming messages.
        """
        t = self.T
        for edge in list(self.G.edges()):
            [a, b] = edge
            i = self.nodes.index(a)
            j = self.nodes.index(b)
            if j < i:
                [i, j] = [j, i]
            t = np.swapaxes(np.swapaxes(t, i, 0), j, 1)
            t = t.reshape(self.d * self.d, -1)
            t = self.infc @ t
            t = t.reshape([self.d for _ in range(self.n)])
            t = np.swapaxes(np.swapaxes(t, j, 1), i, 0)

        for i in range(self.n):
            t = np.swapaxes(t, i, 0)
            t = t.reshape(self.d, -1)
            t = self.local @ t
            if msgin[i] > 0:
                t = (np.eye(self.d) + msgin[i] * self.getinfc) @ t
            t = t.reshape([self.d for _ in range(self.n)])
            t = np.swapaxes(t, i, 0)
        self.T = t

    def marginal(self, i):
        """
        Compute the marginal probability for node i in the region.
        """
        marginal = np.sum(self.T, axis=tuple([dim for dim in range(self.n) if (dim != i)]))
        return marginal

    def pair_marginal(self, n1, n2):
        """
        Compute the joint marginal for nodes n1 and n2 in the region.
        """
        i = self.nodes.index(n1)
        j = self.nodes.index(n2)
        pp = np.sum(self.T, axis=tuple([dim for dim in range(self.n) if (dim != i and dim != j)]))
        if j < i:
            pp = np.transpose(pp)
        return pp

def MC(para):
    """
    Monte Carlo simulation for SIR model.
    """
    [repeats, epar, n, d, t_max, seed, spt, init_state, adjacency_matrix] = para
    marginal_sum = np.zeros([t_max + 1, n, d])
    np.random.seed(seed)
    marginal_sum[0] = init_state * repeats
    infect_multiplier = epar[0] * adjacency_matrix
    for _ in range(repeats):
        marginal = init_state.copy()
        for t in range(1, t_max + 1):
            rand = np.random.rand(spt, n)
            for s in range(spt):
                current_rand = rand[s, :]
                infect_prob = infect_multiplier @ marginal[:, 1]

                infected = (marginal[:, 0] == 1) & (current_rand < infect_prob)
                recovered = (marginal[:, 1] == 1) & (current_rand < epar[1])

                marginal[infected] = [0, 1, 0]
                marginal[recovered] = [0, 0, 1]
            marginal_sum[t] += marginal
    return marginal_sum
