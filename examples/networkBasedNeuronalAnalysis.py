import os
from commons.basicFunctions import assembleData, conditionSelect, saccade_df, \
    computerFrAll, createPlotDF, plotFun
from selectivityMethods.mi import computeMI, plotBarGraphCentrality, \
    plotBarGraphCentralityCompare
import numpy as np
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import export_png
from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)

# preparing data

dirr = os.fsencode("/home/mohsen/projects/neuroScienceWork/data")
tmp = assembleData(dirr)
minTime = np.min([tmp[x].shape[1] for x in range(len(tmp))])

np.sort(np.array([sum(tmp[x].iloc[:, 0:(minTime - 9)].sum()) for x in range(len(tmp))]))

# slicing time to decompose Enc, Memory and saccade times

# Enc
spikesEnc = np.array([conditionSelect(tmp[b], 'inNoStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(tmp))]).transpose()
# Memory
spikesMem = np.array([conditionSelect(tmp[b], 'inNoStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(tmp))]).transpose()
# Saccade
spikesSac = np.array([conditionSelect(tmp[b], 'inNoStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(tmp))]).transpose()


def fit_model_gibbs(K, T, dtmax, hypers, itter, spikes):
    true_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dtmax,
        network_hypers=hypers)
    assert true_model.check_stability()

    true_model.generate(T=T, keep=True, print_interval=50)

    ###########################################################
    # Create a test spike and slab model
    ###########################################################

    test_model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
        K=K, dt_max=dtmax, network_hypers=hypers)

    test_model.add_data(spikes)

    ###########################################################
    # Fit the test model with Gibbs sampling
    ###########################################################
    samples = []
    lps = []

    for itr in range(itter):
        print("Gibbs iteration ", itr)
        test_model.resample_model()
        lps.append(test_model.log_probability())
        samples.append(test_model.copy_sample())

    return lps, samples


network_hypers = {"p": 0.4, "allow_self_connections": False}

a, b = fit_model_gibbs(spikes=spikesEnc, K=spikesEnc.shape[1],
                       T=spikesEnc.shape[0], dtmax=30,
                       hypers=network_hypers, itter=1500)

# def analyze_samples(true_model, samples, lps):
N_samples = len(b)

# Compute sample statistics for second half of samples
A_samples = np.array([s.weight_model.A for s in b])
W_samples = np.array([s.weight_model.W for s in b])
W_effective_samples = np.array([s.weight_model.W_effective for s in b])
lps = np.array(a)

offset = N_samples // 2
A_mean = A_samples[offset:, ...].mean(axis=0)
W_mean = W_samples[offset:, ...].mean(axis=0)

# W_effective_samples[offset:, ...].mean(axis=0)

# set week weight element to zero

for (a1, b1) in np.ndindex(A_mean.shape):
    if A_mean[a1, b1] <= 0.5:
        A_mean[a1, b1] = 0

W_effective_mean = A_mean * W_mean

typ = nx.DiGraph()
G = nx.from_numpy_matrix(15 * W_effective_mean, create_using=typ)


def plot_network(G, figname):
    ##Creates a copy of the graph
    H = G.copy()

    # crates a list for edges and for the weights
    edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())

    #####calculates the degree of each node
    d = nx.degree(H)
    #####creates list of nodes and a list their degrees that will be used later for their sizes
    nodelist, node_sizes = zip(*dict(d).items())

    # positions
    positions = nx.circular_layout(H)

    # Figure size
    plt.figure(figsize=(15, 15))

    # draws nodes
    nx.draw_networkx_nodes(H, positions, node_color='#DA70D6', nodelist=nodelist,
                           #####the node size will be now based on its degree
                           node_size=tuple([x ** 3 for x in node_sizes]), alpha=0.8)

    # Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=8,
                            font_family='sans-serif')

    # draws the edges
    nx.draw_networkx_edges(H, positions, edge_list=edges, style='solid',
                           ###adds width=weights and edge_color = weights
                           ###so that edges are based on the weight parameter
                           ###edge_cmap is for the color scale based on the weight
                           ### edge_vmin and edge_vmax assign the min and max weights for the width
                           width=weights, edge_color=weights, edge_cmap=plt.cm.PuRd,
                           edge_vmin=min(weights), edge_vmax=max(weights))

    dc = nx.degree_centrality(G)
    a1, a2 = zip(*dc.items())
    # displays the graph without axis
    plt.axis('off')
    # saves image
    # plt.savefig("part3.png", format="PNG")
    plt.savefig('Network-' + figname + '.svg', format='svg', dpi=1200)
    # fv = np.array([nodelist, node_sizes]).transpose()
    # fv[:,0][fv[:,1] == fv[:,1].max()],
    return np.argmax(a2), plt.show()


# allPairNodeCon = nx.all_pairs_node_connectivity(G)
# ew =  np.array([sum(allPairNodeCon[ww].values()) for ww in allPairNodeCon.keys()])

fv, _ = plot_network(G, 'Enc')

# Compute the mutual information

saccade_data_set = saccade_df(tmp)
mivaluesNoStim = computeMI(tmp, saccade_data_set, "noStim")

plotBarGraphCentrality(mivaluesNoStim, fv, 'test')
plotBarGraphCentralityCompare(mivaluesNoStim, fv, 'test')

# PSTH

visualAndDelay = computerFrAll(tmp, 'vis')
saccade = computerFrAll(tmp, 'saccade')

export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=tmp[0], period='vis', ind=fv),
                   createPlotDF(DF=saccade, DF2=tmp[fv], period='sac', ind=fv)),
           filename=str(fv) + '.png')

# Centrality

nx.degree_centrality(G)
nx.eigenvector_centrality(G)
nx.load_centrality(G)

nx.average_node_connectivity(G)
nx.edge_connectivity(G)
nx.node_connectivity(G)
nx.max_clique(G)
nx.average_clustering(G)



