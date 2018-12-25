import os
from dataImport.commons.basicFunctions import assembleData, conditionSelect, saccade_df,\
    computerFrAll, createPlotDF, plotFun
from dataImport.selectivityMethods.mi import computeMI, plotBarGraphCentrality, plotBarGraphCentralityCompare
from fitModel.fit_model import fit_model_discrete_time_network_hawkes_spike_and_slab
from fitModel.plot_network import plot_network
import numpy as np
import networkx as nx
import plotly.io as pio
from bokeh.io import export_png
import time
pio.orca.ensure_server()
time.sleep(10)
# from plotly.offline import init_notebook_mode

# init_notebook_mode(connected=True)

# prepare data

# read all neurons
dirr = os.fsencode("/home/mohsen/projects/neuroScienceWork/data")
tmp = assembleData(dirr)

# align end date
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

# network hyper parameter definition

network_hypers = {"p": 0.4, "allow_self_connections": False}

a, b = fit_model_discrete_time_network_hawkes_spike_and_slab(spikes=spikesEnc, K = spikesEnc.shape[1],
                       T=spikesEnc.shape[0], dtmax=30,
                       hypers=network_hypers, itter=50)

# def analyze_samples(true_model, samples, lps):
N_samples = len(b)

# Compute sample statistics for second half of samples
A_samples = np.array([s.weight_model.A for s in b])
W_samples = np.array([s.weight_model.W for s in b])
lps = np.array(a)

offset = N_samples // 2
A_mean = A_samples[offset:, ...].mean(axis=0)
W_mean = W_samples[offset:, ...].mean(axis=0)

# set week weight element to zero

for (a1, b1) in np.ndindex(A_mean.shape):
    if A_mean[a1, b1] <= 0.5:
        A_mean[a1, b1] = 0

W_effective_mean = A_mean * W_mean

typ = nx.DiGraph()
G = nx.from_numpy_matrix(15 * W_effective_mean, create_using=typ)



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

# nx.degree_centrality(G)
# nx.eigenvector_centrality(G)
# nx.load_centrality(G)
#
# nx.average_node_connectivity(G)
# nx.edge_connectivity(G)
# nx.node_connectivity(G)
# nx.max_clique(G)
# nx.average_clustering(G)



