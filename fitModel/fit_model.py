from dataImport.selectivityMethods.mi import computeMI, plotBarGraphCentrality, plotBarGraphCentralityCompare
from dataImport.commons.basicFunctions import saccade_df, computerFrAll, createPlotDF, plotFun
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from fitModel.plot_network import plot_network
from bokeh.io import export_png
import numpy as np
import networkx as nx
import plotly.io as pio
import time

# plotly save configuration
pio.orca.ensure_server()
time.sleep(10)


def fit_model_discrete_time_network_hawkes_spike_and_slab(dtmax, hypers, itter, spikesData, completeData):

    writePath = '/home/mohsen/projects/pyhawkes/data/'
    period, data = zip(*spikesData.items())
    for per in range(len(period)):
        print('State:', '**** ', period[per], ' ****')

        model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
            K=data[per].shape[1], dt_max=dtmax,
            network_hypers=hypers)
        assert model.check_stability()

        model.add_data(data[per])

        ###########################################################
        # Fit the test model with Gibbs sampling
        ###########################################################
        samples = []
        lps = []

        for itr in range(itter):
            print("Gibbs iteration ", itr)
            model.resample_model()
            lps.append(model.log_probability())
            samples.append(model.copy_sample())

        # def analyze_samples(true_model, samples, lps):
        N_samples = len(samples)

        # Compute sample statistics for second half of samples
        A_samples = np.array([s.weight_model.A for s in samples])
        W_samples = np.array([s.weight_model.W for s in samples])
        lps = np.array(lps)

        offset = N_samples // 2
        A_mean = A_samples[offset:, ...].mean(axis=0)
        W_mean = W_samples[offset:, ...].mean(axis=0)

        # set week weight element to zero

        for (a1, b1) in np.ndindex(A_mean.shape):
            if A_mean[a1, b1] <= 0.5:
                A_mean[a1, b1] = 0

        W_effective_mean = A_mean * W_mean

        # Create Graph Objects
        typ = nx.DiGraph()
        G = nx.from_numpy_matrix(15 * W_effective_mean, create_using=typ)

        fv = plot_network(G, writePath + 'Network-' + str(period[per]))

        # Compute the mutual information

        saccade_data_set = saccade_df(completeData)
        mivaluesNoStim = computeMI(completeData, saccade_data_set, "noStim")

        plotBarGraphCentrality(mivaluesNoStim, fv, writePath + 'MutualInformation-' + str(period[per]))
        plotBarGraphCentralityCompare(mivaluesNoStim, fv, writePath + 'MutualInformationCompare-' + str(period[per]))

        # PSTH

        visualAndDelay = computerFrAll(completeData, 'vis')
        saccade = computerFrAll(completeData, 'saccade')

        export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=completeData[0], period='vis', ind=fv),
                           createPlotDF(DF=saccade, DF2=completeData[fv], period='sac', ind=fv)),
                   filename=writePath + 'FiringRate-' + str(period[per]) + '.png')



