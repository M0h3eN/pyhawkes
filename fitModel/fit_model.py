from dataImport.selectivityMethods.mi import computeMI, plotBarGraphCentrality, plotBarGraphCentralityCompare
from dataImport.commons.basicFunctions import saccade_df, computerFrAll, createPlotDF, plotFun
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from fitModel.plot_network import plot_network
from bokeh.io import export_png
import numpy as np
import networkx as nx
import plotly.io as pio
import time
from pymongo import MongoClient

client = MongoClient()
db = client.MCMC_diagnostics


# plotly save configuration
pio.orca.ensure_server()
time.sleep(10)


def fit_model_discrete_time_network_hawkes_spike_and_slab(dtmax, hypers, itter, spikesData, completeData, chainsNumber):

    writePath = '/home/mohsen/projects/pyhawkes/data/'
    period, data = zip(*spikesData.items())

    # Chain loop
    for chain in range(chainsNumber):

        for per in range(len(period)):
            print('State:', '**** ', period[per], 'CHAIN: ', str(chain), ' ****')
            k = data[per].shape[1]

            model = DiscreteTimeNetworkHawkesModelSpikeAndSlab(
                K=k, dt_max=dtmax,
                network_hypers=hypers)
            assert model.check_stability()

            model.add_data(data[per])

            ###########################################################
            # Fit the test model with Gibbs sampling
            ###########################################################
            samples = []
            lps = []

            for itr in range(itter):
                # print("Gibbs iteration ", itr)
                model.resample_model()
                lps.append(model.log_probability())
                samples.append(model.copy_sample())

            # def analyze_samples(true_model, samples, lps):
            N_samples = len(samples)

            # Compute sample statistics for second half of samples
            #A_samples = np.array([s.weight_model.A for s in samples])

            Akeys = {}
            Wkeys = {}
            WeKeys = {}
            a = 1
            for i in range(k):
                for j in range(k):
                    Akeys[a] = "a_" + str(i + 1) + str(j + 1)
                    Wkeys[a] = "w_" + str(i + 1) + str(j + 1)
                    WeKeys[a] = "we_" + str(i + 1) + str(j + 1)
                    a = a + 1

                    # Strange that doesnt know about int


            A_samples = (
                [dict(zip(Akeys.values(), np.array(s.weight_model.A.flatten(), "float"))) for s in samples])

            colName = period[per] + str(chain)

            db.colName.insert_many(A_samples)

            W_samples = np.array([s.weight_model.W for s in samples])
            W_effective_sample = np.array([s.weight_model.W_effective for s in samples])
            LambdaZero_sample = np.array([s.bias_model.lambda0 for s in samples])
            ImpulseG_sample = np.array([np.reshape(s.impulse_model.g, (k,k,s.impulse_model.B)) for s in samples])

            # DIC evaluation

            # theta_bar evaluation

            A_mean = A_samples[:, ...].mean(axis=0)
            W_mean = W_samples[:, ...].mean(axis=0)
            LambdaZero_mean = LambdaZero_sample[:, ...].mean(axis=0)
            ImpulseG_mean = ImpulseG_sample[:, ...].mean(axis=0)

            logLik = np.array(lps)

            # D_hat evaluation

            D_hat = -2 * (model.weight_model.log_likelihood(tuple((A_mean, W_mean))) +
                    model.impulse_model.log_likelihood(ImpulseG_mean) +
                    model.bias_model.log_likelihood(LambdaZero_mean))
            D_bar = -2*np.mean(logLik)
            pD = D_bar - D_hat
            pV = np.var(-2*logLik)/2

            DIC = pD + D_bar

            print("DIC for model in ", str(period[per]), " : ", str(DIC))

            offset = N_samples // 2
            W_effective_mean = W_effective_sample[offset:, ...].mean(axis=0)

            # set week weight element to zero

            # for (a1, b1) in np.ndindex(A_mean.shape):
            #     if A_mean[a1, b1] <= 0.5:
            #         A_mean[a1, b1] = 0
            #
            # W_effective_mean = A_mean * W_mean

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



