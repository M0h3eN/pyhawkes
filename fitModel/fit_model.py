from dataImport.selectivityMethods.mi import computeMI, plotBarGraphCentrality, plotBarGraphCentralityCompare
from dataImport.commons.basicFunctions import saccade_df, computerFrAll, createPlotDF, plotFun
from pyhawkes.models import DiscreteTimeNetworkHawkesModelSpikeAndSlab
from fitModel.plot_network import plot_network
from bokeh.io import export_png
from pymongo import MongoClient
from networkx.readwrite import json_graph
import os
import numpy as np
import networkx as nx
import plotly.io as pio
import time


# plotly save configuration
pio.orca.ensure_server()
time.sleep(10)


def fit_model_discrete_time_network_hawkes_spike_and_slab(dtmax, hypers, itter, spikesData, completeData, chainsNumber, args):

    # MongoDB connection config

    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    paramValuesDB = client.MCMC_param
    diagnosticValuesDB = client.MCMC_diag
    GraphDB = client.Graph
    EstimatedGrapgDB = client.Estimation
    MutualInformation = client.MutualInformation


    period, data = zip(*spikesData.items())

    # Compute the mutual information

    saccade_data_set = saccade_df(completeData)
    mivaluesDict = dict(Stim=computeMI(completeData, saccade_data_set, 'Stim').to_dict('list'),
                                 NoStim=computeMI(completeData, saccade_data_set, 'NoStim').to_dict('list'))
    mivalues = dict(Stim=computeMI(completeData, saccade_data_set, 'Stim'),
                                 NoStim=computeMI(completeData, saccade_data_set, 'NoStim'))

    # Mutual Information ingestion

    MutualInformation['Mi'].insert_one(mivaluesDict)

    # Chain loop
    for chain in range(chainsNumber):

        writePath = args.write + 'Chain' + str(chain + 1)
        if not os.path.exists(writePath):
            os.makedirs(writePath)
        tempPath = writePath

        for per in range(len(period)):
            # directory management
            writePath = tempPath + '/' + period[per]
            if not os.path.exists(writePath):
                os.makedirs(writePath)

            writePath = writePath + '/'

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
            B = samples[1].impulse_model.B

            # Ingestion each model MCMC samples to mongoDB

            Akeys = []
            Wkeys = []
            WeKeys = []
            for i in range(k):
                for j in range(k):
                    Akeys.append("a_" + str(i + 1) + ',' + str(j + 1))
                    Wkeys.append("w_" + str(i + 1) + ',' + str(j + 1))
                    WeKeys.append("we_" + str(i + 1) + ',' + str(j + 1))

            Imkeys = []
            for i in range(k):
                for j in range(k):
                    for p1 in range(B):
                        Imkeys.append("im_" + str(i + 1) + ',' + str(j + 1) + ',' + str(p1 + 1))

            LamKeys = []
            for i in range(k):
                LamKeys.append("la_" + str(i + 1))

            # Strange, BSON doesnt know about python int

            allKeys = Akeys + Wkeys + WeKeys + Imkeys + LamKeys

            All_samples_param = (
                [dict(zip(allKeys, (
                        list(np.array(s.weight_model.A.flatten(), "float")) +
                        list(np.array(s.weight_model.W.flatten(), "float")) +
                        list(np.array(s.weight_model.W_effective.flatten(), "float")) +
                        list(np.array(np.reshape(s.impulse_model.g, (k, k, s.impulse_model.B)).flatten(),
                                      "float")) +
                        list(np.array(s.bias_model.lambda0.flatten(), "float"))
                )
                          )) for s in samples])

            colName = period[per] + '___' + str(chain)
            paramValuesDB[colName].insert_many(All_samples_param)

            A_samples = np.array([s.weight_model.A for s in samples])
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

            modelDiag = {'Model': str(model.__class__).split(".")[2].split("'")[0],
                         'logLik': lps,
                         'D_hat': D_hat,
                         'D_bar': D_bar,
                         'pD': pD,
                         'pV': pV,
                         'DIC': DIC}

            colNameDiag = period[per] + '___' + str(chain)
            diagnosticValuesDB[colNameDiag].insert_one(modelDiag)

            # Compute sample statistics for second half of samples

            offset = N_samples // 2
            W_effective_mean = W_effective_sample[offset:, ...].mean(axis=0)

            # Insert estimated graph after burnIn phase

            EstimatedGrapgDB[colNameDiag].insert_one(dict(zip(WeKeys,
                                                              list(np.array(W_effective_mean.flatten(), "float")))))



            # set week weight element to zero

            # for (a1, b1) in np.ndindex(A_mean.shape):
            #     if A_mean[a1, b1] <= 0.5:
            #         A_mean[a1, b1] = 0
            #
            # W_effective_mean = A_mean * W_mean

            # Create Graph Objects
            typ = nx.DiGraph()
            G0 = nx.from_numpy_matrix(W_effective_mean, create_using=typ)
            G = nx.from_numpy_matrix(15 * W_effective_mean, create_using=typ)

            dataGraph = json_graph.adjacency_data(G0)

            colNameGraph = period[per] + '___' + str(chain)
            GraphDB[colNameGraph].insert_one(dataGraph)

            fv = plot_network(G, writePath + 'Network')

            plotBarGraphCentrality(mivalues[period[per].split("-")[2]], fv, writePath + 'MutualInformation')
            plotBarGraphCentralityCompare(mivalues[period[per].split("-")[2]], fv, writePath + 'MutualInformationCompare')

            # PSTH

            visualAndDelay = computerFrAll(completeData, 'vis')
            saccade = computerFrAll(completeData, 'saccade')

            export_png(plotFun(createPlotDF(DF=visualAndDelay, DF2=completeData[0], period='vis', ind=fv),
                               createPlotDF(DF=saccade, DF2=completeData[fv], period='sac', ind=fv)),
                       filename=writePath + 'FiringRate' + '.png')



