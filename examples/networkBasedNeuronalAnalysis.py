import os
import numpy as np
from argparse import ArgumentParser
from dataImport.commons.basicFunctions import assembleData, conditionSelect
from fitModel.fit_model import fit_model_discrete_time_network_hawkes_spike_and_slab


parser = ArgumentParser(description='This is a Python program for analysis on network of neurons to '
                                    'detect functional connectivity between neurons')


parser.add_argument('-d', '--data',  action='store',
                    dest='data', help='Raw data directory')


parser.add_argument('-H', '--host', action='store',
                    dest='host', help='MongoDB host name')

parser.add_argument('-p', '--port',action='store',
                    dest='port', help='MongoDB port number')

parser.add_argument('-w', '--write', action='store',
                    dest='write', help='Output directory')

parser.add_argument('-s', '--sparsity', action='store',
                    dest='sparsity', help='Initial sparsity of the network', type=float)

parser.add_argument('-S', '--self', action='store_true',
                    default=False,
                    dest='self', help='Allow self connection')

parser.add_argument('-l', '--lag', action='store',
                    dest='lag', help='Impulse response lag', type=int)

parser.add_argument('-i', '--iter', action='store',
                    dest='iter', help='Number of MCMC iteration', type=int)

parser.add_argument('-c', '--chain', action='store',
                    dest='chain', help='Number of MCMC chain', type=int)

parser.add_argument('-v', '--version', action='version',
                    dest='', version='%(prog)s 0.1')


args = parser.parse_args()

# prepare data

# read all neurons
dirr = os.fsencode(args.data)
allNeurons = assembleData(dirr)

# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
np.sort(np.array([sum(allNeurons[x].iloc[:, 0:(minTime - 9)].sum()) for x in range(len(allNeurons))]))

# slicing time to decompose Enc, Memory and saccade times

neuronalData = {'Enc-In-NoStim': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-NoStim': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-NoStim': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-In-Stim': np.array([conditionSelect(allNeurons[b], 'inStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-In-Stim': np.array([conditionSelect(allNeurons[b], 'inStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-In-Stim': np.array([conditionSelect(allNeurons[b], 'inStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-Out-NoStim': np.array([conditionSelect(allNeurons[b], 'OutNoStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Enc-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Mem-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Sac-Out-Stim': np.array([conditionSelect(allNeurons[b], 'outStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose()
                }

# network hyper parameter definition

network_hypers = {"p": args.sparsity, "allow_self_connections": args.self}


# fit model

# Chain loop

# for chain in range(args.chain):
#     fit_par = partial(fit_model_discrete_time_network_hawkes_spike_and_slab, *[args.lag, network_hypers,
#                                                                                args.iter, data, period, allNeurons,
#                                                                                chain,
#                                                                                args])

fit_model_discrete_time_network_hawkes_spike_and_slab(args.lag, network_hypers, args.iter, neuronalData,
                                                      allNeurons, 1, args)

# Gelman-Rubin convergence statistics

from fitModel.GelmanRubin_convergence import compute_gelman_rubin_convergence
compute_gelman_rubin_convergence(args)


