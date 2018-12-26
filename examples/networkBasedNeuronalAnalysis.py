import os
import numpy as np
from dataImport.commons.basicFunctions import assembleData, conditionSelect
from fitModel.fit_model import fit_model_discrete_time_network_hawkes_spike_and_slab




# prepare data

# read all neurons
dirr = os.fsencode("/home/mohsen/projects/neuroScienceWork/data")
allNeurons = assembleData(dirr)

# align end date
minTime = np.min([allNeurons[x].shape[1] for x in range(len(allNeurons))])
np.sort(np.array([sum(allNeurons[x].iloc[:, 0:(minTime - 9)].sum()) for x in range(len(allNeurons))]))

# slicing time to decompose Enc, Memory and saccade times

neuronalData = {'Enc': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 1050:1250].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Mem': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 2500:2700].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose(),
                'Sac': np.array([conditionSelect(allNeurons[b], 'inNoStim').iloc[:, 3150:3350].sum(axis=0)
                      for b in range(len(allNeurons))]).transpose()}

# network hyper parameter definition

network_hypers = {"p": 0.4, "allow_self_connections": False}

fit_model_discrete_time_network_hawkes_spike_and_slab(30, network_hypers, 2000, neuronalData, allNeurons)


