import numpy as np
import pandas as pd
from pymongo import MongoClient

# compute GR convergence for all models


def compute_gelman_rubin_convergence(args):
    # MCMC sample collection config

    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    paramValuesDB = client['MCMC_param']
    GRConvergenceDB = client['Convergence_GR']

    # drop select mongoDB _id

    projection = {"_id": 0}

    # get all model name

    collectionNames = paramValuesDB.list_collection_names()

    # MCMC run iteration

    n = paramValuesDB[collectionNames[0]].count()

    # MCMC chain

    maxNumberOfChain = np.max([int(collectionNames[x].split("_")[3]) for x in range(len(collectionNames))])
    models = np.unique(np.array(([collectionNames[x].split("_")[0] for x in range(len(collectionNames))])))

    for model in models:

        thetaBarArr = []
        varianceArr = []

        for chain in range(maxNumberOfChain + 1):
            thetaBarArr.append(
                pd.DataFrame(list(paramValuesDB[model + '___' + str(chain)].find({}, projection=projection))).mean())
            varianceArr.append(
                pd.DataFrame(list(paramValuesDB[model + '___' + str(chain)].find({}, projection=projection))).var())

        W = pd.DataFrame(varianceArr).mean()
        B = pd.DataFrame(thetaBarArr).var() * n
        varThetaHat = (1 - 1 / n) * W + (1 / n) * B

        rHat = np.sqrt(varThetaHat / W).to_dict()

        # ingest statistics to mongoDB

        print('GR computing State:', '**** ', model, ' ****')

        GRConvergenceDB[model].insert_one(rHat)









