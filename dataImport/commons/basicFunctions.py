import os

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
from .mat_to_dict import loadmat
from pymongo import MongoClient
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Span, Label
from bokeh.palettes import Reds4, Blues4
from bokeh.plotting import figure




# Columns name

def generatorTemp(size):
    for ii in range(size):
        yield 'T' + str(ii)


# generate eye data

def generateEyeDF(eye):
    tempList = {}

    for i in np.arange(4):
        tempList[i] = pd.DataFrame(data=eye[0][0][0][0][i])
    tdf = pd.DataFrame(data=pd.concat([tempList[0], tempList[1],
                                       tempList[2], tempList[3]], axis=1))
    colsName = [
        'SacStartTime',
        'SacStopTime',
        'SacStartPosX',
        'SacStartPosY',
        'SacStopPosX',
        'SacStopPosY',
    ]
    tdf.columns = colsName
    return tdf


# assemble data in all neurons

def assembleData(directory, args):

    client = MongoClient("mongodb://" + args.host + ':' + args.port)
    preProcDB = client.preProccesing

    dirr = directory
    os.chdir(dirr)
    iter = 0
    neurons = {}
    allNeurons = {}
    # spkLenOuter = 0
    # spkLeninner = 0

    for file in os.listdir(dirr):
        filename = os.fsdecode(file)
        if filename.endswith('.mat'):
            print('File:' + filename)
            preProcDB['Raw_Data'].insert_one(loadmat(filename).get('Res'))
            matData = sio.loadmat(filename)
            dictVal = matData.get('Res')
            Eye = dictVal['Eye']
            spk = dictVal['spk']
            Cond = dictVal['Cond']
            spkLenOuter = len(spk[0][0])
            for iter0 in range(spkLenOuter):
                spkLeninner = len(spk[0][0][iter0])
                for iter2 in range(spkLeninner):
                    arrSize = spk[0][0][iter0][iter2]
                    # print(spkLen)
                    if arrSize.size > 0:
                        df = pd.DataFrame(spk[0][0][iter0][iter2])
                        if sum(df.sum()) > 3000:
                            tmp = df  # .iloc[:,0:3799]
                            colName = generatorTemp(tmp.shape[1])
                            tmp.columns = colName
                            neurons[iter] = pd.concat([tmp,
                                                   pd.DataFrame(data={'Cond': Cond[0][0][0]})],
                                                  axis=1)
                            neurons[iter]['stimStatus'] = np.where(neurons[iter]['Cond'] > 8, 0, 1)
                            neurons[iter]['inOutStatus'] = np.where(neurons[iter]['Cond'] % 2 == 1, 1, 0)
                            allNeurons[iter] = pd.concat([neurons[iter],
                                                      generateEyeDF(Eye)], axis=1)
                            print("Neuron" + str(iter))
                            iter = iter + 1
                        else:
                            iter = iter
                            print("Neuron" + str(iter) + " " + "got few action potentials, skipping...")
    return allNeurons


def saccade_df(neurons_df):
    saccade_df = {}
    tmp_list1 = []
    tmp_list2 = []

    neurons_df1 = {}

    for numerator in range(len(neurons_df)):
        saccade_time = neurons_df[numerator]['SacStartTime']
        neurons_df1[numerator] = neurons_df[numerator].iloc[:, 
                   0:(neurons_df[numerator].columns.get_loc("Cond") - 1)]
        for i in range(len(saccade_time)):
            tmp_list1.append(
                neurons_df1[numerator].iloc[i, 
                           (saccade_time[i] - 3000):saccade_time[i]].reset_index(drop=True))
            tmp_list2.append(
                neurons_df1[numerator].iloc[i, 
                           (saccade_time[i] + 1):(saccade_time[i] + 400)].reset_index(drop=True))
        saccade_df[numerator] = pd.concat([pd.DataFrame(tmp_list1),
                                           pd.DataFrame(tmp_list2),
                                           neurons_df[numerator].iloc[:,
                                           (neurons_df[numerator].columns.get_loc("Cond")):
                                           (neurons_df[numerator].columns.get_loc("SacStopPosY"))]], axis=1)
        tmp_list1 = []
        tmp_list2 = []
    return saccade_df


def sacTime(df):
    dfNew = int(df['SacStartTime'].mean())
    return dfNew


# Compute firing rate

def computeFr(df, min, max):
    dtemp = np.mean(df.iloc[:, min:max]) * 1000
    return dtemp

def computeSpikeCount(df, min, max):
    dtemp = np.sum(df.iloc[:, min:max])
    return dtemp


def normalize(DF):
    return (DF - np.min(DF)) / (np.max(DF) - np.min(DF))


# select different conditions

def conditionSelect(df, subStatus):
    if subStatus == 'inStim':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 1)]
    elif subStatus == 'outStim':
        dfNew = df[(df['stimStatus'] == 1) & (df['inOutStatus'] == 0)]
    elif subStatus == 'inNoStim':
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 1)]
    elif subStatus == 'allStim':
        dfNew = df[(df['stimStatus'] == 1)]
    elif subStatus == 'allNoStim':
        dfNew = df[(df['stimStatus'] == 0)]
    else:
        dfNew = df[(df['stimStatus'] == 0) & (df['inOutStatus'] == 0)]
    return dfNew


# general firing rate computations

def computerFrAll(neurons_df, period):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = {}
    if period == 'vis':
        for it in range(lend):
            sep_by_cond[it] = [computeFr(conditionSelect(neurons_df[it],
                                                         'inStim'), 0, 3000),
            #(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'outStim'), 0, 3000),
            #(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'inNoStim'), 0, 3000),
#(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeFr(conditionSelect(neurons_df[it],
                                                         'outNoStim'), 0, 3000)]
#(neurons_df[it].columns.get_loc("Cond") - 1))
    else:
        for it in range(lend):
            inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
            outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
            inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
            outNoStimDF = conditionSelect(saccade_data_frame[it], 'outNoStim')

            sep_by_cond[it] = [computeFr(inStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(outStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(inNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeFr(outNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1))]
    return sep_by_cond


def computerSpkCountAll(neurons_df, period):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = {}
    if period == 'vis':
        for it in range(lend):
            sep_by_cond[it] = [computeSpikeCount(conditionSelect(neurons_df[it],
                                                         'inStim'), 0, 3000),
            #(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                         'outStim'), 0, 3000),
            #(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                         'inNoStim'), 0, 3000),
#(neurons_df[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(conditionSelect(neurons_df[it],
                                                         'outNoStim'), 0, 3000)]
#(neurons_df[it].columns.get_loc("Cond") - 1))
    else:
        for it in range(lend):
            inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
            outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
            inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
            outNoStimDF = conditionSelect(saccade_data_frame[it], 'outNoStim')

            sep_by_cond[it] = [computeSpikeCount(inStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(outStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(inNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1)),
                               computeSpikeCount(outNoStimDF, 0, (saccade_data_frame[it].columns.get_loc("Cond") - 1))]
    return sep_by_cond


def computerFrAllDict(neurons_df):
    lend = len(neurons_df)
    saccade_data_frame = saccade_df(neurons_df)
    sep_by_cond = []

    for it in range(lend):

        inStimDF = conditionSelect(saccade_data_frame[it], 'inStim')
        outStimDF = conditionSelect(saccade_data_frame[it], 'outStim')
        inNoStimDF = conditionSelect(saccade_data_frame[it], 'inNoStim')
        outNoStimDF = conditionSelect(saccade_data_frame[it],
                'outNoStim')

        sep_by_cond.append({'visual': {
            'inStim': computeFr(conditionSelect(neurons_df[it], 'inStim'
                                ), 0, 3000).to_dict('list'),
            'outStim': computeFr(conditionSelect(neurons_df[it],
                                 'outStim'), 0, 3000).to_dict('list'),
            'inNoStim': computeFr(conditionSelect(neurons_df[it],
                                  'inNoStim'), 0, 3000).to_dict('list'
                    ),
            'outNoStim': computeFr(conditionSelect(neurons_df[it],
                                   'outNoStim'), 0, 3000).to_dict('list'
                    ),
            }, 'saccade': {
            'inStim': computeFr(inStimDF, 0,
                                saccade_data_frame[it].columns.get_loc('Cond'
                                ) - 1).to_dict('list'),
            'outStim': computeFr(outStimDF, 0,
                                 saccade_data_frame[it].columns.get_loc('Cond'
                                 ) - 1).to_dict('list'),
            'inNoStim': computeFr(inNoStimDF, 0,
                                  saccade_data_frame[it].columns.get_loc('Cond'
                                  ) - 1).to_dict('list'),
            'outNoStim': computeFr(outNoStimDF, 0,
                                   saccade_data_frame[it].columns.get_loc('Cond'
                                   ) - 1).to_dict('list'),
            }})


"""
0 -> inStim
1 -> outStim
2 -> inNoStim
3 -> outNoStim
"""


# sig.savgol_filter


def createPlotDF(DF, DF2, period, ind):
    if period == 'sac':
        inStim = sig.savgol_filter(DF[ind][0], 415, 3)
        x = np.linspace(0, 3400, len(inStim)) - 3000
        inNoStim = sig.savgol_filter(DF[ind][2], 415, 3)
        outStim = sig.savgol_filter(DF[ind][1], 415, 3)
        outNoStim = sig.savgol_filter(DF[ind][3], 415, 3)
        df = pd.DataFrame(data=dict(x = x,
                                    inStim=inStim,
                                    inNoStim=inNoStim,
                                    outStim=outStim,
                                    outNoStim=outNoStim))
    else:
        inStim = sig.savgol_filter(DF[ind][0], 415, 3)
        inNoStim = sig.savgol_filter(DF[ind][2], 415, 3)
        outStim = sig.savgol_filter(DF[ind][1], 415, 3)
        outNoStim = sig.savgol_filter(DF[ind][3], 415, 3)
        x = np.linspace(0, 3000, len(inStim)) - 1000
        df = pd.DataFrame(data=dict(x=x,
                                    inStim=inStim,
                                    inNoStim=inNoStim,
                                    outStim=outStim,
                                    outNoStim=outNoStim))
    return df


def plotVisDel(DF, s1, s2, s3, xlab, ylab):
    source = ColumnDataSource(DF.reset_index())
    p = figure(title=str(s3), x_axis_label=str(xlab),
               y_axis_label=str(ylab), toolbar_location=None)

    p.line(x='x', y=str(s1), color=Blues4[0],
           source=source, legend="")
    p.line(x='x', y=str(s2), color=Reds4[0],
           source=source, legend="")

    vline = Span(location=1000, dimension='height', line_dash='dashed',
                 line_color='black', line_width=2)
    vline0 = Span(location=0, dimension='height', line_dash='dashed',
                  line_color='grey', line_width=2)

    maxY = max([max(DF.inStim), max(DF.inStim)])
    text0 = Label(x=300, y=maxY, text='Visual period', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    text1 = Label(x=1300, y=maxY, text='Delay period', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    text2 = Label(x=-700, y=maxY, text='Baseline', render_mode='css',
                  border_line_color='black', border_line_alpha=1.5,
                  background_fill_color='white', background_fill_alpha=1.5)
    p.add_layout(vline)
    p.add_layout(vline0)
    p.add_layout(text0)
    p.add_layout(text1)
    p.add_layout(text2)
    return p


def plotSac(DF, s1, s2, s3, xlab, ylab):
    source1 = ColumnDataSource(DF.reset_index())
    s = figure(title=str(s3),
               x_axis_label=str(xlab),
               y_axis_label=str(ylab), toolbar_location=None)
    s.line(x='x', y=str(s1), color=Blues4[0],
           source=source1, legend="in")
    s.line(x='x', y=str(s2), color=Reds4[0],
           source=source1, legend="out")
    vline = Span(location=0, dimension='height', line_dash='dashed',
                 line_color='black', line_width=2)
    s.add_layout(vline)
    # s.x_range=Range1d(-400, 350)
    return s


def plotFun(DF1, DF2):
    wpin = plotVisDel(DF1, 'inStim', 'outStim', 'With stimmulation',
                      xlab='Time from stimulus onset(ms)',
                      ylab='Firing rate(Hz)')
    wsin = plotSac(DF2, 'inStim', 'outStim', 'Saccade period',
                   xlab='Time from saccade onset(ms)', ylab='')
    wpout = plotVisDel(DF1, 'inNoStim', 'outNoStim', 'Without stimmulation',
                       xlab='Time from stimulus onset(ms)',
                       ylab='Firing rate(Hz)')
    wsout = plotSac(DF2, 'inNoStim', 'outNoStim', '',
                    xlab='Time from saccade onset(ms)', ylab='')
    grid = gridplot([wpin, wsin, wpout, wsout], ncols=2,
                    sizing_mode='stretch_both', toolbar_location=None)
    return grid
