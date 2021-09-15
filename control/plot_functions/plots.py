import rospy
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import math
import torch
import anfis_codes.anfis


matplotlib.use('Agg')
dtype = torch.float

robot_path = []
dis_error = []

def plot_mamdani(actor,summary, epoch):
    cose =  actor.layer['consequent'].mamdani_defs
    cose.cache()

    values = cose.cache_output_values

    fig, ax = plt.subplots()
    s = 1

    for key, value in values.items():
        ax.plot([value - 1 / s, value, value + 1 / s], [0,1,0], label =cose.names[key])
    summary.add_figure("Consequent_Membership/Mamdani_output", fig, epoch+1)

def _plot_mfs(var_name, fv, model, summary,epoch):
    '''
        A simple utility function to plot the MFs for a variable.
        Supply the variable name, MFs and a set of x values to plot.
    '''

    zero_length = (model.number_of_mfs[model.input_keywords[0]])
    x = torch.zeros(10000)
    y = -5

    fig, ax = plt.subplots()

    for i in range(10000):
        x[i] = torch.tensor(y)
        y += 0.001
    for mfname, yvals in fv.fuzzify(x):
        temp = 'mf{}'.format(zero_length)
        if (mfname == temp) is False:
            ax.plot(x, yvals.tolist(), label=mfname)
    summary.add_figure('Antecedent_Membership/{}'.format(var_name), fig, epoch+1)

def plot_all_mfs(model,summary,epoch):
    for i, (var_name, fv) in enumerate(model.layer.fuzzify.varmfs.items()):
        _plot_mfs(var_name, fv, model, summary,epoch)
