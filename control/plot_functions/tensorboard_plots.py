import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import math
import torch
import anfis_codes.anfis
import os
from utils.utils import reward, angdiff, wraptopi
from utils.path import test_course, test_course2, test_course3
from torch.utils.tensorboard import SummaryWriter
from plot_functions.plots import plot_mamdani, _plot_mfs, plot_all_mfs

def tensorboard_plot(agent,i,summary):
    summary.add_scalar("Distance_line/mf0/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c, i+1)
    summary.add_scalar("Distance_line/mf0/d",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("Distance_line/mf1/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].c, i+1)
    summary.add_scalar("Distance_line/mf1/d",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].d, i+1)
    summary.add_scalar("Distance_line/mf2/b",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].b, i+1)
    summary.add_scalar("Distance_line/mf2/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c, i+1)

    summary.add_scalar("theta_far/mf0/c",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].c, i+1)
    summary.add_scalar("theta_far/mf0/d",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("theta_far/mf1/c",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].c, i+1)
    summary.add_scalar("theta_far/mf1/d",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf1'].d, i+1)
    summary.add_scalar("theta_far/mf2/b",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].b, i+1)
    summary.add_scalar("theta_far/mf2/c",agent.actor.layer['fuzzify'].varmfs['theta_far'].mfdefs['mf2'].c, i+1)

    summary.add_scalar("theta_near/mf0/c",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].c, i+1)
    summary.add_scalar("theta_near/mf0/d",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("theta_near/mf1/c",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].c, i+1)
    summary.add_scalar("theta_near/mf1/d",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf1'].d, i+1)
    summary.add_scalar("theta_near/mf2/b",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].b, i+1)
    summary.add_scalar("theta_near/mf2/c",agent.actor.layer['fuzzify'].varmfs['theta_near'].mfdefs['mf2'].c, i+1)
