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

def tensorboard_plot(agent,i,summary, test_path, robot_path, control_law_save, dis_error, mae, rmse, best_mae, dis_error_max):

    summary.add_scalar("Error/Dist Error MAE", mae, i+1)
    summary.add_scalar("Error/Dist Error RMSE", rmse, i+1)
    summary.add_scalar("Error/Best Error So far", best_mae, i+1)
    summary.add_scalar("Error/Max Dist Error", dis_error_max, i+1)

    fig, ax = plt.subplots()
    ax.plot(test_path[:-1, 0], test_path[:-1, 1])
    ax.plot(robot_path[:, 0], robot_path[:, 1])
    ax.set_aspect('equal')
    summary.add_figure("Gazebo/Plot", fig, i+1)

    fig, ax = plt.subplots()
    ax.plot(np.array(list(range(1,len(control_law_save)+1))), np.array(control_law_save))

    summary.add_figure("Gazebo/Control_law", fig, i+1)

    fig, ax = plt.subplots()
    ax.plot(np.array(list(range(1,len(dis_error)+1))), np.array(dis_error))

    summary.add_figure("Gazebo/dis_errors", fig, i+1)
    summary.add_scalar("Distance_line/mf0/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].c, i+1)
    summary.add_scalar("Distance_line/mf0/d",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("Distance_line/mf1/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].c, i+1)
    summary.add_scalar("Distance_line/mf1/d",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf1'].d, i+1)
    summary.add_scalar("Distance_line/mf2/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].c, i+1)
    summary.add_scalar("Distance_line/mf2/d",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf2'].d, i+1)
    summary.add_scalar("Distance_line/mf3/b",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b, i+1)
    summary.add_scalar("Distance_line/mf3/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c, i+1)
    # summary.add_scalar("Distance_line/mf2/b",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c, i+1)
    # summary.add_scalar("Distance_line/mf2/c",agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d, i+1)

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

    summary.add_scalar("theta_lookahead/mf0/c",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf0'].c, i+1)
    summary.add_scalar("theta_lookahead/mf0/d",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("theta_lookahead/mf1/c",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf1'].c, i+1)
    summary.add_scalar("theta_lookahead/mf1/d",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf1'].d, i+1)
    summary.add_scalar("theta_lookahead/mf2/b",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf2'].b, i+1)
    summary.add_scalar("theta_lookahead/mf2/c",agent.actor.layer['fuzzify'].varmfs['theta_lookahead'].mfdefs['mf2'].c, i+1)

    summary.add_scalar("distance_target/mf0/d",agent.actor.layer['fuzzify'].varmfs['distance_target'].mfdefs['mf0'].d, i+1)
    summary.add_scalar("distance_target/mf1/a",agent.actor.layer['fuzzify'].varmfs['distance_target'].mfdefs['mf1'].a, i+1)
    critic = agent.critic
    l1 = critic.linear1
    l2 = critic.linear2
    l3 = critic.linear3

    for name, layer in zip(['l1', 'l2', 'l3'], [l1, l2, l3]):
        summary.add_histogram(f"{name}/bias", layer.bias, global_step=i+1)
        summary.add_histogram(f"{name}/weight", layer.weight, global_step=i+1)
