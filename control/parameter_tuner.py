import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import torch
import time
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import torch
import anfis_codes.anfis
import os
import datetime
from rl.ddpg import DDPGagent
from rl.memory import *
from anfis_codes.model import *
from utils.utils import reward, angdiff, wraptopi
from utils.path import test_course, test_course2, test_course3
from torch.utils.tensorboard import SummaryWriter
from plot_functions.plots import plot_mamdani, _plot_mfs, plot_all_mfs
from plot_functions.tensorboard_plots import tensorboard_plot
matplotlib.use('Agg')
dtype = torch.float

x = 0.0
y = 0.0
q1 = 0.0
q2 = 0.0
q3 = 0.0
q4 = 0.0
currentAngle = 0.0
control_law = 0.0

stop = False
path_count = 0
done = False
robot_path = []
dis_error = []
firstPoseTrigger = False


def sequence(parameters, path):
    name = ','.join([f"{k}={v}" for k, v in parameters.items()])
    gamma = 0.9
    tau = 1e-3
    summary = SummaryWriter(f'/home/auvsl/catkin_woojin/tensorboard_storage/{name}')
    # os.mkdir(os.path.join(summary.get_logdir(), 'checkpoints'))
    anf = Anfis().my_model()
    #print(env.action_space.shape)
    #env = gym.make('CartPole-v1')
    num_inputs, num_outputs = 5, 1
    agent = DDPGagent(num_inputs, num_outputs, anf, parameters['hidden_size'], parameters['actor_lr'], parameters['critic_lr'], gamma, tau)
    # agent = DDPGAgent(5, 1, many_error_predefined_anfis_model(), critic_learning_rate=parameters['critic_lr'],
    #                   hidden_size=parameters['hidden_size'], actor_learning_rate=parameters['actor_lr'])
    rospy.init_node('check_odometry')
    # sub = rospy.Subscriber("/odom", Odometry, callback)
    sub = rospy.Subscriber("/odometry/filtered", Odometry, callback)
    pub = rospy.Publisher("/cmd_vel",Twist,queue_size =10)
    timer = 0

    name = f'Gazebo RL {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    #For Desktop
    summary = SummaryWriter(f'/home/auvsl/catkin_woojin/tensorboard_storage/{name}')
    #For jackal
    # summary = SummaryWriter(f'/home/nvidia/catkin_ws/src/woojin/jackal/control/figures/{name}')

    wait_pose()
    best_mae = 10

    epoch = 100
    vel_gain = 1.0
    path_tranform_enable = True
    linear_velocity = 1.5
    update_rate = 10

    for i in range(epoch):
        robot_path = []
        dis_error = []
        control_law_save = []
        stop = False
        path_count = 0

        if path_tranform_enable:
            path_transform()

        while not rospy.is_shutdown():
            ###Wait untill publisher gets connected
            while not pub.get_num_connections() == 1:
                # print(pub.get_num_connections())
                pass

            current_point, target_point, future_point = target_generator(test_path)

            if stop == True:
                print("STOP")
                # os.system('rosservice call /gazebo/reset_world "{}"')
                # rospy.sleep(0.05)
                # os.system('rosservice call /set_pose "{}"')
                # rospy.sleep(0.05)
                break

            new_state = fuzzy_error(current_point, target_point, future_point)

            control_law = agent.get_action(np.array(new_state))
            control_law = control_law.item()*vel_gain

            if (control_law > 4.):
                control_law = 4.
            if (control_law < -4.):
                control_law = -4.

            twist_msg = Twist()
            twist_msg.linear.x = linear_velocity
            twist_msg.angular.z = control_law

            if timer % update_rate == 0:
                agent_update(new_state, linear_velocity, control_law, agent, done, batch_size, new_state[0])
                timer = 0

            pub.publish(twist_msg)
            rospy.sleep(0.001)
            timer += 1
            control_law_save.append(control_law)

        print("Epoch", i+1)

        mae = np.mean(np.abs(dis_error))
        print("MAE", mae)
        if best_mae > mae:
            best_mae = mae
        print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].a)
        print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b)
        print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c)
        print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d)

        rmse = np.sqrt(np.mean(np.power(dis_error, 2)))
        print("RMSE", rmse)

        test_path = np.array(test_path)
        robot_path = np.array(inverse_transform_poses(robot_path))

        tensorboard_plot(agent, i, summary, test_path, robot_path, control_law_save, dis_error, mae, rmse, best_mae)
        plot_all_mfs(agent.actor, summary, i)
        plot_mamdani(agent.actor, summary, i)

        torch.save(agent,'models/anfis_ddpg_trained{}.model'.format(i+1))

        test_path = test_course3()
        # for i in range(len(test_path)):
        #     test_path[i][0] = test_path[i][0] / 1.25
        #     test_path[i][1] = test_path[i][1] / 1.25
        test_path.append([100,0])


if __name__ == '__main__':
    parameter_config = {
        'critic_lr': [1e-4, 1e-5, 1e-6],
        'actor_lr': [1e-3, 1e-4, 1e-5],
        'hidden_size': [8, 16, 32, 64],
        # 'actor_decay': [1, 0.95, 0.9],
        # 'critic_decay': [1, 0.95, 0.9],
        'batch_size': [16, 32, 64, 128]
    }

    test_path = test_course3()

    param_names = list(parameter_config.keys())
    # zip with parameter names in order to get original property
    param_values = (zip(param_names, x) for x in product(*parameter_config.values()))

    for paramset in param_values:
        # use the dict from iterator of tuples constructor
        kwargs = dict(paramset)

        if kwargs['critic_lr'] > kwargs['actor_lr']:
            print(kwargs)
            sequence(kwargs, path)
