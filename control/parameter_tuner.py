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
from itertools import product
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

is_simulation = True

def fuzzy_error(curr, tar, future):
    global dis_error

    A = np.array([ [curr[1]-tar[1], tar[0]-curr[0]], [tar[0]-curr[0], tar[1]-curr[1]] ] )
    b = np.array([ [tar[0]*curr[1]-curr[0]*tar[1]], [x*(tar[0]-curr[0]) + y*(tar[1]-curr[1])] ])
    proj = np.matmul(inv(A),b)
    d = ( x-curr[0] )*(tar[1]-curr[1]) - (y-curr[1])*(tar[0]-curr[0])

    if ( d >0):
        side = 1
    elif ( d < 0):
        side = -1
    else:
        side = 0

    distanceLine= np.linalg.norm(np.array([x,y])-proj.T,2)*side                     ##########################check this
    dis_error.append(distanceLine)
    distance_target = ((tar[0] - x)**2 + (tar[1] - y)**2)**(0.5)

    farTarget = np.array( [0.9*proj[0] + 0.1*tar[0], 0.9*proj[1] + 0.1*tar[1]] )
    th1 = math.atan2(farTarget[1]-y, farTarget[0]-x)
    th2 = math.atan2(tar[1]-curr[1], tar[0]-curr[0])
    th3 = math.atan2(future[1]-tar[1], future[0]-tar[0])
    theta_far = th1 - currentAngle
    theta_near = th2 - currentAngle
    theta_lookahead = th3 - currentAngle
    theta_far = wraptopi(theta_far)
    theta_near = wraptopi(theta_near)


    return [distanceLine,theta_far,theta_near,theta_lookahead,distance_target]

def target_generator(path):
    global path_count
    global stop
    global path_length
    global t_M
    path_length = len(path) - 1
    pos_x = x
    pos_y = y

    current_point = np.array( path[path_count] )
    target = np.array( path[path_count + 1] )

    current_point = t_M @ np.append(current_point, 1)
    target = t_M @ np.append(target, 1)

    current_point = current_point[:-1]
    target = target[:-1]

    A = np.array([ [(current_point[1]-target[1]),(target[0]-current_point[0])], [(target[0]-current_point[0]), (target[1]-current_point[1])] ])
    b = np.array([ [(target[0]*current_point[1] - current_point[0]*target[1])], [(pos_x*(target[0]-current_point[0]) + pos_y*(target[1] - current_point[1]))] ])
    proj = np.matmul(inv(A),b)

    current_point = np.array( [ [current_point[0]],[current_point[1]] ] )
    target = np.array( [ [target[0]],[target[1]] ] )
    temp1 = proj-current_point      ####dot product
    temp2 = target - current_point
    projLen = (temp1[0]*temp2[0] + temp1[1]*temp2[1])  / np.linalg.norm(target - current_point,2)**2

    if (projLen > 1):
        path_count += 1
    if (path_count == path_length-1):
        stop = True

    if ( (path_count == (path_length-2)) or (path_count == (path_length -1)) ):
        curr = np.array(path[path_count])
        tar = np.array(path[path_count+1])
        future = np.array(path[path_count+1])
    else:
        curr = np.array(path[path_count])
        tar = np.array(path[path_count+1])
        future = np.array(path[path_count+2])

    # if self.transform is not None:
    curr = t_M @ np.append(curr, 1)
    tar = t_M @ np.append(tar, 1)
    future = t_M @ np.append(future, 1)

    curr = curr[:-1]
    tar = tar[:-1]
    future = future[:-1]


    return curr, tar, future

def callback(msg):
    global x
    global y
    global q1
    global q2
    global q3
    global q4
    global currentAngle
    global robot_path
    global stop
    global done
    global currentAngle
    global firstPoseTrigger
    global battery_status
    global is_simulation
    firstPoseTrigger = True
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q2 = msg.pose.pose.orientation.x
    q3 = msg.pose.pose.orientation.y
    q4 = msg.pose.pose.orientation.z
    q1 = msg.pose.pose.orientation.w
    currentAngle = math.atan2(2*(q1*q4+q2*q3),1-2*(q3**2+q4**2))
    if is_simulation == False:
        battery_status = 0
    if stop == False:
        robot_path.append([x,y])
    # print('x position: ',x)
    # print('y position: ',y)

def agent_update(new_state, linear_velocity, control_law, agent, done, batch_size, curr_dis_error):
    rewards = reward(new_state, linear_velocity, control_law)/15.
    ####do this every 0.075 s
    state = agent.curr_states
    new_state = np.array(new_state)
    agent.curr_states = new_state

    # last_10_dis_error = np.mean(dis_error[-20:])
    # if abs(curr_dis_error) > 0.125:
    agent.memory.push(state,control_law,rewards,new_state,done)   ########control_law aftergain or before gain?
    if len(agent.memory) > batch_size and abs(curr_dis_error) > 0.10:
        agent.update(batch_size)



def path_transform():
    global x,y
    global currentAngle
    global t_M
    global in_M
    pose = (x,y)
    theta = currentAngle
    print(pose, theta)


    T = np.array([
        [1, 0, pose[0]],
        [0, 1, pose[1]],
        [0, 0, 1],
    ])

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])

    t_M = T @ R

    T = np.array([
        [1, 0, -pose[0]],
        [0, 1, -pose[1]],
        [0, 0, 1],
    ])

    R = np.array([
        [np.cos(-theta), -np.sin(-theta), 0],
        [np.sin(-theta), np.cos(-theta), 0],
        [0, 0, 1],
    ])

    in_M = R @ T
    #return transform, inverse_transform

def inverse_transform_poses(robot_path):
    poses = []

    for p in robot_path:
        p = np.array([*p, 1])
        poses.append(in_M @ p)

    return poses

def wait_pose():
    print("Waiting for initial pose")
    counter = 0
    while not firstPoseTrigger:
        counter += 1
        if counter % 100 == 0:
            print("Waiting for robot initial pose")
        rospy.sleep(1/60.)
    print("found initial pose", (x, y), currentAngle)



if __name__ == '__main__':
    parameter_config = {
        'actor_lr': [1e-4, 1e-5],
        'critic_lr': [1e-3, 1e-4, 1e-5],
        'hidden_size': [8, 16, 32, 64],
        # 'actor_decay': [1, 0.95, 0.9],
        # 'critic_decay': [1, 0.95, 0.9],
        'batch_size': [64,128]
        # 'batch_size': [16, 32, 64, 128]
    }


    param_names = list(parameter_config.keys())
    # zip with parameter names in order to get original property
    param_values = (zip(param_names, x) for x in product(*parameter_config.values()))

    for paramset in param_values:
        # use the dict from iterator of tuples constructor
        kwargs = dict(paramset)

        if kwargs['critic_lr'] > kwargs['actor_lr']:
            print(kwargs)
            parameters = kwargs
            name = ','.join([f"{k}={v}" for k, v in parameters.items()])
            gamma = 0.9
            tau = 1e-3
            summary = SummaryWriter(f'/home/auvsl/catkin_woojin/tensorboard_storage/hyper_tuner/{name}')
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

            # name = f'Gazebo RL {datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

            #For Desktop
            # summary = SummaryWriter(f'/home/auvsl/catkin_woojin/tensorboard_storage/{name}')
            #For jackal
            # summary = SummaryWriter(f'/home/nvidia/catkin_ws/src/woojin/jackal/control/figures/{name}')

            wait_pose()
            best_mae = 10

            epoch = 150
            vel_gain = 1.0
            path_tranform_enable = True
            linear_velocity = 1.5
            update_rate = 10

            test_path = test_course3()
            test_path.append([100,0])

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
                        agent_update(new_state, linear_velocity, control_law, agent, done, parameters['batch_size'], new_state[0])
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
                # print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].a)
                # print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].b)
                # print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].c)
                # print(agent.actor.layer['fuzzify'].varmfs['distance_line'].mfdefs['mf3'].d)

                rmse = np.sqrt(np.mean(np.power(dis_error, 2)))
                print("RMSE", rmse)

                test_path = np.array(test_path)
                robot_path = np.array(inverse_transform_poses(robot_path))

                tensorboard_plot(agent, i, summary, test_path, robot_path, control_law_save, dis_error, mae, rmse, best_mae)
                plot_all_mfs(agent.actor, summary, i)
                plot_mamdani(agent.actor, summary, i)

                # torch.save(agent,'models/anfis_ddpg_trained{}.model'.format(i+1))

                test_path = test_course3()
                # for i in range(len(test_path)):
                #     test_path[i][0] = test_path[i][0] / 1.25
                #     test_path[i][1] = test_path[i][1] / 1.25
                test_path.append([100,0])
