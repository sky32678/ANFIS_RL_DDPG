# Train ANFIS using Deep Deterministic Policy Gradient in ROS

## The environment:

Pytorch

ROS Melodic (Ubuntu 18.04) or ROS Noetic (Ubuntu 20.04)

If you use ROS Melodic, you should use Python 3

##Tensorboard
 
### If you run it on the Vehicle with Laptop
`ssh -L 16007:127.0.0.1:16007 USER_NAME@IP_ADDRESS`
`tensorboard --logdir=figures --samples_per_plugin images=999 --port=16007`

### Local Machine
`tensorboard --logdir=figures --samples_per_plugin images=999`

## Run Simulation
`bash simulation.sh`
to open jackal simulator in Gazebo

and run

`python main.py`
