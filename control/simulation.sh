#!/bin/bash

echo "Setting URDF Extra to="
export JACKAL_URDF_EXTRAS='/home/auvsl/catkin_woojin/online_rl/control/urdf/friction.urdf'
echo "Set JACKAL_URDF_EXTRAS=$JACKAL_URDF_EXTRAS"
# exit gracefully by returning a status

roslaunch launch/simulation.launch
exit 0

