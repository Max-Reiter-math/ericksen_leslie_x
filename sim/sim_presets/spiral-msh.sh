#!/bin/bash 
# experiments for comparison of discretization error and visualization
# call by: 
#   sim/sim_presets/spiral-msh.sh
# 40,20,16,10,5
python input/spiral-msh.py -d 2 -dh 5 -xdmf &&
python input/spiral-msh.py -d 2 -dh 10 -xdmf &&
python input/spiral-msh.py -d 2 -dh 16 -xdmf &&
python input/spiral-msh.py -d 2 -dh 20 -xdmf &&
python input/spiral-msh.py -d 2 -dh 40 -xdmf

