#!/bin/bash 
# MULTIPLE EXPERIMENT SETUP
# call by plugging 'bash sim/sim_presets/annihilation.sh' into the terminal
# ---
# the following command makes sure the script exits if an error occurs
set -e
# ANNIHILATION, defaults: "dim" : 3, "dh" : 2**4, "dt" : 0.0005, "T" : 0.1
# DG METHODS 
python -m sim -m "lpdg" -e annihilation_2 -vtx -d 3 -dt 0.001 -a 0.1 -fsr 0.005 -sid "comp-lpdg-annihilation_2"
# CG Methods
python -m sim -m "LhP" -e annihilation_2 -vtx -d 3 -dt 0.001 -fsr 0.005 -sid "comp-LhP-annihilation_2"
# VELOCITY DRIVEN FLOW "dim" : 3, "dh" : 2**4, "dt" : 0.0005, "T" : 2.0
# DG METHODS 
python -m sim -m "lpdg" -e velocity_driven_flow -vtx -d 3 -dt 0.01 -a 0.1 -fsr 0.005 -sid "comp-lpdg-velocity_driven_flow"
# CG Methods
python -m sim -m "LhP" -e velocity_driven_flow -vtx -d 3 -dt 0.01 -fsr 0.005 -sid "comp-LhP-velocity_driven_flow"
#--------------
# FP 
python -m sim -m "FPhD" -e annihilation_2 -vtx -d 3 -fsr 0.005 -sid "comp-FPhD-annihilation_2"
# FP 
python -m sim -m "FPhD" -e velocity_driven_flow -vtx -d 3 -fsr 0.005 -sid "comp-FPhD-velocity_driven_flow"
