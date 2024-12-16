#!/bin/bash 
# UNIT TEST TO LET EVERY EXPERIMENT RUN
# call by plugging 'bash sim/sim_presets/unittest.sh' into the terminal
# EXPECTED BEHAVIOUR: If all simulations run correctly all of their folders should show up in the output folder
# ---
# the following command makes sure the script exits if an error occurs
set -e
# SPIRAL
# DG METHODS 
python -m sim -m "lpdg" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-lpdg-spiral"
python -m sim -m "ldg" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-ldg-spiral"
# CG Methods
python -m sim -m "LhP" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -sid "unittest-LhP-spiral"
python -m sim -m "Lh" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -sid "unittest-Lh-spiral"
python -m sim -m "LL2P" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -sid "unittest-LL2P-spiral"
python -m sim -m "LL2" -e spiral -vtx -dh 5 -dt 0.001 -T 0.003 -sid "unittest-LL2-spiral"
# FP 
python -m sim -m "FPhD" -e spiral -vtx -dh 5 -dt 0.0001 -T 0.0003 -sid "unittest-FPhD-spiral"
python -m sim -m "FPL2D" -e spiral -vtx -dh 5 -dt 0.0001 -T 0.0003 -sid "unittest-FPL2D-spiral"
# SMOOTH 
# DG METHODS 
python -m sim -m "lpdg" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-lpdg-smooth"
python -m sim -m "ldg" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-ldg-smooth"
# CG Methods
python -m sim -m "LhP" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -sid "unittest-LhP-smooth"
python -m sim -m "Lh" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -sid "unittest-Lh-smooth"
python -m sim -m "LL2P" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -sid "unittest-LL2P-smooth"
python -m sim -m "LL2" -e smooth -vtx -dh 8 -d 2 -dt 0.001 -T 0.003 -sid "unittest-LL2-smooth"
# FP 
python -m sim -m "FPhD" -e smooth -vtx -dh 8 -d 2 -dt 0.0001 -T 0.0003 -sid "unittest-FPhD-smooth"
python -m sim -m "FPL2D" -e smooth -vtx -dh 8 -d 2 -dt 0.0001 -T 0.0003 -sid "unittest-FPL2D-smooth"
# ANNIHILATION 
# DG METHODS 
python -m sim -m "lpdg" -e annihilation_2_dg -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-lpdg-annihilation_2_dg"
python -m sim -m "ldg" -e annihilation_2_dg -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-ldg-annihilation_2_dg"
# CG Methods
python -m sim -m "LhP" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LhP-annihilation_2"
python -m sim -m "Lh" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-Lh-annihilation_2"
python -m sim -m "LL2P" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LL2P-annihilation_2"
python -m sim -m "LL2" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LL2-annihilation_2"
# FP 
python -m sim -m "FPhD" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.0001 -T 0.0003 -sid "unittest-FPhD-annihilation_2"
python -m sim -m "FPL2D" -e annihilation_2 -vtx -dh 8 -d 3 -dt 0.0001 -T 0.0003 -sid "unittest-FPL2D-annihilation_2"
# VELOCITY DRIVEN FLOW 
# DG METHODS 
python -m sim -m "lpdg" -e velocity_driven_flow_dg -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-lpdg-velocity_driven_flow_dg"
python -m sim -m "ldg" -e velocity_driven_flow_dg -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -a 0.1 -sid "unittest-ldg-velocity_driven_flow_dg"
# CG Methods
python -m sim -m "LhP" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LhP-velocity_driven_flow"
python -m sim -m "Lh" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-Lh-velocity_driven_flow"
python -m sim -m "LL2P" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LL2P-velocity_driven_flow"
python -m sim -m "LL2" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.001 -T 0.003 -sid "unittest-LL2-velocity_driven_flow"
# FP 
python -m sim -m "FPhD" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.0001 -T 0.0003 -sid "unittest-FPhD-velocity_driven_flow"
python -m sim -m "FPL2D" -e velocity_driven_flow -vtx -dh 8 -d 3 -dt 0.0001 -T 0.0003 -sid "unittest-FPL2D-velocity_driven_flow"
