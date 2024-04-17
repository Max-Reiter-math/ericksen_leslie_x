#!/bin/bash 
# experiments for comparison of discretization error and visualization
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LhP-st" -T 0.05 || true
python -m sim -m "Lh" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-Lh-st" -T 0.05 || true
python -m sim -m "LL2P" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LL2P-st" -T 0.05 || true
python -m sim -m "LL2" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LL2-st" -T 0.05 

