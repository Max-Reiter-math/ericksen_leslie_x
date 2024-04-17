#!/bin/bash 
# experiments for comparison of discretization error and visualization
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LhP-st1" || true
python -m sim -m "Lh" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-Lh-st1" || true
python -m sim -m "LL2P" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LL2P-st1" || true
python -m sim -m "LL2" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-LL2-st1" || true
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.001 -tur 0 -fsr 0.05 -sid "spiral-LhP-st2" || true
python -m sim -m "Lh" -e spiral -vtx -dh 10 -dt 0.001 -tur 0 -fsr 0.05 -sid "spiral-Lh-st2" || true
python -m sim -m "LL2P" -e spiral -vtx -dh 10 -dt 0.001 -tur 0 -fsr 0.05 -sid "spiral-LL2P-st2" || true
python -m sim -m "LL2" -e spiral -vtx -dh 10 -dt 0.001 -tur 0 -fsr 0.05 -sid "spiral-LL2-st2"
# tested
