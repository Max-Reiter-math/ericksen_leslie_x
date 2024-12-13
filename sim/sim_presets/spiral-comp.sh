#!/bin/bash 
# experiments for comparison of discretization error and visualization
# CG
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -sid "spiral-comp-LhP" || true
python -m sim -m "Lh" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -sid "spiral-comp-Lh" || true
# DG
python -m sim -m "lpdg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.1 -sid "spiral-comp-lpdg-a1" || true
python -m sim -m "ldg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.1 -sid "spiral-comp-ldg-a1" || true
python -m sim -m "lpdg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.05 -sid "spiral-comp-lpdg-a05" || true
python -m sim -m "ldg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.05 -sid "spiral-comp-ldg-a05" || true
python -m sim -m "lpdg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.01 -sid "spiral-comp-lpdg-a01" || true
python -m sim -m "ldg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.01 -sid "spiral-comp-ldg-a01" || true
python -m sim -m "lpdg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-comp-lpdg-a005" || true
python -m sim -m "ldg" -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-comp-ldg--a005" 
# FP
python -m sim -m "FPhD" -e spiral -vtx -dh 10 -dt 0.0001 -tur 0 -fsr 0.01 -sid "spiral-comp-FPhD" || true
# tested
