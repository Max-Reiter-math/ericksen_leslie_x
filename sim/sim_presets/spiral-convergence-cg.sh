#!/bin/bash 
# experiments for convergence analysis
# dt = 0.05
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.0500 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh10-dt0500-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 16 -dt 0.0500 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh16-dt0500-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 20 -dt 0.0500 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh20-dt0500-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 40 -dt 0.0500 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh40-dt0500-a005"  || true
# dt = 0.01
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.0100 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh10-dt0100-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 16 -dt 0.0100 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh16-dt0100-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 20 -dt 0.0100 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh20-dt0100-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 40 -dt 0.0100 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh40-dt0100-a005"  || true
# dt = 0.005
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.0050 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh10-dt0050-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 16 -dt 0.0050 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh16-dt0050-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 20 -dt 0.0050 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh20-dt0050-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 40 -dt 0.0050 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh40-dt0050-a005"  || true
# dt = 0.001
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.0010 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh10-dt0010-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 16 -dt 0.0010 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh16-dt0010-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 20 -dt 0.0010 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh20-dt0010-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 40 -dt 0.0010 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh40-dt0010-a005"  || true
# dt = 0.0005
python -m sim -m "LhP" -e spiral -vtx -dh 10 -dt 0.0005 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh10-dt0005-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 16 -dt 0.0005 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh16-dt0005-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 20 -dt 0.0005 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh20-dt0005-a005"  || true
python -m sim -m "LhP" -e spiral -vtx -dh 40 -dt 0.0005 -tur 0 -fsr 0.01 -a 0.005 -sid "spiral-LhP-dh40-dt0005-a005"  
# tested
