#!/bin/bash 
# Used as reference solutions
python -m sim -m FPhD -e smooth -cp -vtx -dh 80 -dt 0.0001 -tur 0 -fsr 0.05 -sid "smooth-fp-dh-80-dt-0001" || true
python -m sim -m FPhD -e spiral -vtx -dh 10 -dt 0.0005 -tur 0 -fsr 0.05 -sid "spiral-fp-dh-10-dt-0005" || true
python -m sim -m FPhD -e spiral -vtx -dh 10 -dt 0.005 -tur 0 -fsr 0.05 -sid "spiral-fp-dh-10-dt-005" || true
python -m sim -m FPhD -e spiral -vtx -dh 10 -dt 0.01 -tur 0 -fsr 0.05 -sid "spiral-fp-dh-10-dt-01"
# tested