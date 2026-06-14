#!/bin/bash
module load nvhpc

for i in 1000 2000 5000 10000 20000 50000 100000
do
   ./bin/molDyn-cuda molDyn.par  $i >> log.txt
done
# ./bin/molDyn molDyn.par