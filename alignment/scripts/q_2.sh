#!/bin/bash

rm commands_2.txt 2>/dev/null
./run_2.sh
lines=$(wc -l < commands_2.txt)
echo "${lines} lines in commands_2.txt"
qsub -S /bin/bash -V -cwd -j y -N MWG_2 -t 1:${lines} script_wrapper_2.sh

qstat
read -rsp $'Press enter to continue...\n'
