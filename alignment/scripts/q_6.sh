#!/bin/bash

rm commands_6.txt 2>/dev/null
./run_6.sh
lines=$(wc -l < commands_6.txt)
echo "${lines} lines in commands_6.txt"
qsub -S /bin/bash -V -cwd -j y -N MWG_6 -t 1:${lines} script_wrapper_6.sh

qstat
read -rsp $'Press enter to continue...\n'
