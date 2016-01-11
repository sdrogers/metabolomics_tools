#!/bin/bash

rm commands_4.txt 2>/dev/null
./run_4.sh
lines=$(wc -l < commands_4.txt)
echo "${lines} lines in commands_4.txt"
qsub -S /bin/bash -V -cwd -j y -N MWG_4 -t 1:${lines} script_wrapper_4.sh

qstat
read -rsp $'Press enter to continue...\n'
