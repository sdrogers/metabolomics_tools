#!/bin/bash
echo $SGE_TASK_ID
OUTPUT=$(sed -n ${SGE_TASK_ID}p commands_2.txt)
echo $OUTPUT
eval $OUTPUT
