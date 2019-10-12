#!/bin/bash

read -p "Please enter your team name: " fullname
dname="$fullname$(date +"%y-%m-%d-%T")"
echo $dname
mkdir /home/nvidia/AI4AV/$fullname/DATA/$dname
python3 /home/nvidia/AI4AV/$fullname/record/record.py $dname
dir=(`ls /home/nvidia/AI4AV/$fullname/DATA/$dname/`)
if [ ${#dir[@]} -lt 2 ]
then rm -r /home/nvidia/AI4AV/$fullname/DATA/$dname
fi
