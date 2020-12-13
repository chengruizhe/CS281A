#!/bin/bash

#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=16 # number of cores per task

# I think gpu:4 will request 4 of any kind of gpu per node,
# and gpu:v100_32:8 should request 8 v100_32 per node
#SBATCH --gres=gpu:4
#SBATCH --nodelist=zanino# if you need specific nodes
#SBATCH -t 1-00:00 # time requested (D-HH:MM)

#SBATCH -D /home/eecs/chengruizhe/RRM
#SBATCH -o /home/eecs/chengruizhe/slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e /home/eecs/chengruizhe/slurm/slurm.%N.%j.err # STDERR


# print some info for context
pwd
hostname
date

echo starting job...

# activate your virtualenv
# source /data/drothchild/virtualenvs/pytorch/bin/activate
# or do your conda magic, etc.
source ~/.bashrc
conda activate py37

# python will buffer output of your script unless you set this
# if you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed
export PYTHONUNBUFFERED=1

# do ALL the research
srun python cifar.py -a resnet --depth 20 --epochs 60 --schedule 30 45 --gamma 0.1 --wd 1e-4 --gpu-id 0,1,2,3 --checkpoint checkpoints/cifar10/resnet-110


# print completion time
date