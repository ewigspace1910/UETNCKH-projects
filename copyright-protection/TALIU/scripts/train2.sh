#!/bin/bash
#SBATCH --job-name=tampered # define job name
#SBATCH --nodes=1             # define node
#SBATCH --gpus-per-node=1     # define gpu limmit in 1 node
#SBATCH --ntasks=1            # define number tasks
#SBATCH --cpus-per-task=8    # There are 24 CPU cores
#SBATCH --time=7-23:00:00     # Max running time = 10 minutes
#SBATCH --output=tamaug.out
#SBATCH --nodelist=node001


python xxx.py  "local_configs/metaseg/tiny/metaseg.tiny.512x512.doctamper.py" \
 --work-dir ../saves/test \
 --gpu-id 0 --seed 123 --tamaug --tamaug-rate 0.15 




