#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
. /etc/profile
srun echo $(seq 2500 5000) | xargs -P 8 -n 1 bash attack.sh ER 64000 5.00 DegU