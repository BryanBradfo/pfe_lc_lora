#!/bin/bash
#PBS -N vitjob
#PBS -o /home/users/nus/t0934135/scratch/pfe_lc_lora/job_output_vit.txt
#PBS -e /home/users/nus/t0934135/scratch/pfe_lc_lora/job_error_vit.txt
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1
#PBS -q normal

source /home/users/nus/t0934135/miniconda3/etc/profile.d/conda.sh

conda activate py310

# Change to your specific directory
cd ${PBS_O_WORKDIR}

# Execute your Python script, replace 'your_script.py' with your actual script name
python /home/users/nus/t0934135/scratch/pfe_lc_lora/vith32pretrained.py
