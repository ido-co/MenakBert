#! /bin/sh
#SBATCH --job-name=menakbert
#SBATCH --output=menakbert2.out
#SBATCH --error=menakbert2.err
#SBATCH --partition=killable
#SBATCH --time=500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --exclude=n-804
#SBATCH --export=ALL

python main.py