#!/bin/bash
#SBATCH -p main
#SBATCH -N1
#SBATCH -n6
source .venv/bin/activate
python simulation.py --parallel --csv -T 2 -a 0.1 0.2 0.3 0.4 0.5 0.6 