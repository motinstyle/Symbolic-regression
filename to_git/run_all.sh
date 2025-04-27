#!/bin/bash

# Read the file line by line
while IFS= read -r line
do
    echo "Submitting: $line"
    
    # Submit the job to the SLURM scheduler
    sbatch --job-name=inverse_error \
            --nodes=1 \
            --output=/home/kubalik/inverse_error/jobs/%A.out \
            --error=/home/kubalik/inverse_error/jobs/%A.err \
            --partition=compute \
            --nodelist=node-01,node-02,node-03,node-04,node-05 \
            --time=01:00:00 \
            --cpus-per-task=2 \
            --mem=10G \
            --wrap="$line"
done < all_commands.txt