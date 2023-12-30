#!/bin/bash

#SBATCH --job-name=dinolocal                # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=piyush@robots.ox.ac.uk  # Where to send mail
#SBATCH --nodes=1                           # Node count
#SBATCH --ntasks=1                          # Total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # Number of CPU cores per task
#SBATCH --time=72:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=gpu                     # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:4                        # Requesting 4 GPUs
#SBATCH --mem=96000                         # 96GB of memory in MBs
#SBATCH --output=slurm_outputs/dinolocal_%j.log    # Standard output and error log
# -------------------------------

# List all modules that can be loaded
module avail
module load cuda/11.7
module load anaconda


# # Activate conda environment
# conda deactivate
# source activate py39-pt201-cu122

python -c "import torch; print(torch.__version__)"

# # Command to run
# data_dir=/scratch/shared/beegfs/piyush/datasets/PouringIROS2019/resized_data_cut_clips_frames/
# output_dir=/work/piyush/experiments/dino-local/vit_small_ps8/
# python -m torch.distributed.launch --nproc_per_node=4 main_dino.py \
#     --arch vit_small \
#     --patch_size 8 \
#     --data_path $data_dir \
#     --output_dir $output_dir \
#     --batch_size_per_gpu 16
