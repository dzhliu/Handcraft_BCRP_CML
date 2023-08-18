#!/bin/sh
#SBATCH -o job_trainAttack.out
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:turing:1
#SBATCH --job-name=trainAttack

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

#Job name
#SBATCH --job-name=start

#Output file
#SBATCH --output=/home/nfs/yanqiqiao/backdoor-attacks-against-federated-learning-masteroutputs/%x.%j.out
module use /opt/insy/modulefiles
module load miniconda/3.9

# Your job commands go below here

#echo "Sourcing Ablation venv"
conda activate FLP37updated
echo -ne "Executing script "
echo $1
echo -ne "Running on node "
hostname
echo "Standard output:"

srun python train_attack.py --device='cuda:0' --benign_model_name='vgg11_fmnist_benign_bn_avgp.pt'

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
