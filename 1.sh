#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short

# The default run (wall-clock) time is 1 minute
#SBATCH --time=04:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# The default number of CPUs per task is 1, however CPUs are always allocated per 2, so for a single task you should use 
#SBATCH --cpus-per-task=2

# The default memory per node is 1024 megabytes (1GB)
#SBATCH --mem=8GB

#SBATCH --gres=gpu:1

# Set mail type to 'END' to receive a mail when the job finishes (with usage statistics)
#SBATCH --mail-type=END

#SBATCH -o job.out

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Use this simple command to check that your sbatch settings are working (it should show the GPU that you requested)
/usr/bin/nvidia-smi

#Job name
#SBATCH --job-name=vgg11cifar10

#Output file
#SBATCH --output=/home/nfs/yanqiqiao/backdoor-attacks-against-federated-learning-masteroutputs/%x.%j.out
#module use /opt/insy/modulefiles
#module load cuda/11.1 cudnn/11.1-8.0.5.39 miniconda/3.9
#module list
module use /opt/insy/modulefiles
module load miniconda/3.9
#module load cuda/11.8

# Your job commands go below here

#echo "Sourcing Ablation venv"
#conda activate attack
#conda activate FLP37updated
conda activate yqFL

echo -ne "Executing script "
echo $1
echo -ne "Running on node "
hostname
echo "Standard output:"

srun python main.py --device='cuda:0' --batch_size=128 #--model='resnet11'

# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
