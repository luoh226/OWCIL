#!/bin/bash

#SBATCH --mem=16G                # memory
#SBATCH --gres=gpu:1             # Number of GPU(s): 1 for DTW, 3 for Feature extract.
#SBATCH --time=10-00:00:00       # time (DD-HH:MM:SS) 3 days by default; 5-00:00:00 (5 DAYS) / UNLIMITED;
#SBATCH --ntasks=1               # Number of "tasks”/#nodes (use with distributed parallelism).
#SBATCH --cpus-per-task=4        # Number of CPUs allocated to each task/node (use with shared memory parallelism).

# DTW Templates

hostname
whoami
echo "//////////////////////////////"

echo 'CUDA_VISIBLE_DEVICES:'
echo $CUDA_VISIBLE_DEVICES
echo "//////////////////////////////"

#source activate testenv
#echo "Loaded Env."

dataset=cifar100
k_fold=3
num_workers=2
init_cls=10
increment=10
seed=1993,1999,2024

#for method in ewc lwf pass il2a ssre fetril fecam;
for method in fecam;
do
for ood_ratio in 0.0 0.2 0.4 0.6 0.8 1.0;
do
# train model
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=false dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
# get thresh
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=true ood_method="MSP" dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=true ood_method="ENERGY" dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
## inference & evaluation
python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="MSP" ood_ratio=${ood_ratio} dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="ENERGY" ood_ratio=${ood_ratio} dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="None" ood_ratio=${ood_ratio} dataset=${dataset} k_fold=${k_fold} num_workers=${num_workers} init_cls=${init_cls} increment=${increment} seed=${seed}
done
done
