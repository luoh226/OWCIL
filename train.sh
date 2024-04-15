
seed=1993
method=ewc
# train model
python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=false seed=${seed}
# get thresh
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=true ood_method="MSP"
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=false get_thresh=true ood_method="ENERGY"
## inference & evaluation
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="MSP"
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="ENERGY"
#python -u main.py --config=./exps/${method}.json --cfg-options eval_only=true get_thresh=false ood_method="None"
