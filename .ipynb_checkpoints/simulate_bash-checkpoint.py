from utilities.misc import dict_to_json
import os

problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": 0.3})

st = "python3 main.py --path_results \"/data/uab-giq/scratch/matias/data-vans/\" --qlr 0.01 --acceptange_percentage 0.01 --n_qubits 2 --reps 10 --qepochs 1000 --problem_config {} --show_tensorboarddata 0 --optimizer sgd --training_patience 200".format(problem_config)

os.system(st)
