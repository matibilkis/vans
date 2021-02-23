from utilities.misc import dict_to_json
import os
bd=1.5
problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bd)), ('H', (0., 0., 2*bd)), ('H', (0., 0., 3*bd))], "multiplicity":1, "charge":0, "basis":"sto-3g"});QUBITS = 8

#data/uab-giq/scratch/matias/data-vans/
st = "python3 main.py --path_results \"../data-vans/\" --qlr 0.01 --acceptance_percentage 0.05 --n_qubits {} --reps 1000 --qepochs 1000 --problem_config {} --optimizer sgd".format(QUBITS,problem_config)

print(st)
#os.system(st)
