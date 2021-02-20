import pickle as np
from utilities.variational import VQE


vqe_handler = VQE(n_qubits=12, problem_config={"problem" : "XXZ", "g":1.0,"J":1.0})
