import numpy as np
import cirq
import tensorflow_quantum as tfq
from utilities.circuit_basics import Basic
import tensorflow as tf
import time
from utilities.chemical import ChemicalObservable
from utilities.qmodels import *

class VQE(Basic):
    def __init__(self, n_qubits=3, lr=0.01, optimizer="sgd", epochs=1000, patience=200,
                random_perturbations=True, verbose=0, noise_config={}, problem_config={}):
        """
        lr: learning_rate for each iteration of gradient descent
        optimizer: we give two choices, Adam and SGD. If SGD, we implement Algorithm 4 of qacq to adapt learning rate.
        epochs: number of gradient descent iterations (in this project)
        patience: EarlyStopping parameter
        random_perturbations: if True adds to model's trainable variables random perturbations around (-pi/2, pi/
        verbose: display progress or not

        noise_config:see circuit_basics (inherits properties from the circuit_with_noise)

        problem_config: dictionary that specifies the structure of the hamiltonian. Its keys will depend on the problem.
                        condensed matter:
                                problem_config["problem"] in ["XXZ", "TFIM"]
                                problem_config["g"]
                                problem_config["J"]
                        chemical:
                                problem_config["problem"] in [{molecule_name}] (for now should be H2)
                                problem_config["geometry"]
                                problem_config["charge"] (optional)
                                problem_config["multiplicity"] (optional)
                                problem_config["basis"]  (optional)
        Notes:
               (1) we add noise to parameters with probability %10
               (2) Hamiltonians:
               (2.1) &&ising model&& H = - g \sum_i \Z_i - (J) *\sum_i X_i X_{i+1}
               (2.2) &&xxz model$&&  H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
               (2.3) molecular hamiltonians, see chemical.py
               (3) Some callbacks are used: EarlyStopping and TimeStopping.
               (4) Note that we construct the circuit according to the number of qubits required, we should add a bool check in case circuit's qubits are not enough in chemical.py
        """

        super(VQE, self).__init__(n_qubits=n_qubits, noise_config=noise_config)


        #### MACHINE LEARNING CONFIGURATION
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.random_perturbations = random_perturbations
        self.verbose=verbose
        self.max_time_training = 300 #let's give 5 minutes for each circuit to be trained, at most
        self.gpus=tf.config.list_physical_devices("GPU")
        self.optimizer = {"ADAM":tf.keras.optimizers.Adam, "SGD":tf.keras.optimizers.SGD}[optimizer.upper()]
        self.repe=0 #this is to have some control on the number of VQEs done (for tensorboard)
        ##### HAMILTONIAN CONFIGURATION
        self.observable = self.give_observable(problem_config)

        #### NOISE CONFIGURATION
        ### this is inherited from circuit_basics: self.noise, self.q_batch_size

    def give_observable(self,problem_config):
        """
        problem_config: dictionary that specifies the structure of the hamiltonian. Its keys will depend on the problem.
                        condensed matter:
                                problem_config["problem"] in ["XXZ", "TFIM"]
                                problem_config["g"]
                                problem_config["J"]
                        chemical:
                                problem_config["problem"] in [{molecule_name}] (for now should be H2)
                                problem_config["geometry"]
                                problem_config["charge"] (optional)
                                problem_config["multiplicity"]  (optional)
                                problem_config["basis"]  (optional)
        """
        with open("utilities/hamiltonians/cm_hamiltonians.txt") as f:
            hams = f.readlines()
        possible_hamiltonians = [x.strip().upper() for x in hams]

        with open("utilities/hamiltonians/chemical_hamiltonians.txt") as f:
            hams = f.readlines()
        possible_hamiltonians += ([x.strip().upper() for x in hams])

        if problem_config["problem"] not in possible_hamiltonians:
            raise NameError("Hamiltonian {} is not invited to VANS yet. Available hamiltonians: {}\n".format(problem_config["problem"],possible_hamiltonians))

        #### CONDENSED MATTER HAMILTONIANS ####
        if problem_config["problem"].upper() in ["XXZ","TFIM"]:
            for field in ["g","J"]:
                if field not in problem_config.keys():
                    raise ValueError("You have not specified the fields correctly. Check out your problem_config back again. Current dict: {}".format(problem_config))
            if problem_config["problem"].upper()=="TFIM":
                #H = -J \sum_i^{n} X_i X_{i+1} - g \sum_i^{n} Z_i
                observable = [-float(problem_config["g"])*cirq.Z.on(q) for q in self.qubits]
                for q in range(len(self.qubits)):
                    observable.append(-float(problem_config["J"])*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
                return observable
            elif problem_config["problem"].upper()=="XXZ":
                #H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
                observable = [float(problem_config["g"])*cirq.Z.on(q) for q in self.qubits]
                for q in range(len(self.qubits)):
                    observable.append(cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
                    observable.append(cirq.Y.on(self.qubits[q])*cirq.Y.on(self.qubits[(q+1)%len(self.qubits)]))
                    observable.append(float(problem_config["J"])*cirq.Z.on(self.qubits[q])*cirq.Z.on(self.qubits[(q+1)%len(self.qubits)]))
                return observable

        elif problem_config["problem"].upper() in ["H2"]:
            oo = ChemicalObservable()
            for key,defvalue in zip(["geometry","multiplicity", "charge", "basis"], [None,1,0,"sto-3g"]):
                if key not in list(problem_config.keys()):
                    raise ValueError("{} not specified in problem_config. Dictionary obtained: {}".format(key, problem_config))
            observable=oo.give_observable(self.qubits, problem_config["geometry"], problem_config["multiplicity"], problem_config["charge"], problem_config["basis"])
            return observable
        else:
            raise NotImplementedError("The specified hamiltonian is in the list but we have not added to the code yet! Devs, take a look here!\problem_config[problem]: {}".format(problem_config["problem"].upper()))


    def vqe(self, indexed_circuit, symbols_to_values=None):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)

        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]

        if self.noise is True, we've a noise model!
        Every detail of the noise model is inherited from circuit_basics
        """
        if self.noise is False:
            circuit, symbols, index_to_symbol = self.give_circuit(indexed_circuit)
            tfqcircuit = tfq.convert_to_tensor([circuit])
        else:
            circuit, symbols, index_to_symbol = self.give_circuit_with_noise(indexed_circuit)
            tfqcircuit = tfq.convert_to_tensor(circuit)

        model = QNN(symbols=symbols, observable=self.observable, batch_sizes=self.q_batch_size)
        model(tfqcircuit) #this defines the weigths
        model.compile(optimizer=self.optimizer(lr=self.lr), loss=EnergyLoss())

        #in case we have already travelled the parameter space,
        if symbols_to_values:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))

        ### add noise only %10 of the times.
        if self.random_perturbations:
            if np.random.uniform()<.1:
                model.trainable_variables[0].assign(model.trainable_variables[0] + tf.random.uniform(model.trainable_variables[0].shape.as_list())*0.01)

        calls=[tf.keras.callbacks.EarlyStopping(monitor='energy', patience=self.patience, mode="min", min_delta=10**-3),TimedStopping(seconds=self.max_time_training)]

        if hasattr(self, "tensorboarddata"):
            self.repe+=1
            calls.append(tf.keras.callbacks.TensorBoard(log_dir=self.tensorboarddata+"/logs/{}".format(self.repe)))

        if len(self.gpus)>0:
            with tf.device(self.gpus[0]):
                training_history = model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)), verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)
        else:
            training_history = model.fit(x=tfqcircuit, y=tf.zeros((self.q_batch_size,)),verbose=self.verbose, epochs=self.epochs, batch_size=self.q_batch_size, callbacks=calls)

        energy = model.cost_value.result()
        final_params = model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return energy, resolver, training_history


    def test_circuit_qubits(self,circuit):
        """
        This function is only for testing. If there's not parametrized unitary on every qubit, raise error (otherwise TFQ runs into trouble).
        """
        if self.noise is True:
            effective_qubits = list(circuit[0].all_qubits())
            for k in self.qubits:
                if k not in effective_qubits:
                    raise Error("NOT ALL QUBITS AFFECTED")
        else:
            effective_qubits = list(circuit[0].all_qubits())
            for k in self.qubits:
                if k not in effective_qubits:
                    raise Error("NOT ALL QUBITS AFFECTED")

class TimedStopping(tf.keras.callbacks.Callback):
    '''Stop training when enough time has passed.

        # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self, seconds=None, verbose=1):
        super(TimedStopping, self).__init__()
        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose>0:
                print('Stopping after %s seconds.' % self.seconds)
