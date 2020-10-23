import numpy as np
import cirq
import tensorflow_quantum as tfq
from utilities.circuit_basics import Basic
import tensorflow as tf
import time

class VQE(Basic):
    def __init__(self, n_qubits=3, lr=0.01, epochs=100, patience=100, random_perturbations=True, verbose=0, g=1, J=0, problem="TFIM", noise_model=None):
        """

        lr: learning_rate for each iteration of gradient descent

        epochs: number of gradient descent iterations (in this project)

        patience: EarlyStopping parameter

        random_perturbations: if True adds to model's trainable variables random perturbations around (-pi/2, pi/
        problem: "ising", "xxz"

        noise_model: implemented in batches.
            if None: self.noise_model = False
            else: passed thorugh the Basic, to inherit the circuit_with_noise
                if should be in the form of {"channel":"depolarizing", "p":float, "q_batch_size":int}

        verbose: display progress or not

        Notes:

               (1) we add noise with probability %10

               (2) Hamiltonians:
               (2.1) &&ising model&& H = - g \sum_i \Z_i - (J) *\sum_i X_i X_{i+1}
               (2.2) &&xxz model$&&  H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
               (2.3) -- TODO --- import hamiltonian from .xyz file

               (3) Some callbacks are used: EarlyStopping and TimeStopping, which is set to 60*self.n_qubits (we can change this!)
        """

        super(VQE, self).__init__(n_qubits=n_qubits, noise_model=noise_model)

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.random_perturbations = random_perturbations
        self.verbose=verbose
        self.observable = self.give_observable(problem, g, J)
        self.max_time_training = 60*self.n_qubits
        self.hams = ["xxz", "TFIM"]
        if noise_model is None:
            self.noise=False
        else:
            self.noise=True
        self.gpus=tf.config.list_physical_devices("GPU")

    def give_observable(self,problem, g,J):
        if problem=="TFIM":
            return self.TFIM(g,J)
        elif problem=="xxz":
            return self.xxz_obs(g,J)
        else:
            print("please specify your hamiltonian.\n Available ones: "+str(self.hams))
            return
    def xxz_obs(self, g=1, J=0):
        """
        H = \sum_i^{n} X_i X_{i+1} + Y_i Y_{i+1} + J Z_i Z_{i+1} + g \sum_i^{n} \sigma_i^{z}
        """
        self.g = g
        self.J = J
        observable = [float(g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
            observable.append(cirq.Y.on(self.qubits[q])*cirq.Y.on(self.qubits[(q+1)%len(self.qubits)]))
            observable.append(float(J)*cirq.Z.on(self.qubits[q])*cirq.Z.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def TFIM(self, g=1, J=0):
        self.g=g
        self.J=J
        observable = [-float(g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def vqe(self, indexed_circuit, symbols_to_values=None):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)

        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]

        if self.noise is True, we've a noise model!
        Every detail of the noise model is inherited from circuit_basics
        """
        if self.noise is True:
            circuit, symbols, index_to_symbol = self.give_circuit_with_noise(indexed_circuit)
        else:
            circuit, symbols, index_to_symbol = self.give_circuit(indexed_circuit)
        model = self.TFQ_model(symbols, symbols_to_values=symbols_to_values)
        energy, training_history = self.train_model(circuit, model)
        final_params = model.trainable_variables[0].numpy()
        resolver = {"th_"+str(ind):var  for ind,var in enumerate(final_params)}
        return energy, resolver, training_history

    def give_energy(self, indexed_circuit, symbols_to_values):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)
        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]

        if self.noise is True, we've a noise model!
        Every detail of the noise model is inherited from circuit_basics
        """

        if self.noise is True:
            circuit, symbols, index_to_symbol = self.give_circuit_with_noise(indexed_circuit)
            tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(c, symbols_to_values)] for c in circuit)
            tfqcircuit = tf.squeeze(tfqcircuit)
            tfq_layer = tfq.layers.Expectation()(tfqcircuit, operators=tfq.convert_to_tensor([self.observable]*self.q_batch_size))
            averaged_unitaries = tf.math.reduce_mean(tfq_layer, axis=0)
            energy = np.squeeze(tf.math.reduce_sum(averaged_unitaries, axis=-1))

        else:
            circuit, symbols, index_to_symbol = self.give_circuit(indexed_circuit)
            tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(circuit, symbols_to_values)])
            tfq_layer = tfq.layers.Expectation()(tfqcircuit, operators=tfq.convert_to_tensor([self.observable]*self.q_batch_size))
            energy = np.squeeze(tf.math.reduce_sum(tfq_layer, axis=-1))
        return energy


    def TFQ_model(self, symbols, symbols_to_values=None):
        """
        symbols: continuous parameters to optimize on
        symbol_to_value: dictionary with initial seeds (if provided, not used at first run for instance)

        notice if self.noise is False self.q_batch_size=1
        """

        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        output = tfq.layers.Expectation()(
                circuit_input,
                symbol_names=symbols,
                operators=tfq.convert_to_tensor([self.observable]*self.q_batch_size),
                initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi))
        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=adam, loss='mse')

        if symbols_to_values:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))
        if self.random_perturbations:
            ### add noise only %10 of the times.
            if np.random.uniform()<.1:
                model.trainable_variables[0].assign( model.trainable_variables[0] + tf.random.uniform(model.trainable_variables[0].shape.as_list())*np.pi/2 )
        return model

    def test_circuit_qubits(self,circuit):
        """
        testing: if there's not parametrized unitary on every qubit, raise error (otherwise TFQ runs into trouble).
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


    def train_model(self, circuit, model):
        """
        circuit:
            if self.noise:
                circuit is a list of cirq.Circuit (parametrized with the symbols), with len(circuit)=self.q_batch_size
            else:
                cirq object with the parametrized unitaries unresolved (sympy symbols)
        model: TFQ_model output
        """

        if self.testing:
            self.test_circuit_qubits(circuit)

        if self.noise is True:
            tfqcircuit = tfq.convert_to_tensor(circuit)
        else:
            tfqcircuit = tfq.convert_to_tensor([circuit])

        qoutput = tf.ones((self.q_batch_size, 1))*self.lower_bound_Eg

        #### use a gpu if available (when running in colab for instace) ###
        if len(self.gpus)>0:
            with tf.device(self.gpus[0]):
                h=model.fit(x=tfqcircuit, y=qoutput, batch_size=self.q_batch_size, epochs=self.epochs,
                      verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience, mode="min", min_delta=10**-3), TimedStopping(seconds=self.max_time_training)])
        else:
            h=model.fit(x=tfqcircuit, y=qoutput, batch_size=self.q_batch_size, epochs=self.epochs,
                      verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience, mode="min", min_delta=10**-3), TimedStopping(seconds=self.max_time_training)])
        if self.noise is True:
            preds = model(tfqcircuit)
            averaged_unitaries = tf.math.reduce_mean(preds, axis=0)
            energy = np.squeeze(tf.math.reduce_sum(averaged_unitaries, axis=-1))
        else:
            energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
        return energy,h



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
