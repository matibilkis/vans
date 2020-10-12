import numpy as np
import cirq
import tensorflow_quantum as tfq
from circuit_basics import Basic
import tensorflow as tf

class VQE(Basic):
    def __init__(self, n_qubits=3, lr=0.01, epochs=100, patience=100, random_perturbations=True, verbose=0, g=1, J=0, noise=0.0):
        """
        lr: learning_rate for each iteration of gradient descent
        epochs: number of gradient descent iterations (in this project)
        patience: EarlyStopping parameter
        random_perturbations: if True adds to model's trainable variables random perturbations around (-pi/2, pi/2) with probability %10
        verbose: display progress or not

        &&ising model&& H = - g/2 \sum_i \Z_i - (J/2)*\sum_i X_i X_{i+1}
        """

        super(VQE, self).__init__(n_qubits=n_qubits)

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.random_perturbations = random_perturbations
        self.verbose=verbose
        self.observable = self.ising_obs(g=g, J=J)
        self.noise = float(noise)


    def ising_obs(self, g=1, J=0):
        self.g=g
        self.J=J
        observable = [-float(0.5*g)*cirq.Z.on(q) for q in self.qubits]
        for q in range(len(self.qubits)):
            observable.append(-float(0.5*J)*cirq.X.on(self.qubits[q])*cirq.X.on(self.qubits[(q+1)%len(self.qubits)]))
        return observable

    def vqe(self, indexed_circuit, symbols_to_values=None):
        """
        indexed_circuit: list with integers that correspond to unitaries (target qubit deduced from the value)

        symbols_to_values: dictionary with the values of each symbol. Importantly, they should respect the order of indexed_circuit, i.e. list(symbols_to_values.keys()) = self.give_circuit(indexed_circuit)[1]
        """
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
        """
        circuit, symbols, index_to_symbol = self.give_circuit(indexed_circuit)
        tfqcircuit = tfq.convert_to_tensor([cirq.resolve_parameters(circuit, symbols_to_values)])
        if self.noise > 0:
            tfq_layer = tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise)))(tfqcircuit, operators=tfq.convert_to_tensor([self.observable]))
        else:
            tfq_layer = tfq.layers.Expectation()(tfqcircuit, operators=tfq.convert_to_tensor([self.observable]))
        energy = np.squeeze(tf.math.reduce_sum(tfq_layer, axis=-1))
        return energy


    def TFQ_model(self, symbols, symbols_to_values=None):
        """
        symbols: continuous parameters to optimize on
        symbol_to_value: if not None, dictionary with initial seeds
        """

        circuit_input = tf.keras.Input(shape=(), dtype=tf.string)
        if self.noise > 0:
            output = tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(self.noise)))(
                        circuit_input,
                        symbol_names=symbols,
                        operators=tfq.convert_to_tensor([self.observable]),
                        initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi))
        else:
            output = tfq.layers.Expectation()(
                        circuit_input,
                        symbol_names=symbols,
                        operators=tfq.convert_to_tensor([self.observable]),
                        initializer=tf.keras.initializers.RandomUniform(minval=-np.pi, maxval=np.pi))
        model = tf.keras.Model(inputs=circuit_input, outputs=output)
        adam = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=adam, loss='mse')

        if symbols_to_values:
            model.trainable_variables[0].assign(tf.convert_to_tensor(np.array(list(symbols_to_values.values())).astype(np.float32)))
        if self.random_perturbations:
            ### actually add noise only %10 of the times.
            if np.random.uniform()<.1:
                model.trainable_variables[0].assign( model.trainable_variables[0] + tf.random.uniform(model.trainable_variables[0].shape.as_list())*np.pi/2 )
        return model

    def train_model(self, circuit, model):
        """
        circuit: cirq object with the parametrized unitaries unresolved (sympy symbols)
        model: TFQ_model output
        """

        ## testing: if there's not parametrized unitary on every qubit, raise error.
        effective_qubits = list(circuit.all_qubits())
        for k in self.qubits:
            if k not in effective_qubits:
                raise Error("NOT ALL QUBITS AFFECTED")

        tfqcircuit = tfq.convert_to_tensor([circuit])

        qoutput = tf.ones((1, 1))*self.lower_bound_Eg
        h=model.fit(x=tfqcircuit, y=qoutput, batch_size=1, epochs=self.epochs,
                  verbose=self.verbose, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patience, mode="min", min_delta=10**-3)])
        energy = np.squeeze(tf.math.reduce_sum(model.predict(tfqcircuit), axis=-1))
        return energy,h
