import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime

# N=1000
# episodes = np.arange(1,N,1)
# tt = .75*len(episodes)/np.log(1/0.05)
# def schedule(k):
#     return max(0.05, np.exp(-k/tt))


class Duel_DQN:
    def __init__(self, env, use_tqdm=False, learning_rate = 0.01,
        size_rp=10**5, name="DueDQN", policy="exp-decay", ep=0.01, tau=0.1):

        self.env = env
        self.n_actions = len(self.env.solver.alphabet)
        self.use_tqdm=use_tqdm
        ### define qnets ###
        self.prim_qnet = self.build_q_network(learning_rate = learning_rate)
        self.target_qnet = self.build_q_network()
        self.tau = tau #how soft the update is

        ### policy ##
        self.policy = policy
        self.ep0 = ep
        self.exp_decay_effective_explotation = 0.5 #percentage of time at which ep(t0) = \ep0 with #ep(t) = \ep0 exp[- t / t0]

        self.name = name

        ### define buffer ####
        state_shape = self.env.depth_circuit #we will modify this.
        self.replay_buffer = ReplayBuffer(state_shape, size=size_rp, use_per=True)

        ### some info to save ###
        self.info = "len(alphabet): {}\nalphabet_gates: {}, \nobservable_name: {}\ndepth_circuit: {}\nn_qubits: {}\nname: {}\nlr: {}\n\npolicy: {}\nep0: {}\nexp_decay_effective_explotation: {}\n\nstate_shape: {}\nbuffer_size: {}".format(len(self.env.solver.alphabet),self.env.solver.alphabet_gates,
        self.env.solver.observable_name, self.env.depth_circuit, self.env.n_qubits,
        self.name, learning_rate, self.policy, self.ep0, self.exp_decay_effective_explotation,
        state_shape, self.replay_buffer.size )


    def build_q_network(self, learning_rate=0.01):
        '''this function creates the q network
        using architecture of https://arxiv.org/pdf/1511.06581.pdf

        In particular takes as input a sequence of gates borrowed from the alphabet (-1 means no gate yet)
        Notice we normalize the input with this Lambda layer.
        '''

        model_input = Input(shape=(self.env.state_shape))
        x = Lambda(lambda layer: layer / self.n_actions)(model_input)

        x = Dense(64, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(model_input)
        x = Dense(64, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
        x = Dense(64, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

        val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 1))(x)  # custom splitting layer

        val_stream = Flatten()(val_stream)
        val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

        adv_stream = Flatten()(adv_stream)
        adv = Dense(self.n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

        reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
        q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

        model = Model(model_input, q_vals)
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model


    def update_target_parameters(self, tau=0.01):
        prim_weights = self.prim_qnet.get_weights()
        targ_weights = self.target_qnet.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.target_qnet.set_weights(weights)
        return

    def learn_step(self, batch_size=32):
        ## sample from buffer ##
        if self.replay_buffer.use_per:
            (states, actions, rewards, nstates ,dones), importance, indices  = self.replay_buffer.get_minibatch(batch_size=batch_size)
        else:
            states, actions, rewards, nstates ,dones  = self.replay_buffer.get_minibatch(batch_size=batch_size)

        #### prepare labels ###
        arg_q_max = self.prim_qnet.predict(tf.stack(states)).argmax(axis=1)
        future_q_vals = self.target_qnet.predict(tf.stack(nstates))
        nextq = future_q_vals[range(batch_size), arg_q_max]
        target_q = rewards + (1-dones)*nextq
        with tf.GradientTape() as tape:
            tape.watch(self.prim_qnet.trainable_variables)
            qvalues = self.prim_qnet(tf.stack(states))
            Q = tf.reduce_sum(tf.multiply(qvalues, tf.keras.utils.to_categorical(actions, self.n_actions)), axis=1)
            error = target_q - Q #this is for importance sample
            loss = tf.keras.losses.Huber()(target_q, Q)
            if self.replay_buffer.use_per:
                loss = tf.reduce_mean(loss*importance) #not entirely sure if this works or not (?)

        grads = tape.gradient(loss, self.prim_qnet.trainable_variables)
        self.prim_qnet.optimizer.apply_gradients(zip(grads, self.prim_qnet.trainable_variables))
        self.update_target_parameters(tau=self.tau)
        if self.replay_buffer.use_per:
            self.replay_buffer.set_priorities(indices, error)
        return loss.numpy()

    def give_action(self,state, ep=0.01):
        if np.random.random() < ep:
            random_action = np.random.choice(range(self.prim_qnet.output_shape[-1]))
            return random_action
        else:
            return self.prim_qnet.predict(tf.expand_dims(state,axis=0)).argmax(axis=1)[0]


    def schedule(self, k, total_timesteps):
        if self.policy == "exp-decay":
            tt = self.exp_decay_effective_explotation*total_timesteps/np.log(1/self.ep0) #not best idea to calculate each time...
            return max(self.ep0, np.exp(-k/tt))
        else:
            return self.ep0

    def learn(self, total_timesteps, episodes_before_learn=100, batch_size=32):
        lhist=[]
        pt=[]
        rcum=[]
        rehist = []
        cumre=0
        episodes = np.arange(1,total_timesteps+1,1)
        start = datetime.now()

        self.env.reset()
        for k in tqdm(episodes, disable=self.use_tqdm):
            done = False
            state=self.env.reset()
            while not done:
                action = self.give_action(state, ep=self.schedule(k, total_timesteps))
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.add_experience(action, [state, next_state], reward, done)
                state = next_state
            cumre+=reward
            rehist.append(reward)
            rcum.append(cumre)
            if k>episodes_before_learn:
                lhist.append(self.learn_step(batch_size=batch_size))

            #####greedy prob#####
            state = self.env.reset()
            done = False
            while not done:
                action = self.give_action(state, ep=0)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
            pt.append(reward)
        end = datetime.now()
        self.info += "batch_size: {}\ntotal_timesteps: {}\nepisodes_before_learn: {}\ntau (target update): {}\n\nTOTAL_TIME: {}".format(batch_size, total_timesteps, episodes_before_learn, self.tau, end-start)
        self.save_learning_curves(rcum/np.array(episodes), rehist, pt, lhist)
        return


    def save_learning_curves(self,rcum_per_e, rehist, pt, lhist):
        if not os.path.exists(self.name):
            os.makedirs(self.name)
            with open(self.name+"/number_run.txt", "w+") as f:
                f.write("0")
                f.close()
            number_run = 0
        else:
            with open(self.name+"/number_run.txt", "r") as f:
                a = f.readlines()[0]
                f.close()
            with open(self.name+"/number_run.txt", "w") as f:
                f.truncate(0)
                number_run = int(a)+1
                f.write(str(int(a)+1))
                f.close()

        dir_to_save = self.name+"/run_"+str(number_run)
        os.makedirs(dir_to_save)
        os.makedirs(dir_to_save+"/data_collected")

        #### save learning_curves ####
        np.save(dir_to_save+"/data_collected/cumulative_reward_per_episode", rcum_per_e, allow_pickle=True )
        np.save(dir_to_save+"/data_collected/reward_history", rehist, allow_pickle=True )
        np.save(dir_to_save+"/data_collected/pgreedy", pt, allow_pickle=True )
        self.replay_buffer.save(dir_to_save+"/data_collected") #save buffer experiences (which are actually what we are interested in.



        ### save some info of the model ###
        with open(dir_to_save+"/info_model.txt", "w") as f:
            a = f.write(self.info)
            f.close()

        #### make the plot ####
        plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid((1,2), (0,0))
        ax2 = plt.subplot2grid((1,2), (0,1))
        ax1.plot(pt, alpha=0.6,c="blue", linewidth=1,label="greedy policy")
        ax1.scatter(np.arange(1,len(rehist)+1), rehist, alpha=0.5, s=50, c="black", label="reward")
        ax1.plot(rcum_per_e, alpha=0.6, linewidth=9,c="red",label="cumulative reward")
        ax2.plot(range(len(lhist)), lhist, alpha=0.6, linewidth=1,c="blue",label="critic loss")
        ax1.legend(prop={"size":20})
        ax2.legend(prop={"size":20})
        plt.savefig(dir_to_save+"/learning_curves.png")

        return

class ReplayBuffer:
    def __init__(self, state_shape, size=10**5, use_per=True):
        self.size = size
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.states = np.empty((self.size, 2, state_shape), dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)
        self.use_per = use_per
        self.max_reward = -1

    def add_experience(self, action, states, reward, terminal):

        # self.max_reward = max(self.max_reward, reward)
        self.actions[self.current] = action
        self.states[self.current] = states
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)

        self.count = max(self.count, self.current+1) #
        self.current = (self.current + 1) % self.size #

    def get_minibatch(self, batch_size=32, priority_scale=0.7):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        # Get sampling probabilities from priority list
        if self.use_per:
            scaled_priorities = self.priorities[:self.count] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            # Get a random number from history_length to maximum frame written with probabilities based on priority weights
            if self.use_per:
                index = np.random.choice(np.arange(self.count), p=sample_probabilities)
            else:
                index = random.randint(0,self.count - 1)
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.states[idx][0])
            new_states.append(self.states[idx][1])

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1/self.count * 1/sample_probabilities[[index for index in indices]]
            importance = importance / importance.max()
            return (states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]), importance, indices
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer in some folder"""
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.states)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)
        np.save(folder_name+"/priorities.npy", self.priorities)

    def load(self, folder_name):
        """Loads the replay buffer from some folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.states = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')
        self.count = len(self.rewards)
        self.current = len(self.rewards)
        self.priorities = np.load(folder_name+"/priorities.npy")
