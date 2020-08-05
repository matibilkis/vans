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
import pickle


class qnet(tf.keras.Model):
    def __init__(self, seed_val=0.08, state_shape=3, n_actions=6, learning_rate=10**-3):

        super(qnet,self).__init__()


        self.lprocess = Dense(64, input_shape=[state_shape])#dom_uniform_initializer(minval=-seed_val, maxval=seed_val))
        self.lprocess1 = Dense(64)#,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        self.loutput = Dense(n_actions)#, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
#         self.optimizer = tf.keras.optimizers.Adam(lr=10**-2)
        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.n_actions = n_actions

    def call(self, inputs):
        feat = self.lprocess(inputs)
        feat = tf.nn.relu(feat)
        feat = self.lprocess1(feat)
        feat = tf.nn.relu(feat)
        feat = self.loutput(feat)
        return feat



    # @tf.function
    def train_step(self, data):
        states, actions, target_q = data

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            qvalues = self(tf.stack(states))
            Q = tf.reduce_sum(tf.multiply(qvalues, tf.keras.utils.to_categorical(actions, self.n_actions)), axis=1)
            error = target_q - Q  # this is for importance sample
            loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(target_q, Q))

            grads = tape.gradient(loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return loss, error



class DQN:
    def __init__(self, env, use_tqdm=False, learning_rate = 0.01,
        size_rp=10**5, name="DueDQN", policy="exp-decay", ep=0.01,
        tau=0.1, priority_scale=0.5, plotter=False, use_per=False,
        fits_per_ep=1, clip_rew=10**-3):

        """fits_per_ep is the number of fits in the Q-network per episode (a bit experimental.. but works!)"""

        self.name = name
        self.use_tqdm = use_tqdm

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
        self.number_run = number_run
        os.makedirs(dir_to_save)
        os.makedirs(dir_to_save+"/data_collected")
        self.dir_to_save = dir_to_save

        rewlogdir = self.name+"/logs/scalars/run_"+str(self.number_run)+"/reward"
        cumrewlogdir = self.name+"/logs/scalars/run_"+str(self.number_run)+"/cumreward"
        greedylogdir = self.name+"/logs/scalars/run_"+str(self.number_run)+"/greedy"
        losslogdir = self.name+"/logs/scalars/run_"+str(self.number_run)+"/loss"


        self.fw_rew = tf.summary.create_file_writer(rewlogdir)
        self.fw_cumrew = tf.summary.create_file_writer(cumrewlogdir)
        self.fw_greedy = tf.summary.create_file_writer(greedylogdir)
        self.fw_loss = tf.summary.create_file_writer(losslogdir)


        self.env = env
        self.n_actions = len(self.env.solver.alphabet) - self.env.solver.n_qubits #this is to avoid identity
        #since last n_qubits gates are identity.

        ## this is for HPC
        self.use_tqdm=use_tqdm
        self.plotter = plotter #toplot or not to plot

        state_shape = self.env.depth_circuit  # We will modify this.

        ### define qnets ###
        self.prim_qnet = qnet(state_shape=state_shape, learning_rate=learning_rate, n_actions=self.n_actions)
        self.target_qnet = qnet(state_shape=state_shape,n_actions=self.n_actions)

        self.prim_qnet.compile(loss="mse")
        self.prim_qnet(tf.random.uniform((1,1,state_shape)))
        self.target_qnet(tf.random.uniform((1,1,state_shape)))

        self.tau = tau  # how soft the update is
        self.clip_rew=clip_rew
        # Define Policy
        self.policy = policy
        self.ep0 = ep
        self.exp_decay_effective_exploitation = 0.5  # percentage of time at which ep(t0) = \ep0 with #ep(t) = \ep0 exp[- t / t0]
        self.fits_per_ep = fits_per_ep


        # Define Buffer
        self.replay_buffer = ReplayBuffer(state_shape, size=size_rp, use_per=use_per)
        self.priority_scale = priority_scale
        # Some info to save
        self.info = f"len(alphabet): {len(self.env.solver.alphabet)}\n" \
                    f"alphabet_gates: {self.env.solver.alphabet_gates}, \n" \
                    f"observable_name: {self.env.solver.observable_name}\n" \
                    f"depth_circuit: {self.env.depth_circuit}\n" \
                    f"n_qubits: {self.env.n_qubits}\n" \
                    f"name: {self.name}\n" \
                    f"lr: {learning_rate}\n\n" \
                    f"policy: {self.policy}\n" \
                    f"ep0: {self.ep0}\n" \
                    f"exp_decay_effective_exploitation: {self.exp_decay_effective_exploitation}\n\n" \
                    f"state_shape: {state_shape}\n" \
                    f"priority_scale: {priority_scale}\n" \
                    f"buffer_size: {self.replay_buffer}"


    def learn_step(self,batch_size):
        # Sample from buffer
        if self.replay_buffer.use_per:
            (states, actions, rewards, nstates, dones), importance, indices = self.replay_buffer.get_minibatch(batch_size=batch_size, priority_scale=self.priority_scale)
        else:
            states, actions, rewards, nstates, dones = self.replay_buffer.get_minibatch(batch_size=batch_size)

        # Prepare labels
        arg_q_max = self.prim_qnet.predict(tf.stack(states)).argmax(axis=1)
        future_q_vals = self.prim_qnet.predict(tf.stack(nstates))
        nextq = future_q_vals[range(batch_size), arg_q_max]
        target_q = rewards + (1-dones)*nextq

        data = states, actions, target_q
        loss, error = self.prim_qnet.train_step(data)
            #if self.replay_buffer.use_per:
            #    loss = tf.reduce_mean(loss*importance)  # not entirely sure if this works or not (?)

        # grads = [tf.clip_by_value(k, np.min(), self.clip_rew) for k in tape.gradient(loss, self.prim_qnet.trainable_variables)]

        self.update_target_parameters()
        if self.replay_buffer.use_per:
            self.replay_buffer.set_priorities(indices, error)
        return loss

    def give_action(self,state, ep=0.01):
        if np.random.random() < ep:
            random_action = np.random.choice(range(self.n_actions))
            return random_action
        else:
            return self.prim_qnet.predict(tf.expand_dims(state,axis=0)).argmax(axis=1)[0]

    def update_target_parameters(self):
        prim_weights = self.prim_qnet.get_weights()
        targ_weights = self.target_qnet.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(self.tau * prim_weights[i] + (1 - self.tau) * targ_weights[i])
        self.target_qnet.set_weights(weights)
        return

    def schedule(self, k, total_timesteps):
        if self.policy == "exp-decay":
            tt = self.exp_decay_effective_exploitation * total_timesteps / np.log(1 / self.ep0)  # not best idea to calculate each time...
            return max(self.ep0, np.exp(-k/tt))
        else:
            return self.ep0

    def learn(self, total_timesteps, episodes_before_learn=100, batch_size=32):
        loss_history = []
        pt = []
        cumulative_reward_history = []
        reward_history = []
        cumulative_reward = 0
        episodes = np.arange(1, total_timesteps+1, 1)
        start = datetime.now()
        loss_step = 0
        trajectories={"episode":[], "reward":[], "sequence":[], "params":[]}

        self.env.reset()

        for k in tqdm(episodes, disable=self.use_tqdm):
            done = False

            state = self.env.reset()
            while not done:
                action = self.give_action(state, ep=self.schedule(k, total_timesteps))
                next_state, reward, done, info = self.env.step(action)
                self.replay_buffer.add_experience(action, [state, next_state], reward, done)
                state = next_state
            trajectories["episode"].append(k)
            trajectories["reward"].append(reward)
            trajectories["sequence"].append(state)
            trajectories["params"].append(self.env.solver.final_params)
            cumulative_reward += reward
            reward_history.append(reward)
            cumulative_reward_history.append(cumulative_reward)
            if k > episodes_before_learn:
                for kk in range(self.fits_per_ep):
                    loss_step+=1
                    loss_history.append(self.learn_step(batch_size=batch_size))
                    with self.fw_loss.as_default():
                        tf.summary.scalar('loss', tf.convert_to_tensor(loss_history[-1]), step=loss_step)

            with self.fw_rew.as_default():
                tf.summary.scalar('reward', tf.convert_to_tensor(reward), step=k)

            with self.fw_cumrew.as_default():
                tf.summary.scalar('cumulative reward', tf.convert_to_tensor(cumulative_reward/(k)), step=k)

            state = self.env.reset()
            done = False
            while not done:
                action = self.give_action(state, ep=0)
                next_state, reward, done, info = self.env.step(action,evaluating=True)
                state = next_state
            pt.append(reward)

            with self.fw_greedy.as_default():
                tf.summary.scalar('greedy energy', tf.convert_to_tensor(reward), step=k)


        end = datetime.now()
        self.info += f"batch_size: {batch_size}\n" \
                     f"total_timesteps: {total_timesteps}\n" \
                     f"episodes_before_learn: {episodes_before_learn}\n" \
                     f"tau (target update): {self.tau}\n\n" \
                     f"TOTAL_TIME: {end - start}"

        with open(self.dir_to_save+'/data_collected/circuits.pickle', 'wb') as handle:
            pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.save_learning_curves(cumulative_reward_history/np.array(episodes), reward_history, pt, loss_history)

    def save_learning_curves(self, cum_reward_per_e, reward_history, pt, loss_history):

        # Save learning curves
        np.save(self.dir_to_save +"/data_collected/cumulative_reward_per_episode", cum_reward_per_e, allow_pickle=True)
        np.save(self.dir_to_save + "/data_collected/loss", loss_history, allow_pickle=True)
        np.save(self.dir_to_save +"/data_collected/reward_history", reward_history, allow_pickle=True)
        np.save(self.dir_to_save+"/data_collected/pgreedy", pt, allow_pickle=True)
        self.replay_buffer.save(self.dir_to_save+"/data_collected")  # save buffer experiences (which are actually what we are interested in.

        # Save some info of the model
        with open(self.dir_to_save+"/info_model.txt", "w") as f:
            f.write(self.info)
            f.close()

        if self.plotter:
            # Make the plot
            plt.figure(figsize=(20, 20))
            ax1 = plt.subplot2grid((1, 2), (0, 0))
            ax2 = plt.subplot2grid((1, 2), (0, 1))
            ax1.plot(pt, alpha=0.6, c="blue", linewidth=1, label="greedy policy")
            ax1.scatter(np.arange(1, len(reward_history) + 1), reward_history, alpha=0.5, s=50, c="black",
                        label="reward")
            ax1.plot(cum_reward_per_e, alpha=0.6, linewidth=9, c="red", label="cumulative reward")
            ax2.plot(range(len(loss_history)), loss_history, alpha=0.6, linewidth=1, c="blue", label="critic loss")
            ax1.legend(prop={"size": 20})
            ax2.legend(prop={"size": 20})
            plt.savefig(self.dir_to_save + "/learning_curves.png")


class ReplayBuffer:
    def __init__(self, state_shape, size=10**4, use_per=True):
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
        # if reward+.05 >= self.max_reward:
            # self.max_reward = reward
        self.priorities[self.current] = max(self.priorities.max(), 1)

        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size

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

    def set_priorities(self, indices, errors, offset=0.01):
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
        count = len(np.load(folder_name+"/cumulative_reward_per_episode.npy"))
        self.priorities = np.load(folder_name+"/priorities.npy")
        self.count = count
        self.current = count

#
