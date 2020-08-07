import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import warnings
from collections import deque
import random
from tqdm import tqdm as tqdm
import os
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')



class Critic_rnn(tf.keras.Model):
    def __init__(self,tau=0.05, seed_val = 0.05, n_actions=6, state_shape=3, learning_rate=10**-2):
        super(Critic_rnn,self).__init__()

        self.tau = tau
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.mask = tf.keras.layers.Masking(mask_value=-1.,input_shape=(state_shape,))

        self.lstm = tf.keras.layers.LSTM(250, return_sequences=True, input_shape=(state_shape,))
        self.l1 = Dense(60, kernel_regularizer=tf.keras.regularizers.l1(0.01))
        self.l2 = Dense(60,   kernel_regularizer=tf.keras.regularizers.l1(0.01))
        self.l3 = Dense(n_actions)

        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    def update_target_parameters(self,primary_net, tau=0.01):
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.set_weights(weights)
        return

    @tf.function
    def greedy_act(self, tf_state):
        return tf.argmax(self(tf_state), axis=-1)

    def give_action(self,state, ep=0.01):
        if np.random.random() < ep:
            random_action = np.random.choice(range(self.n_actions))
            return random_action
        else:
            idx = self.greedy_act(tf.expand_dims(np.array(state), axis=0))
            idx =idx.numpy()[0]
            return idx

    def call(self, inputs):
        feat = self.mask(tf.cast(inputs/self.n_actions, tf.float32))
        feat = self.l1(inputs)
        feat = tf.nn.relu(feat)
        feat = self.l2(feat)
        feat = tf.nn.relu(feat)
        feat = self.l3(feat)
        feat = tf.nn.tanh(feat)
        return feat




class ReplayBuffer():
    def __init__(self, buffer_size=10**6, ps=.3):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.priorities = []
        self.ps=ps

    def add(self, experience, priority=0):
        if not isinstance(experience, tuple):
            raise ValueError("buffer wants tuples!")
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
            self.priorities.append(priority)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.priorities[self.count] = priority  #the most recent is self.count

    def size(self):
        return self.count

    def sample(self, batch_size):
        batch = []
        pro=(np.array(self.priorities)**self.ps)/np.sum(np.array(self.priorities)**self.ps)
        if self.count < batch_size:
            indices = np.random.choice(range(self.count), self.count, p=pro)
        else:
            indices = np.random.choice(range(self.count), int(batch_size), p=pro)
        for idx in indices:
            batch.append(self.buffer[idx])
        return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
#


class RecurrentModel:
    def __init__(self, env, use_tqdm=False, learning_rate = 0.01,
        size_rp=10**6, name="RDQN", policy="fixed_ep", ep=0.01,
        tau=0.1, priority_scale=0.5):

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

        logdir = self.name+"/logs/scalars/run_"+str(self.number_run)
        self.fw = tf.summary.create_file_writer(logdir)


        self.env = env
        self.n_actions = len(self.env.solver.alphabet) - self.env.solver.n_qubits #this is to avoid identity
        #since last n_qubits gates are identity.

        state_shape = self.env.depth_circuit  # We will modify this.

        ### define qnets ###
        self.prim_qnet = Critic_rnn(state_shape=state_shape, learning_rate=learning_rate, n_actions=self.n_actions)
        self.target_qnet = Critic_rnn(state_shape=state_shape,n_actions=self.n_actions)

        self.prim_qnet.compile(loss="mse")

        self.prim_qnet(tf.random.uniform((1,self.prim_qnet.state_shape)))
        self.target_qnet(tf.random.uniform((1,self.prim_qnet.state_shape)))
        self.prim_qnet.compile(loss="mse")

        self.target_qnet.update_target_parameters(self.prim_qnet, tau=1)

        self.tau = tau  # how soft the update is
        # Define Policy
        self.policy = policy
        self.ep0 = ep
        self.exp_decay_effective_exploitation = 0.5  # percentage of time at which ep(t0) = \ep0 with #ep(t) = \ep0 exp[- t / t0]

        # Define Buffer
        self.replay_buffer = ReplayBuffer(buffer_size=size_rp, ps=priority_scale)
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
                    f"buffer_size: {self.replay_buffer.buffer_size}"
                # Save some info of the model
        with open(self.dir_to_save+"/info_model.txt", "w") as f:
            f.write(self.info)
            f.close()

    def learn_step(self, bs=32):

        states, actions, ns, rewards, dones = np.transpose(self.replay_buffer.sample(bs))
        concst=tf.concat([tf.cast(tf.reshape(tf.stack(states), (bs,1,3)), tf.float32), tf.cast(tf.reshape(tf.stack(ns), (bs,1,3)), tf.float32)], axis=1)

        with tf.GradientTape() as tape:
            tape.watch(self.prim_qnet.trainable_variables)


            preds = self.prim_qnet(concst)
            predst = self.target_qnet(concst)

            argsgreedy=tf.argmax(preds[:,1,:], axis=-1)
            qnext = predst.numpy()[range(bs),1,argsgreedy]
            target_q = rewards + (1-dones)*qnext

            qvals_update = tf.reduce_sum(tf.multiply(preds[:,0,:], tf.keras.utils.to_categorical(actions, self.prim_qnet.n_actions)), axis=1)

            loss=tf.keras.losses.MeanSquaredError()(tf.stack(target_q),qvals_update)


            g= tape.gradient(loss, self.prim_qnet.trainable_variables)
        self.prim_qnet.optimizer.apply_gradients(zip(g, self.prim_qnet.trainable_variables))
        self.target_qnet.update_target_parameters(self.prim_qnet, tau=self.tau)
        return loss


    def schedule(self, k, total_timesteps):
        if self.policy == "exp-decay":
            tt = self.exp_decay_effective_exploitation * total_timesteps / np.log(1 / self.ep0)  # not best idea to calculate each time...
            return max(self.ep0, np.exp(-k/tt))
        else:
            return self.ep0

    def learn(self, total_timesteps, batch_size=64):
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

            stuck_count=0
            episode=[]
            done=False

            state = self.env.reset()
            while not done:
                action = self.prim_qnet.give_action(state, ep=self.schedule(k, total_timesteps))
                next_state, reward, done, info = self.env.step(action)

                if str(next_state) == str(state) or stuck_count>5:
                    print("che", state, next_state, action, reward)
                    reward=-1.
                episode.append((state, action, next_state, reward, done))
                state = next_state

            for step in episode:
                self.replay_buffer.add(step, priority=max(10**-10,reward))

            if k>batch_size:
                ll = self.learn_step(bs=batch_size)
                with self.fw.as_default():
                     tf.summary.scalar('losss', ll, step=k)


            trajectories["episode"].append(k)
            trajectories["reward"].append(reward)
            trajectories["sequence"].append(state)
            trajectories["params"].append(self.env.solver.final_params)
            cumulative_reward += reward
            reward_history.append(reward)
            cumulative_reward_history.append(cumulative_reward)

            with self.fw.as_default():
                tf.summary.scalar('reward', tf.convert_to_tensor(reward), step=k)

            with self.fw.as_default():
                tf.summary.scalar('cumulative reward', tf.convert_to_tensor(cumulative_reward/(k)), step=k)

            #### greeedy energy ####
            state = self.env.reset()
            done = False
            while not done:
                action = self.prim_qnet.give_action(state, ep=0)
                next_state, reward, done, info = self.env.step(action,evaluating=True)
                state = next_state
            pt.append(reward)

            with self.fw.as_default():
                tf.summary.scalar('greedy energy', tf.convert_to_tensor(reward), step=k)


        end = datetime.now()
        self.infotrain = f"batch_size: {batch_size}\n" \
                     f"total_timesteps: {total_timesteps}\n" \
                     f"episodes_before_learn: {episodes_before_learn}\n" \
                     f"tau (target update): {self.tau}\n\n" \
                     f"TOTAL_TIME: {end - start}"

        # Save some info of the model
        with open(self.dir_to_save+"/info_train.txt", "w") as f:
            f.write(self.infotrain)
            f.close()

        with open(self.dir_to_save+'/data_collected/circuits.pickle', 'wb') as handle:
            pickle.dump(trajectories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.save_learning_curves(cumulative_reward_history/np.array(episodes), reward_history, pt, loss_history)
