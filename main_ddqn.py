import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import warnings
from collections import deque
import random
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, VAnsatz
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import os
warnings.filterwarnings('ignore')

class Critic(tf.keras.Model):
    def __init__(self,tau=0.05, seed_val = 0.05, output_dim=4):
        super(Critic,self).__init__()

        self.tau = tau
        self.output_dim = output_dim

        #self.input_layer = tf.keras.Input(shape=())

        self.l1 = Dense(60)#,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l2 = Dense(60)#, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l3 = Dense(output_dim)#, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)) #n_actions in the alphabet



    def update_target_parameters(self,primary_net):
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(self.tau * prim_weights[i] + (1 - self.tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def give_action(self,state, ep=0.01, more_states=1):
        if np.random.random() < ep:
            random_action = np.random.choice(range(self.output_dim))
            return random_action
        else:
            qvals = np.squeeze(self(tf.expand_dims(state, axis=0)))
            action_gredy = np.random.choice(np.where(qvals == np.max(qvals))[0])
            return action_gredy

    def call(self, inputs):
        feat = tf.nn.sigmoid(self.l1(inputs))
        feat = tf.nn.sigmoid(self.l2(feat))
        feat = tf.nn.sigmoid(self.l3(feat))
        return tf.multiply(2.0,feat)



class ReplayBuffer():
    def __init__(self, buffer_size=10**6):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, experience):
        if not isinstance(experience, tuple):
            raise ValueError("buffer wants tuples!")
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, int(batch_size))
        return batch

    def clear(self):
        self.buffer.clear()
        self.count = 0



def learning_step(critic, critic_target, buffer, optimizer, batch_size=30):
    for k in range(25):

        batch =buffer.sample(batch_size)
        states, actions, next_states, rewards, dones = np.transpose(batch)

        qpreds = critic(tf.stack(states))
        labels = qpreds.numpy()
        for inda, act in enumerate(actions):
            if dones[inda] is False:
                labels[inda,act] = np.max(np.squeeze(critic_target(tf.expand_dims(next_states[inda], axis=0))))
            else:
                labels[inda, act] = rewards[inda]


        with tf.GradientTape() as tape:
            tape.watch(critic.trainable_variables)
            qpreds = critic(tf.stack(states))

            loss = tf.keras.losses.MSE(labels, qpreds)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, critic.trainable_variables)
        optimizer.apply_gradients(zip(grads, critic.trainable_variables))
        critic_target.update_target_parameters(critic)
        return loss.numpy()



# names = ["Ising_High_TFields_HX",  "Ising_High_TFields_hybrid_3",


# qubs = [3, 3, 2 ]
# tot_eps = [10**4, 1000, 1000]

names=["Ising_High_TFields_hybrid_3"]
qubs=[3]
tot_eps = [500]

ind=0
for observable_name, nqubits in zip(names, qubs):
    buffer = ReplayBuffer()
    solver = CirqSolver(n_qubits = nqubits ,observable_name=observable_name)


    critic = Critic(output_dim=len(solver.alphabet))
    critic_target = Critic(output_dim=len(solver.alphabet), tau=0.01)
    env = VansEnv(solver,depth_circuit=solver.n_qubits, state_as_sequence=True, printing=False)
    env.reset()
    critic(tf.expand_dims(env.state, axis=0))
    critic_target(tf.expand_dims(env.state, axis=0))
    optimizer = tf.keras.optimizers.Adam(lr=0.01)

    r=[]
    pt=[]
    lhist=[]
    cumre=0
    episodes = np.arange(1,tot_eps[ind],1)
    ind+=1

    rhist=[]
    action_hist=[]
    tt = .75*len(episodes)/np.log(1/0.05)
    def schedule(k):
        return max(0.05, np.exp(-k/tt))

    #generate big buffer
    for k in tqdm(episodes):
        state = env.reset()
        done = False
        while not done:
            action = critic.give_action(state, ep=schedule(k))
            action_hist.append(action)
            next_state, reward, done, info = env.step(action)
            buffer.add((state, action, next_state, reward, done))
            state = next_state
        cumre+=reward
        r.append(cumre)
        lhist.append(learning_step(critic, critic_target, buffer, optimizer, batch_size=8))

    #     ####greedy prob#####
        state = env.reset()
        done = False
        while not done:
            action = critic.give_action(state, ep=0)
            next_state, reward, done, info = env.step(action)
            state = next_state
        #if (k>10**3):
        pt.append(reward)



    plt.figure(figsize=(20,20))
    ax1 = plt.subplot2grid((1,2), (0,0))
    ax2 = plt.subplot2grid((1,2), (0,1))


    ax1.plot(pt, alpha=0.6,c="blue", linewidth=1,label="greedy policy")
    ax1.plot(r/episodes, alpha=0.6, linewidth=9,c="red",label="cumulative reward")
    ax2.set_ylabel("Loss", size=50)
    ax2.plot(range(len(lhist)), lhist, alpha=0.6, linewidth=1,c="blue",label="critic loss")
    ax1.legend(prop={"size":20})
    plt.savefig(observable_name + "q_" + str(nqubits)+"410_4")
