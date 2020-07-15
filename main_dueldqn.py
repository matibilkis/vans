import gym
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, CirqSmartSolver
from vans_gym.models import Duel_DQN

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    observable_name = "Ising_High_TFields"

    solver = CirqSmartSolver(n_qubits = 2, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=solver.n_qubits, state_as_sequence=True, printing=False)

    model = Duel_DQN(env)

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=10, episodes_before_learn=2)

    env.close()
