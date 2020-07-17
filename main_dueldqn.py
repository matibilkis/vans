import gym
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, CirqSmartSolver
from vans_gym.models import DuelDQN

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    observable_name = "Ising_High_TFields"

    solver = CirqSmartSolver(n_qubits=3, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=2, state_as_sequence=True, printing=False)

    model = DuelDQN(env, policy="exp-decay", use_tqdm=True)

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=100, episodes_before_learn=20)

    env.close()
