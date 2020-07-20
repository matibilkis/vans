import argparse
import gym
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, CirqSmartSolver
from vans_gym.models import DuelDQN

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    observable_name = "Ising_High_TFields"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_qubits", type=int, default=3)
    parser.add_argument("--depth_circuit", type=int, default=2)
    parser.add_argument("--total_timesteps", type=int, default=100)
    parser.add_argument("--episodes_before_learn", type=int, default=10)
    parser.add_argument("--use_tqdm", type=int, default=False)
    parser.add_argument("--plotter", type=int)

    args = parser.parse_args()

    solver = CirqSmartSolver(n_qubits = args.n_qubits or 2, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=args.depth_circuit or 2, state_as_sequence=True, printing=False)

    model = DuelDQN(env, policy="exp-decay", use_tqdm=not args.use_tqdm, plotter=args.plotter) #0 not using, 1 using

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=args.total_timesteps, episodes_before_learn=args.episodes_before_learn)

    env.close()
