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
    parser.add_argument("--names", type=int,default=-1)
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--depth_circuit", type=int, default=2)
    parser.add_argument("--total_timesteps", type=int, default=4)
    parser.add_argument("--episodes_before_learn", type=int, default=2)
    parser.add_argument("--use_tqdm", type=int, default=1)#tqdm 0 not using, 1 using
    parser.add_argument("--plotter", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--priority_scale", type=float, default=0.5)

    args = parser.parse_args()
    solver = CirqSmartSolver(n_qubits = args.n_qubits, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=args.depth_circuit, state_as_sequence=True, printing=False)

    model = DuelDQN(env, policy="exp-decay", name=args.names, use_tqdm=not args.use_tqdm, plotter=args.plotter, learning_rate = args.learning_rate, tau = args.tau, priority_scale=args.priority_scale)
    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=args.total_timesteps, episodes_before_learn=args.episodes_before_learn)

    env.close()
