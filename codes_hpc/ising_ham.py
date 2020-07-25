import argparse
import gym
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, CirqSmartSolver
from vans_gym.models import DuelDQN

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    observable_name = "Ising_"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--names", type=str, default="hola")
    parser.add_argument("--policy_agent", type=str, default="exp-decay")
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--depth_circuit", type=int, default=2)
    parser.add_argument("--total_timesteps", type=int, default=2)
    parser.add_argument("--episodes_before_learn", type=int, default=2)
    parser.add_argument("--use_tqdm", type=int, default=False)
    parser.add_argument("--plotter", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--ep", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--priority_scale", type=float, default=0.4)

    args = parser.parse_args()
    print(args)

    solver = CirqSmartSolver(n_qubits = args.n_qubits or 2, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=args.depth_circuit or 2, state_as_sequence=True, printing=False)

    model = DuelDQN(env, name=args.names, policy=args.policy_agent, ep=args.ep, use_tqdm=not args.use_tqdm, plotter=args.plotter, learning_rate=args.learning_rate, tau=args.tau, priority_scale=args.priority_scale) #0 not using, 1 using

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=args.total_timesteps, episodes_before_learn=args.episodes_before_learn)

    #env.close() ##if werun this on hpc error since plt
