import argparse
import gym
from vans_gym.envs import VansEnvsSeq
from vans_gym.solvers import CirqSolverR, Checker
from vans_gym.models import DuelDQN

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    observable_name = "Ising_"
    # observable_name= "EasyIsing_"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--names", type=str, default="DuelDQN")
    parser.add_argument("--policy_agent", type=str, default="exp-decay")
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--depth_circuit", type=int, default=3)
    parser.add_argument("--total_timesteps", type=int, default=150)
    parser.add_argument("--episodes_before_learn", type=int, default=2)
    parser.add_argument("--use_tqdm", type=int, default=False)
    parser.add_argument("--plotter", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ep", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--priority_scale", type=float, default=0.0)
    parser.add_argument("--qlr", type=float, default=0.05)
    parser.add_argument("--qepochs", type=int, default=50)


    args = parser.parse_args()

    solver = CirqSolverR(n_qubits = args.n_qubits, observable_name=observable_name,qlr=args.qlr,qepochs=args.qepochs)
    checker = Checker(solver)

    env = VansEnvsSeq(solver, checker, depth_circuit=args.depth_circuit)

    model = DuelDQN(env, name=args.names, policy=args.policy_agent, ep=args.ep, use_tqdm=not args.use_tqdm, plotter=args.plotter, learning_rate=args.learning_rate,tau=args.tau, priority_scale=args.priority_scale, use_per=False) #tqdm 0 not using, 1 using

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=args.total_timesteps, episodes_before_learn=args.episodes_before_learn, batch_size=args.batch_size)

    #env.close() ##if werun this on hpc error since plt
