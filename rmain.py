import argparse
import gym
from vans_gym.envs import VansEnvsSeq
from vans_gym.solvers import CirqSolverR, Checker
from vans_gym.models.RDQN import RecurrentModel


import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    observable_name = "Ising_"
    # observable_name= "EasyIsing_"
    #
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--names", type=str, default="DuelDQN")
    # parser.add_argument("--policy_agent", type=str, default="exp-decay")
    # parser.add_argument("--n_qubits", type=int, default=2)
    # parser.add_argument("--depth_circuit", type=int, default=3)
    # parser.add_argument("--total_timesteps", type=int, default=10)
    # parser.add_argument("--episodes_before_learn", type=int, default=2)
    # parser.add_argument("--use_tqdm", type=int, default=False)
    # parser.add_argument("--plotter", type=int)
    # parser.add_argument("--learning_rate", type=float, default=0.00001)
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--ep", type=float, default=0.05)
    # parser.add_argument("--tau", type=float, default=0.01)
    # parser.add_argument("--priority_scale", type=float, default=0.7)
    # parser.add_argument("--qlr", type=float, default=0.05)
    # parser.add_argument("--qepochs", type=int, default=50)
    # parser.add_argument("--fitsep", type=int, default=1)


    # args = parser.parse_args()

    solver = CirqSolverR(n_qubits = 2, observable_name=observable_name,qlr=0.1,qepochs=50)
    checker = Checker(solver)

    env = VansEnvsSeq(solver, checker, depth_circuit=3)

    model = RecurrentModel(env)
    model.learn(total_timesteps=10**3)
