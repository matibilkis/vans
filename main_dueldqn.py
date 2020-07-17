import argparse
import gym
from vans_gym.envs import VansEnv
from vans_gym.solvers import CirqSolver, CirqSmartSolver
from vans_gym.models import Duel_DQN
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_qubits", type=int)
    parser.add_argument("--depth_circuit", type=int)
    parser.add_argument("--total_timesteps", type=int)
    parser.add_argument("--episodes_before_learn", type=int)
    parser.add_argument("--use_tqdm", type=int) #if entry, model uses tqdm
    parser.add_argument("--plotter", type=int) #if entry, model uses tqdm

    args = parser.parse_args()

    observable_name = "Ising_High_TFields"

    solver = CirqSmartSolver(n_qubits = args.n_qubits or 2, observable_name=observable_name)
    env = VansEnv(solver, depth_circuit=args.depth_circuit or 2, state_as_sequence=True, printing=False)

    model = Duel_DQN(env, policy="exp-decay", use_tqdm=not args.use_tqdm, plotter= args.plotter) #0 not using, 1 using

    print("\n------------------------ Training ------------------------\n")
    model.learn(total_timesteps=args.total_timesteps or 100, episodes_before_learn=args.episodes_before_learn or 10)

    env.close()
