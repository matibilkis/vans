import gym

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver
from vans_gym.solvers import CirqSolver
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings('ignore') #this is due to W projector construction in CirqSolver: search for ot=float(coeff) in vans_gym/solvers/cirq_solver

if __name__ == "__main__":
    n_qubits = 3

    maximum_number_of_gates = 8 #notice this will lead to final sequence of length maximum_number_of_gates+1


    tensorboard_folder = "./tensorboard/"

    solver = CirqSolver(n_qubits)
    # solver = PennylaneSolver(n_qubits)

    env = VansEnv(solver, maximum_number_of_gates)
    # env = Monitor(env)  # Useful to display more information on Tensorboard
    # check_env(env) #Why does this run for 4 times ?? Is it because of the cores i have?

    model = PPO('MlpPolicy', env, n_steps=50, tensorboard_log=tensorboard_folder)

    print("\n------------ Training ----------------\n")

    model.learn(total_timesteps=1000)

    print("\n------------- Testing ----------------\n")

    obs = env.reset()
    # for i in range(maximum_number_of_gates+1):
    for i in range(1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    print("Final set of gates", env.state_indexed)
    print("Final reward", reward)
    print("Final state", env.quantum_state)
    env.close()
