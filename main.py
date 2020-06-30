import gym

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver
from vans_gym.solvers import CirqSolver
from stable_baselines3.common.monitor import Monitor
from callbacks import GreedyCallback
# from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    n_qubits = 3

    maximum_number_of_gates = 9
    tensorboard_folder = "./tensorboard/"

    solver = CirqSolver(n_qubits)
    # solver = PennylaneSolver(n_qubits)

    env = VansEnv(solver, maximum_number_of_gates)
    # env = Monitor(env)  # Useful to display more information on Tensorboard
    # check_env(env) #Why does this run for 4 times ?? Is it because of the cores i have?

    model = PPO('MlpPolicy', env, n_steps=10, tensorboard_log=None)

    # test_env = VansEnv(solver, maximum_number_of_gates, training_env=False)
    eval_callback = GreedyCallback()

    print("\n------------------------ Training ------------------------\n")

    model.learn(total_timesteps=50, callback=eval_callback)

    print("\n------------------------ Testing ------------------------\n")

    obs = env.reset()
    # for i in range(maximum_number_of_gates+1):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        # if done:
        #   obs = env.reset()

    print("\n-------------------- Final results --------------------\n")
    print("Final set of gates", env.state_indexed)
    print("Final reward", reward)
    print("Final state", env.quantum_state)
    env.close()
