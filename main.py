import gym

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


if __name__ == "__main__":
    n_qubits = 3
    maximum_number_of_gates = 9
    tensorboard_folder = "./tensorboard/"

    solver = PennylaneSolver(n_qubits)
    env = VansEnv(solver, maximum_number_of_gates, bandit=True)
    env = Monitor(env)  # Useful to display more information on Tensorboard

    # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
    #                              log_path='./logs/', eval_freq=1,
    #                              deterministic=True, render=False)

    model = PPO('MlpPolicy', env, n_steps=10,
                tensorboard_log=tensorboard_folder)

    print("\n------------ Training ----------------\n")

    model.learn(total_timesteps=2000)

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
