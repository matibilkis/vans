import gym
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver #Notice that this could be replaced by Luckasz code (in principle)
from stable_baselines3.common.callbacks import EvalCallback



if __name__ == "__main__":
    n_qubits = 3
    maximum_number_of_gates = 9

    solver = PennylaneSolver(n_qubits)
    env = VansEnv(solver, maximum_number_of_gates, bandit=True)
    # check_env(env) #Why does this run for 4 times ??

    # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
    #                              log_path='./logs/', eval_freq=1,
    #                              deterministic=True, render=False)
    ### Matias wanted to use this to draw the agent's evolution... dismissed for now
    model = PPO('MlpPolicy', env, n_steps=10)

    print("\n------------ Training ----------------\n")

    model.learn(total_timesteps=2)

    print("\n------------- Testing ----------------\n")

    obs = env.reset()
    # for i in range(maximum_number_of_gates+1):
    for i in range(1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        quantum_state = env.quantum_state
        env.render()
        if done:
          obs = env.reset()

    print("Final reward", reward)
    print("Final state", quantum_state)
    env.close()
