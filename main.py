import gym
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver

if __name__ == "__main__":
    n_qubits = 3
    maximum_number_of_gates = 8

    solver = PennylaneSolver(n_qubits)
    env = VansEnv(solver, maximum_number_of_gates)
    check_env(env)

    model = PPO('MlpPolicy', env, n_steps=256)

    print("\n------------ Training ----------------\n")

    model.learn(total_timesteps=40000)

    print("\n------------- Testing ----------------\n")

    obs = env.reset()
    for i in range(maximum_number_of_gates+1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        quantum_state = env.quantum_state
        env.render()
        if done:
          obs = env.reset()

    print("Final reward", reward)
    print("Final state", quantum_state)
    env.close()