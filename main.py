import gym
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver

if __name__ == "__main__":
    n_qubits = 3

    solver = PennylaneSolver(n_qubits)
    env = VansEnv(solver)
    check_env(env)

    model = PPO('MlpPolicy', env)

    model.learn(total_timesteps=100)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()