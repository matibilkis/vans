import gym
from stable_baselines3.common.env_checker import check_env

from stable_baselines3 import PPO
from vans_gym.envs import VansEnv

if __name__ == "__main__":
    n_qubits = 6

    env = VansEnv(n_qubits)
    check_env(env)

    model = PPO('MlpPolicy', env)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()