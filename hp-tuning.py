import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C

import optuna

from vans_gym.envs import VansEnv
from vans_gym.solvers import PennylaneSolver

if __name__ == "__main__":
    n_qubits = 3
    maximum_number_of_gates = 8

    solver = PennylaneSolver(n_qubits)
    env = VansEnv(solver, maximum_number_of_gates)
    check_env(env)

    def objective(trial):
        n_steps = trial.suggest_int('n_steps', 16, 1024)
        batch_size = trial.suggest_int('batch_size', 4, 64)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
        ent_coef = trial.suggest_uniform('ent_coef', 0., 0.01)

        model = PPO('MlpPolicy', env, n_steps=n_steps, batch_size=batch_size, gamma=gamma, ent_coef=ent_coef)

        print("\n------------ Training ----------------\n")

        model.learn(total_timesteps=200000)

        print("\n------------- Testing ----------------\n")

        obs = env.reset()
        for i in range(maximum_number_of_gates + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            quantum_state = env.quantum_state
            env.render()
            if done:
                break

        print("Final set of gates", env.state_indexed)
        print("Final reward", reward)
        print("Final state", quantum_state)

        return -reward

    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

    print(study.best_params)

    env.close()
