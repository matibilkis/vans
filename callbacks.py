from stable_baselines3.common.callbacks import BaseCallback


class GreedyCallback(BaseCallback):
    """
    Callback for printing reward of greedy strategy.
    """
    def __init__(self, verbose=0):
        super(GreedyCallback, self).__init__(verbose)
        # self.check_freq = check_freq
        # self.save_path = os.path.join(log_dir, 'best_model')
        # self.best_mean_reward = -np.inf

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        print("\n\n************** Updating policy **************\n\n")

        done = False
        list_gates = []
        self.training_env.set_attr("in_callback", True)
        while not done:
            new_gate = self.model.policy.predict(self.training_env.get_attr("quantum_state"), deterministic=True)[0]
            list_gates.append(new_gate[0])
            _, _, done, _ = self.training_env.step(new_gate)
        self.training_env.set_attr("in_callback", False)

        print("Deterministic policy gives: ", list_gates)

        return True
