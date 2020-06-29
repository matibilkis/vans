from stable_baselines3.common.callbacks import BaseCallback
#
#
class GreedyCallback(BaseCallback):
    """
    Callback for printing reward of greedy strategy.
    """
    def __init__(self, check_freq: int, verbose=0):
        super(GreedyCallback, self).__init__(verbose)
        self.check_freq = check_freq
        # self.save_path = os.path.join(log_dir, 'best_model')
        # self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        print("on_step")

        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True
