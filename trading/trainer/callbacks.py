from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def save_mean_reward_to_file(path: str, mean_reward: float):
    with open(path, "w") as file:
        file.write(f"Mean reward {mean_reward}")


class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path: str, save_name: str, verbose: int = 0):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_name = save_name
        self.best_mean_reward = -np.inf
        self.mean_ep_reward = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos')
        if infos[0]:
            ep_reward = infos[0]['episode']['r']
            self.mean_ep_reward.append(ep_reward)
            mean_reward = sum(self.mean_ep_reward) / len(self.mean_ep_reward)
            print(f'mean reward: {mean_reward}')
            # print(f'n_cals: {self.n_calls}')

            if mean_reward > self.best_mean_reward and self.n_calls > 2e4:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, self.save_name))
                save_mean_reward_to_file(path=self.save_path + 'mean_reward.txt', mean_reward=mean_reward)
                message = f'Model with mean reward {mean_reward} saved to {self.save_path + self.save_name}'
                logger.info(message)
                if logger.getEffectiveLevel() > logging.WARNING:
                    print(message)

        return True