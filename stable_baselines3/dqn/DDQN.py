import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3.dqn.dqn_cop import DQN
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

environment = gym.make('CartPole-v1')
model = DQN("MlpPolicy", environment, learning_starts=64, train_freq=(64, "step"), verbose=0, target_update_interval=1000, gradient_steps=16)
evaluate = EvalCallback(environment, eval_freq=5000, n_eval_episodes=10)
model.learn(total_timesteps=500000, callback=evaluate)
    