import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from stable_baselines3.dqn.dqn_cop import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.dqn.models.helper import multiple_envs, str2bool, algos
from stable_baselines3.common.logger import CSVOutputFormat, Logger
import gymnasium as gym
import torch as th
from gymnasium import Wrapper
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--buffer', type=str2bool, default=True)
parser.add_argument('--secondnet', type=str2bool, default=True)
parser.add_argument('--prio', type=str2bool, default=False)
parser.add_argument('--duel', type=str2bool, default=True)
parser.add_argument('--num_env', type=int, default=8)
args = parser.parse_args() 

def main(buffer: bool, secondnet: bool, prio: bool, duel: bool, num_env: int):
    
    algo = algos(buffer, secondnet, prio, duel)

    tmp_path = "./logged_results/CartPole/"

    envs = multiple_envs(num_env, 'CartPole-v1')

    environment = gym.make('CartPole-v1')
    model = DQN("MlpPolicy", envs, batch_size=64, learning_starts=1000, train_freq=(256, "step"), verbose=0, prio_replay=prio, duel=duel,
                target_update_interval=10, gradient_steps=128, buffer_size=100000, use_buffer=False, use_second_net=False, 
                exploration_final_eps=0.04, exploration_fraction=0.16, gamma=0.99, learning_rate=0.0023, n_steps=50000, policy_kwargs=dict(net_arch=[256, 256])
              )
    
    csv_out = CSVOutputFormat(os.path.join(tmp_path, algo + ".csv"))
    new_logger = Logger(folder=None, output_formats=[csv_out])
    model.set_logger(new_logger)
    evaluate = EvalCallback(environment, eval_freq=50, n_eval_episodes=10)
    model.learn(total_timesteps=1200000, callback=evaluate, log_interval=50)

if __name__ == "__main__":
    main(args.buffer, args.secondnet, args.prio, args.duel, args.num_env)