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
parser.add_argument('--prio', type=str2bool, default=True)
parser.add_argument('--num_env', type=int, default=8)
args = parser.parse_args() 

def main(file: bool, buffer: bool, secondnet: bool, num_env: int, prio: bool):

    algo= algos(buffer, secondnet, prio)

    tmp_path = "./logged_results/LunarLander/" + str(num_env) + "envs/"

    envs = multiple_envs(num_env,'LunarLander-v3')

    environment = gym.make('LunarLander-v3')
    model = DQN("MlpPolicy", envs, batch_size=128, learning_starts=0, train_freq=(4, "step"), verbose=0,
                target_update_interval=250, gradient_steps=-1, buffer_size=50000, use_buffer=buffer, use_second_net=secondnet, 
                exploration_final_eps=0.1, exploration_fraction=0.12, gamma=0.99, learning_rate=0.00063, n_steps=100000, policy_kwargs=dict(net_arch=[256, 256])
              )
    
    csv_out = CSVOutputFormat(os.path.join(tmp_path, algo + ".csv"))
    #tensorboard_out = TensorBoardOutputFormat(os.path.join(tmp_path, "tensorboard"))
    new_logger = Logger(folder=None, output_formats=[csv_out])
    model.set_logger(new_logger)
    evaluate = EvalCallback(environment, eval_freq=50, n_eval_episodes=10)
    model.learn(total_timesteps=1200000, callback=evaluate, log_interval=50)

if __name__ == "__main__":
    main(args.file, args.buffer, args.secondnet, args.num_env)