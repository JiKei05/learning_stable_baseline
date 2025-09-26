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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--buffer', type=str2bool, default=True)
parser.add_argument('--secondnet', type=str2bool, default=True)
parser.add_argument('--prio', type=str2bool, default=True)
parser.add_argument('--num_env', type=int, default=8)
args = parser.parse_args() 


def main(buffer: bool, secondnet: bool, prio: bool, num_env: int):

    tmp_path = "./logged_results/MountainCar/"

    algo = algos(buffer, secondnet, prio)

    envs = multiple_envs(num_env, 'MountainCar-v0')

    environment = gym.make('MountainCar-v0')
    model = DQN("MlpPolicy", envs, batch_size=128, learning_starts=1000, train_freq=(16, "step"), verbose=0,
                target_update_interval=600, gradient_steps=8, buffer_size=10000, use_buffer=buffer, use_second_net=secondnet, prio_replay=prio,
                exploration_final_eps=0.07, exploration_fraction=0.2, gamma=0.98, learning_rate=0.004, n_steps=120000, policy_kwargs=dict(net_arch=[256, 256])
              )
    
    csv_out = CSVOutputFormat(os.path.join(tmp_path, algo + ".csv"))
    #tensorboard_out = TensorBoardOutputFormat(os.path.join(tmp_path, "tensorboard"))
    new_logger = Logger(folder=None, output_formats=[csv_out])
    model.set_logger(new_logger)
    evaluate = EvalCallback(environment, eval_freq=50, n_eval_episodes=10)
    model.learn(total_timesteps=6000, callback=evaluate, log_interval=50)

if __name__ == "__main__":
    main(args.buffer, args.secondnet, args.prio, args.num_env)