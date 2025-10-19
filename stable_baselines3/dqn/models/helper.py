
import sys
import os

import scipy.integrate
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import gymnasium as gym
from gymnasium import Wrapper
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy
import numpy as np
import argparse

#For wrapping multiple envs in a Vec
class CompatibilityWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
    def get_wrapper_attr(self, name):
        return getattr(self, name)
    
def make_envs(env_id: str, seed: int, rank: int, atari:bool):
    def init():
        env = AtariWrapper(gym.make(env_id)) if atari else gym.make(env_id)
        env = CompatibilityWrapper(env)
        if not atari: env.reset(seed=seed+rank)
        return env
    return init

def multiple_envs(num: int, env: str, atari: bool=False):
    seed = random.randint(0, 1000)
    return SubprocVecEnv([make_envs(env, seed, i, atari) for i in range(num)])

#For visualizing logged results
def viz(dir: str):
    dataset = {}
    for entry in os.scandir(dir):  
        if entry.is_file():  # check if it's a file
            dataset[str(entry.path)] = pd.read_csv(entry.path)
    for i in dataset:
        df = dataset[i]
        df = df[df['eval/mean_ep_length'].notna()]
        df = df[df['time/total_timesteps'].notna()]
        x = df['time/total_timesteps']
        y = df['eval/mean_ep_length']
        plt.plot(x, y, label=i)
        plt.legend()
    plt.show()      
    

def table(env: str, num_env: str):
    table = {'time_steps': [1000000, 500000, 250000, 200000, 150000, 100000, 50000, 20000]}
    tmp_path = './stable_baselines3/dqn/models/logged_results/' + env +  '/' + num_env + 'envs/'
    dataset = {}
    for entry in os.scandir(tmp_path):  
        if entry.is_file():  # check if it's a file
            dataset[os.path.basename(entry.path)] = pd.read_csv(entry.path)

    for i in dataset:
        l = []
        for  j in table['time_steps']:
            df = pd.DataFrame.copy(dataset[i])
            df = df[df['time/total_timesteps'] <= j]
            df = df[df['eval/mean_ep_length'].notna()]
            df = df[df['time/total_timesteps'].notna()]
            l.append(sum(df['eval/mean_ep_length']))
        table[i] = l  
    return pd.DataFrame(table)       


#For evaluating performance
def performance(file: str):
    df = pd.read_csv(file)
    df = df[df['time/total_timesteps'] <= 15000]
    print(len(df))
    df = df[df['eval/mean_ep_length'].notna()]
    df = df[df['time/total_timesteps'].notna()]
    x = df['time/total_timesteps']
    y = df['eval/mean_ep_length']
    return sum(y)


table = {'time_steps': [1000000, 500000, 250000, 200000, 150000, 100000, 50000, 20000]}
def comp(file: str):
    data = pd.read_csv(file)
    l = []
    file_name = os.path.basename(os.path.dirname(file)) 
    for i in [1000000, 500000, 250000, 200000, 150000, 100000, 50000, 20000]:
        df = pd.DataFrame.copy(data)
        df = df[df['time/total_timesteps'] <= i]
        df = df[df['eval/mean_ep_length'].notna()]
        df = df[df['time/total_timesteps'].notna()]
        l.append(sum(df['eval/mean_ep_length']))
    table[file_name] = l  


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def algos(buffer: bool, secondnet: bool, prio: bool, duel: bool):
    res = ""

    if secondnet: res += "2net_" 
    else: res += "1net_"

    if buffer: res += "buffer_"

    if prio: res += "prio_"

    if duel: res += "duel"

    return res






