import os
from datetime import datetime
import torch as th
from torch import nn
import gym
import gym_minigrid
import argparse

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('-t','--total_timesteps', type=int, default=2500000)
    parser.add_argument('-e', '--env', type=str, default='MiniGrid-DoorKey-6x6-v0') 
    parser.add_argument('-s', '--saves_logs', type=str, default='minigrid_cnn_logs')
    parser.add_argument('--num_exp', type=int, default=1)
    return parser.parse_args()


class MinigridCNN(BaseFeaturesExtractor):
    """
    CNN for minigrid:
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        final_dim = 64
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, final_dim, (2, 2)),
            nn.ReLU(),
            nn.Flatten())
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
 	
        #n = observation_space.shape[1]
        #m = observation_space.shape[2]
        #self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*final_dim
        #self.linear = nn.Sequential(nn.Linear(self.image_embedding_size, features_dim), nn.ReLU())
 
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
 
 
def main(args):
    config = {
        "total_timesteps": 100000000,
        "env_name": "MiniGrid-DoorKey-6x6-v0",
    }
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
     
    def make_env():
        env = gym.make(config["env_name"])
        env = gym_minigrid.wrappers.ImgObsWrapper(env)
        env = Monitor(env)
        obs = env.reset()
        return env
     
     
    env = DummyVecEnv([make_env])
    
    eval_callback = EvalCallback(VecTransposeImage(env), best_model_save_path=str('logs/'+args.saves_logs+'_'+dt_string),
                                 log_path=str('logs/'+args.saves_logs+'_'+dt_string), eval_freq=1000,
                                 deterministic=True, render=False)
    
    policy_kwargs = dict(
        features_extractor_class=MinigridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        )

    if args.load:
        print(f'loading model{args.load}')
        model = PPO.load(args.load)
    else:
        model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1,tensorboard_log=str(args.saves_logs))
    
    for exp in range(args.num_exp):
        model.learn(total_timesteps=args.total_timesteps,tb_log_name='run_{}'.format(exp),callback=eval_callback)


if __name__ == "__main__":
    args = parse_args()
    main(args)
