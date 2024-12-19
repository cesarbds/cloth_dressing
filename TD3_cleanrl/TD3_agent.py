import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from TD3_cleanrl.td3_cleanrl import *


class TD3_agent():
    def __init__(self, env, batch_size = 128,  gamma: float = 0.99, buffer_size: int = int(1e5),
                 learning_rate = 3e-4, tau: float = 0.005, policy_frequency: int = 2, exploration_noise: float = 0.1,
                 save_model: bool = True, enable_logging: bool = False, learning_starts: int = 1e3,
                 policy_noise: float = 0.2, noise_clip: float = 0.5, target_network_frequency: int = 1,
                 autotune: bool = True, seed:int = 42, exp_name: str  = "Test_TD3", 
                 total_timesteps: int = 1000000, device: str = 'cuda', run_name: str = f"TD3_GMM_{int(time.time())}") -> None:
        
        #environment
        self.env = env

        #logging
        if enable_logging:
            wandb.init(
                project="test_cleanrl_td3_gmm",
                entity=None,
                sync_tensorboard=True,
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
            self.writer = SummaryWriter(f"runs/{run_name}")
            self.writer.add_text(
            "hyperparameters","first_test_td3_cleanrl_gmm",
            )
        else:
            print("Logging is disabled. Wandb will not track this run.")

        #build network
        self.actor = Actor_TD3(env).to(device)
        self.qf1 = QNetwork_TD3(env).to(device)
        self.qf2 = QNetwork_TD3(env).to(device)
        self.qf1_target = QNetwork_TD3(env).to(device)
        self.qf2_target = QNetwork_TD3(env).to(device)
        self.target_actor = Actor_TD3(env).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate)

        #training parameters
        self.buffer_size = buffer_size
        self.batch_size  = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        self.policy_frequency = policy_frequency
        self.exploration_noise = exploration_noise
        self.save_model = save_model
        self.enable_logging = enable_logging
        self.learning_starts = learning_starts
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.target = target_network_frequency
        self.autotune = autotune
        self.seed = seed
        self.exp_name = exp_name
        self.total_timesteps = total_timesteps
        self.device = device
        self.run_name = run_name


    def train(self):
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        self.env.observation_space.dtype = np.float32
        rb = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        )
        start_time = time.time()
        # TRY NOT TO MODIFY: start the game
        obs, _, state = self.env.reset(seed=self.seed)
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array([self.env.action_space.sample() for _ in range(1)])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, termination, infos, _ = self.env.step(actions[0])
            episode_reward += rewards
            episode_timesteps += 1
            
            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            rb.add(obs, real_next_obs, actions, rewards, termination, infos)
            if episode_timesteps > 2.5e2:
                    termination = True  
            if termination:
                print(f"Total T: {global_step} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                if self.enable_logging:
                    self.writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                    self.writer.add_scalar("charts/episodic_length", episode_timesteps, global_step)
                # Reset environment
                new_env = False
                while  new_env  == False:
                    state, new_env, state = self.env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1  

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = rb.sample(self.batch_size)
                with torch.no_grad():
                    clipped_noise = (torch.randn_like(data.actions, device=self.device) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    ) * self.target_actor.action_scale

                    next_state_actions = (self.target_actor(data.next_observations) + clipped_noise).clamp(
                        self.env.action_space.low[0], self.env.action_space.high[0]
                    )
                    qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if global_step % 100 == 0:
                    if self.enable_logging:
                        self.writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                        self.writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                        self.writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                        self.writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                        self.writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                        self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                        
                        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
        if self.save_model:
            model_path = f"runs/{self.run_name}/{self.exp_name}.cleanrl_model"
            torch.save((self.actor.state_dict(), self.qf1.state_dict(), self.qf2.state_dict()), model_path)
            print(f"model saved to {model_path}")
        if self.enable_logging:
            self.writer.close()   