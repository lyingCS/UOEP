from model.agents.DDPG import DDPG
from model.agents.BehaviorDDPG import BehaviorDDPG
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import utils

class Wolpertinger(DDPG):

    def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation,
                      do_actor_update=True, do_critic_update=True):

        # Get current Q estimate
        current_critic_output = self.facade.apply_critic(observation,
                                                         utils.wrap_batch(policy_output, device=self.device),
                                                         self.critic)
        current_Q = current_critic_output['q']

        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target, self.critic_target)
        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic_target)
        target_Q = target_critic_output['q']
        # target_Q = reward + self.gamma * (done_mask * target_Q).detach()
        target_Q = reward + self.gamma * ((1-done_mask.int()) * target_Q).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()

        # Regularization loss
        #         critic_reg = current_critic_output['reg']

        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        policy_output = self.facade.apply_policy(observation, self.actor, self.critic)
        critic_output = self.facade.apply_critic(observation, policy_output, self.critic)
        actor_loss = -critic_output['q'].mean()

        # Regularization loss
        #         actor_reg = policy_output['reg']

        if do_actor_update and self.actor_lr > 0:
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return critic_loss, actor_loss

    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
            policy_output = self.facade.apply_policy(observation, self.actor, self.critic, epsilon, do_explore=True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation

    def run_an_episode(self, epsilon, initial_observation=None, with_train=False, pick_rows=None):
        '''
        Run episode for a batch of user
        @input:
        - epsilon: greedy epsilon for random exploration
        - initial_observation
        - with_train: apply batch training for each step of the episode
        - pick_rows: pick certain rows of the data when reseting the environment
        '''
        # observation --> state, action
        if initial_observation:
            observation = initial_observation
        elif pick_rows:
            observation = self.facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
        else:
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        step = 0
        done = [False] * self.batch_size
        train_report = None
        while sum(done) < len(done):
            step += 1
            with torch.no_grad():
                # sample action
                policy_output = self.facade.apply_policy(observation, self.actor, self.critic, epsilon, do_explore=True)
                # policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=False)
                # apply action on environment and update replay buffer
                # next_observation, reward, done, info = self.facade.env_step(policy_output)
                next_observation, reward, done, info, ret = self.facade.env_step_without_new_sample(policy_output)
                # update replay buffer
                if not pick_rows:
                    self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
                # observate for the next step
                observation = next_observation
            if with_train:
                train_report = self.step_train()
        return ret