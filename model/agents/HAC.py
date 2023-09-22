import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.DDPG import DDPG
# from model.agents.BehaviorDDPG import BehaviorDDPG
    
class HAC(DDPG):
    @staticmethod
    def parse_model_args(parser):
        '''
        - args from DDPG:
            - episode_batch_size
            - batch_size
            - actor_lr
            - critic_lr
            - actor_decay
            - critic_decay
            - target_mitigate_coef
            - args from BaseRLAgent:
                - gamma
                - n_iter
                - train_every_n_step
                - initial_greedy_epsilon
                - final_greedy_epsilon
                - elbow_greedy
                - check_episode
                - with_eval
                - save_path
        '''
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--hyper_actor_coef', type=float, default=0.1, 
                            help='hyper actor loss coefficient')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        from DDPG:
            self.episode_batch_size
            self.batch_size
            self.actor_lr
            self.critic_lr
            self.actor_decay
            self.critic_decay
            self.actor
            self.actor_target
            self.actor_optimizer
            self.critic
            self.critic_target
            self.critic_optimizer
            self.tau
            from BaseRLAgent:
                self.gamma
                self.n_iter
                slef.train_every_n_step
                self.check_episode
                self.with_eval
                self.save_path
                self.facade
                self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.hyper_actor_coef = args.hyper_actor_coef
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr, 
                                                         weight_decay=args.behavior_decay)

                                    
    def action_before_train(self):
        super().action_before_train()
        self.training_history["hyper_actor_loss"] = []
        self.training_history["behavior_loss"] = []
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
#         reward = torch.FloatTensor(reward)
#         done_mask = torch.FloatTensor(done_mask)
        
        critic_loss, actor_loss, hyper_actor_loss = self.get_hac_loss(observation, policy_output, reward, done_mask, next_observation)
        behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['hyper_actor_loss'].append(hyper_actor_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['hyper_actor_loss'][-1], 
                              self.training_history['behavior_loss'][-1])}
    
    def get_hac_loss(self, observation, policy_output, reward, done_mask, next_observation, 
                      do_actor_update = True, do_critic_update = True):
        
        # critic loss
        
        # Get current Q estimate
        hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
        current_critic_output = self.facade.apply_critic(observation, hyper_output, self.critic)
        current_Q = current_critic_output['q']
        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic_target)
        target_Q = target_critic_output['q']
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        if do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # actor loss
        
        # Compute actor loss
        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            critic_output = self.facade.apply_critic(observation, policy_output, self.critic)
            actor_loss = -critic_output['q'].mean()
            # Optimize the actor 
            actor_loss.backward()
            self.actor_optimizer.step()
            
        # hyper actor loss
        
        if do_actor_update and self.hyper_actor_coef > 0:
            self.actor_optimizer.zero_grad()
            policy_output = self.facade.apply_policy(observation, self.actor)
            inferred_hyper_output = self.facade.infer_hyper_action(observation, policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inferred_hyper_output['Z'], policy_output['Z']).mean()
            # Optimize the actor 
            hyper_actor_loss.backward()
            self.actor_optimizer.step()
            
        return critic_loss, actor_loss, hyper_actor_loss

    def get_behavior_loss(self, observation, policy_output, next_observation, do_update = True):
        observation, exposure, feedback = self.facade.extract_behavior_data(observation, policy_output, next_observation)
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
        
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss