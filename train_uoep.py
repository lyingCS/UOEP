from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os

from model.agents import *
from model.policy import *
from model.critic import *
from model.facade import *
from env import *

import utils


if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    # init_parser.add_argument('--env_class', type=str, default='RL4RSEnvironment', help='Environment class.')
    init_parser.add_argument('--env_class', type=str, default='KREnvironment_GPU', help='Environment class.')
    init_parser.add_argument('--policy_class', type=str, default='OneStagePolicy', help='Policy class')
    init_parser.add_argument('--critic_class', type=str, default='GeneralCritic', help='Critic class')
    init_parser.add_argument('--agent_class', type=str, default='UOEP', help='Learning agent class')
    init_parser.add_argument('--facade_class', type=str, default='OneStageFacade', help='Environment class.')
    init_parser.add_argument('--population_size', type=int, default=5, help='Population size.')

    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)

    population_size = max(initial_args.population_size, 1)
    envClass = eval('{0}.{0}'.format(initial_args.env_class))
    policyClass = eval('{0}.{0}'.format(initial_args.policy_class))
    criticClass = eval('{0}.{0}'.format(initial_args.critic_class))
    agentClass = eval('{0}.{0}'.format(initial_args.agent_class))
    facadeClass = eval('{0}.{0}'.format(initial_args.facade_class))
    
    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12, help='random seed')
    parser.add_argument('--cuda', type=int, default=1, help='cuda device number; set to -1 (default) if using cpu')
    
    # customized args
    parser = envClass.parse_model_args(parser)
    parser = policyClass.parse_model_args(parser)
    parser = criticClass.parse_model_args(parser)
    parser = agentClass.parse_model_args(parser)
    parser = facadeClass.parse_model_args(parser)
    args, _ = parser.parse_known_args()
    
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    utils.set_random_seed(args.seed)
    
    # Environment
    print("Loading environment")
    envs = []
    for _ in range(population_size):
        envs.append(envClass(args))
    
    # Agent
    print("Setup policy:")
    policys = []
    for _ in range(population_size):
        policy = policyClass(args, envs[_])
        policy.to(device)
        policys.append(policy)
    print(policy)
    print("Setup critic:")
    critic = criticClass(args, envs[0], policy)
    critic.to(device)
    print(critic)
    print("Setup agent with data-specific facade")
    facades = []
    for i in range(population_size):
        facade = facadeClass(args, envs[i], policys[i], critic)
        facades.append(facade)
    agent = agentClass(args, facades)
    
    
    try:
        print(args)
        agent.train()
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + utils.get_local_time() + ' ' + '-' * 20)
            exit(1)
    
    
    