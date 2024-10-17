"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

# from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import random
import time
import gym
import d4rl
import torch
import numpy as np
import wandb
from critic import Q_Critic, Critic, V_Critic, VQ_Critic
import os
from get_args import get_args
from tqdm import tqdm
import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from delayedreward import CumulativeRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader, delay_reward
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger
from gym.wrappers import TransformReward
MAX_EPISODE_LEN = 1000
HAMMER_REWARD_SCALE_CONST, PEN_REWARD_SCALE_CONST, MUJOCO_REWARD_SCALE_CONST = 1000, 100, 1

class Experiment:
    def __init__(self, variant, device):
        
        self.USE_GRAD_CLIP, self.USE_LR_SCHEDULER = (variant['grad_clip'] == 1), (variant['lr_scheduler'] == 1)
        self.variant = variant  
        if variant['actor_rl_coeff'] < 1e-10: assert variant['stoc'] == 1 and variant['use_entropy_reg'] == 1, "Supervised learning must be standard!"
        self.env_name = variant['env']
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        # std_lst = sum([[x+"-"+y+"-v2" for y in ['medium', 'medium_replay', 'random']] for x in ['hopper', 'walker2d', 'halfcheetah', 'ant']]) + sum([[x+'-'+y+"-v1" for y in ["human", "cloned", "expert"]] for x in ['door', 'relocate', 'hammer', 'pen']])
        #print(std_lst)
        #exit(0)
        if variant['custom_dataset'] == 0: # variant['env'] in std_lst:
            self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
                variant["env"]
            )
        else:
            raise NotImplementedError('Error!') 
        
        self.delayed_reward_flag = variant['delayed_reward']
        
        assert self.delayed_reward_flag != 1, "Error!"
        
        if self.delayed_reward_flag > 1:
            self.offline_trajs = delay_reward(self.offline_trajs, self.delayed_reward_flag)
        else:
            print("not delayed!")
        # initialize by offline trajs
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        
        if variant['rl_algo'] in ['AWR', 'PPO']:
            if variant['rl_algo'] != 'AWR':
                self.online_buffer = ReplayBuffer(99999999) # will be manually cleared
            else:
                self.online_buffer = ReplayBuffer(99999999, self.offline_trajs) # initialization with offline buffer
                returns = [self.online_buffer.trajectories[i]["rewards"].sum() for i in range(len(self.online_buffer.trajectories))]
                sorted_inds = np.argsort(returns)  # lowest to highest
                self.online_buffer.trajectories = [
                    self.online_buffer.trajectories[ii] for ii in sorted_inds
                ]

        self.aug_trajs = []

        if variant['stoc'] == 0: assert variant['use_entropy_reg'] == 0, "Error!"

        self.device = device
        self.target_entropy = -self.act_dim
        self.model = DecisionTransformer(
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            action_range=self.action_range,
            max_length=variant["K"],
            eval_context_length=variant["eval_context_length"],
            max_ep_len=MAX_EPISODE_LEN,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
            stochastic_policy=(variant['stoc'] == 1),
            ordering=variant["ordering"],
            init_temperature=variant["init_temperature"],
            target_entropy=self.target_entropy,
        ).to(device=self.device)

        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        if self.USE_LR_SCHEDULER == 1:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
            )
        else: self.scheduler = None
        if variant['stoc'] == 1:
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4 * variant['temperature_learnable'],
                betas=[0.9, 0.999],
            )

        else:
            self.log_temperature_optimizer = torch.nn.ModuleList() #torch.nn.parameter.Parameter(torch.zeros(1)) 
        
        is_mujoco_flag = ((variant["env"].find("hopper-") != -1) or (variant["env"].find("ant-") != -1) or (variant["env"].find("walker2d-") != -1) or (variant["env"].find("halfcheetah-") != -1))
         
        if variant['rl_algo'] in ["TD3", "SAC", "AWAC"]:
            self.critic = Q_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        elif variant['rl_algo'] == 'IQL':
            self.critic = VQ_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        else:
            self.critic = V_Critic(state_dim=self.state_dim, action_dim=self.act_dim, time_dim=variant['critic_time_dim'], time_aware=variant['critic_time_aware'], activation=variant['critic_activation'], normalization=variant['critic_normalization'], with_layernorm = ((variant["override_layernorm"] != -1) and ((is_mujoco_flag == False) or (variant["override_layernorm"] == 1)))).to(device=self.device)
        
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=variant['critic_learning_rate'])

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        
        self.reward_scale = 1.0 if ("relocate" in variant['env'] or "pen" in variant['env'] or 'hammer' in variant['env'] or 'maze2d' in variant['env']) else 0.001 
        
        # warning: relocate and door might need to retest - reward scale modified from 0.001 to 1.0 !
        
        self.logger = Logger(variant)

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict(),
        }
        if self.USE_LR_SCHEDULER == 1:
            to_save["scheduler_state_dict"] = self.scheduler.state_dict()
        
        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.USE_LR_SCHEDULER == 1:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log_temperature_optimizer.load_state_dict(
                checkpoint["log_temperature_optimizer_state_dict"]
            )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        totrets, totshape = [], []
        
        
        if env_name.find('antmaze') != -1: # False: 
            one_step_traj_count = 0
            
            # if it is antmaze, rule out trajectories that are 1-len
            
            idx = []
            print("before-len:", len(trajectories))
            for i in range(len(trajectories)):
                #print('state:', trajectories[i]['state'])
                # print('obs:', trajectories[i]['observations'])
                
                trajectories[i]['rewards'] -= 1
                print(trajectories[i]['rewards'])
                if trajectories[i]['observations'].shape[0] == 1 and self.variant['remove_trivial_trajs'] == 1:
                    one_step_traj_count += 1
                    if one_step_traj_count >= 10: continue
                
                idx.append(i)
            # medium-play: 10752 -> 1316
            # medium-diverse: 2955 -> 1203
            # umaze-diverse: 1035 -> 1020
            # large-diverse: 7182 -> 1743
            # large-play: 13499 -> 1870
            trajectories = [trajectories[i] for i in idx] 
            print("after-len:", len(trajectories))
            #print("subreward!")
            #for i in range(len(trajectories)):
            #     print("start:", trajectories[i]['observations'][0], "end:", trajectories[i]['observations'][-1], "len:", trajectories[i]['observations'].shape)
            #     print("total_reward:", trajectories[i]['rewards']) 
            
            #exit(0)
        
        for i in range(len(trajectories)):
            #print("REW-before:", trajectories[i]['rewards'].sum())
            #print("traj:", trajectories[i])
            if env_name.find("hammer-") != -1: trajectories[i]['rewards'] /= HAMMER_REWARD_SCALE_CONST 
            elif (env_name.find("pen-") != -1 and env_name.find("open") == -1) or env_name.find('relocate-') != -1 or env_name.find('door')!= -1: trajectories[i]['rewards'] /= PEN_REWARD_SCALE_CONST
            else: trajectories[i]['rewards'] /= MUJOCO_REWARD_SCALE_CONST
            #print("REW-after:", trajectories[i]['rewards'].sum(), trajectories[i]['rewards'].shape)
            totrets.append(trajectories[i]['rewards'].sum())
            totshape.append(trajectories[i]['rewards'].shape)
        
        totrets, totshape = np.array(totrets), np.array(totshape)
        print("totrets-max:", totrets.max(), "totrets-mean:", totrets.mean())
        
        def count_values_in_array(arr):
            # Find unique values and their counts
            values, counts = np.unique(arr, return_counts=True)
            
            # Combine the counts and values into a list of tuples
            count_value_pairs = list(zip(values, counts))
            
            # Sort the list of tuples by the value (the first element of each tuple)
            sorted_count_value_pairs = sorted(count_value_pairs, key=lambda x: x[0])
            
            # Print the sorted counts and values
            for value, count in sorted_count_value_pairs:
                print(f"Value: {value}, Count: {count}")
            
        count_values_in_array(totshape)
        #exit(0)
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def determine_rl_params(self, variant):
        if variant['rl_algo'] == 'TD3':
            rl_params = {"algo": "TD3", "TD3_policy_noise": self.variant['TD3_policy_noise'], "TD3_tau": self.variant['TD3_tau'], "TD3_noise_clip": self.variant["TD3_noise_clip"]}
        elif variant['rl_algo'] == 'SAC': 
            assert args.num_actor_update_interval == 1, "Error!"
            rl_params = {"algo": "SAC", "SAC_tau": self.variant['SAC_tau']} 
        elif variant["rl_algo"] == 'AWAC':
            assert args.num_actor_update_interval == 1, "Error!"
            rl_params = {"algo": "AWAC", "AWAC_normalize_adv": self.variant['AWAC_normalize_adv'], "AWAC_tau":self.variant['AWAC_tau'], "AWAC_beta": self.variant['AWAC_beta'], 'AWAC_soft_flag': self.variant['AWAC_soft_flag']}
        elif variant["rl_algo"] == 'PPO':
            assert args.num_actor_update_interval == 1, "Error!"
            rl_params = {"algo": "PPO", "PPO_eps_clip":self.variant['PPO_eps_clip'], "PPO_td_lambda": self.variant['PPO_td_lambda'], "PPO_old_logprob_generated_in_training": self.variant['PPO_old_logprob_generated_in_training']}
        elif variant["rl_algo"] == 'AWR':
            assert args.num_actor_update_interval == 1, "Error!"
            rl_params = {"algo": "AWR", "AWR_beta": self.variant["AWR_beta"], "AWR_normalize_adv": self.variant['AWR_normalize_adv'], 'AWR_td_lambda': self.variant['AWR_td_lambda']}
        elif variant['rl_algo'] == 'IQL':
            assert args.num_actor_update_interval == 1, "Error!"
            rl_params = {"algo": 'IQL', 'IQL_ratio': self.variant['IQL_ratio'], 'IQL_beta': self.variant['IQL_beta'], "IQL_tau": self.variant['IQL_tau']}
        
        rl_params["normalized_rl_coeff"], rl_params['gamma'] = self.variant['normalized_rl_coeff'], self.variant['gamma']
        
        return rl_params 


    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN
        t0 = time.time()  
        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            print("target-return:", target_return)
            #exit(0)
            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                self.variant['online_data_mode'],
                max_ep_len=max_ep_len,
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                noise_level=(self.variant['expl_noise'] if self.variant['stoc'] == 0 else 0),
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
            )
        t1 = time.time()
        self.replay_buffer.add_new_trajs(trajs)
        if self.variant['rl_algo'] in ["PPO", "AWR"]:
            self.online_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)
        print("collect:", t1 - t0, "addtraj:", time.time() - t1, "total-return:", returns)
        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")

        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        
        rl_params = self.determine_rl_params(self.variant)

        trainer = SequenceTrainer(
            model=self.model,
            critic=self.critic,
            optimizer=self.optimizer,
            critic_optimizer=self.critic_optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            actor_update_interval=self.variant['num_actor_update_interval'],
            rl_params=rl_params,
            entropy_flag=self.variant["use_entropy_reg"],
            scheduler=(self.scheduler if self.variant['lr_scheduler'] else None),
            device=self.device,
            use_grad_clip_flag=self.USE_GRAD_CLIP,
            use_lr_scheduler_flag=self.USE_LR_SCHEDULER,
            variant=self.variant,
            state_mean = self.state_mean,
            state_std = self.state_std,
            action_range=self.action_range,
            reward_scale=self.reward_scale,
            state_dim=self.state_dim,
            action_dim=self.act_dim 
        )

        writer = None
        #(SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            # in every iteration, prepare the data loader
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"] * self.variant['num_actor_update_interval'],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range
            )
            
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                update_critic=True,
                rl_coeff=(0 if (self.variant['RL_from_start'] == 0) else self.variant['actor_rl_coeff']),
                sup_coeff=(1 if (self.variant['RL_from_start'] == 0) else self.variant['actor_sup_coeff']),#self.variant['actor_sup_coeff'],
                pretrain_flag=True 
            )
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            
            outputs_wandb = outputs
            outputs_wandb["iter_num"], outputs_wandb["total_transitions_sampled"] = self.pretrain_iter, self.total_transitions_sampled,
            wandb.log(outputs_wandb)
            
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1

    def evaluate(self, eval_fns, video_debug=0, video_name=""):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        
        print("video debug:", video_debug, "video_name:", video_name)
        #exit(0)
        for eval_fn in eval_fns: 
            o = eval_fn(self.model, video_debug, video_name)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        if self.env_name.find('antmaze') != -1:
            outputs['evaluation/SR'] = np.sum(o['evaluation/length'] < (700 if (self.env_name.find("umaze") != -1) else 1000)) 

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")

        rl_params = self.determine_rl_params(self.variant)

        trainer = SequenceTrainer(
            model=self.model,
            critic=self.critic,
            optimizer=self.optimizer,
            critic_optimizer=self.critic_optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            actor_update_interval=self.variant['num_actor_update_interval'],
            rl_params=rl_params, 
            entropy_flag=self.variant["use_entropy_reg"],
            scheduler=(self.scheduler if self.variant['lr_scheduler'] else None),
            device=self.device,
            use_grad_clip_flag=self.USE_GRAD_CLIP,
            use_lr_scheduler_flag=self.USE_LR_SCHEDULER,
            variant=self.variant,
            state_mean = self.state_mean,
            state_std = self.state_std,
            action_range=self.action_range,
            reward_scale=self.reward_scale,
            state_dim=self.state_dim,
            action_dim=self.act_dim   
        )
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
            )
        ]
        writer = None # (SummaryWriter(self.logger.log_path) if self.variant["log_to_tb"] else None)
        while self.online_iter < self.variant["max_online_iters"]:

            outputs = {}
            
            tot_len = 0
            t0 = time.time()
            while True:
            
                augment_outputs = self._augment_trajectories(
                    online_envs,
                    self.variant["online_rtg"],
                    n=self.variant["num_online_rollouts"],
                )
                outputs.update(augment_outputs)
            
                tot_len += augment_outputs['aug_traj/length']
                if tot_len >= self.variant['minimum_sapairs_per_iter']: break 

            t1 = time.time()

            print("replay_buffer_rew:", [self.replay_buffer.trajectories[i]['rewards'].sum() for i in range(len(self.replay_buffer.trajectories))])
            # exit(0)
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"] * self.variant['num_actor_update_interval'],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False
 
            t2 = time.time()
             
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
                update_critic=True,
                rl_coeff=self.variant['actor_rl_coeff'],
                sup_coeff=self.variant['actor_sup_coeff'],
                online_buffer = (self.online_buffer if self.variant['rl_algo'] in ['PPO', 'AWR'] else None)
            )
            t25 = time.time()
            outputs.update(train_outputs)

            if self.variant['rl_algo'] == "PPO":
                self.online_buffer.clear()
            elif self.variant['rl_algo'] == "AWR":
                self.online_buffer.prune(self.variant['AWR_buffer_size'])
            if evaluation:
                eval_outputs, eval_reward = self.evaluate(eval_fns, self.variant['video_debug'], self.EXP_NAME+"-iter"+str(self.online_iter))
                outputs.update(eval_outputs)

            t3 = time.time()

            outputs["time/total"] = time.time() - self.start_time

            outputs_wandb = outputs
            outputs_wandb["iter_num"], outputs_wandb["total_transitions_sampled"] = self.pretrain_iter, self.total_transitions_sampled,
            wandb.log(outputs_wandb)
            
            is_mujoco_flag = ((self.variant["env"].find("hopper-") != -1) or (self.variant["env"].find("ant-") != -1) or (self.variant["env"].find("walker2d-") != -1) or (self.variant["env"].find("halfcheetah-") != -1))
            
            if outputs_wandb['total_transitions_sampled'] >= (600000 if is_mujoco_flag else 1050000): exit(0)
            # log the metrics
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )
            t4 = time.time()
            if self.online_iter % 10 == 0:
                self._save_model(
                    path_prefix=self.logger.log_path,
                    is_pretrain_model=False,
                )

            t5 = time.time()
            print("outerloop - collectdata:", t1 - t0, "construct dataloader:", t2 - t1, "train:", t25 - t2, "prune:", t3 - t25, "metric:", t4 - t3, "save:", t5 - t4)

            self.online_iter += 1

    def __call__(self, EXP_NAME):
        
        self.EXP_NAME = EXP_NAME
        
        utils.set_seed_everywhere(args.seed)
       
        import d4rl

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_flag,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_flag * entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )
            
        def mse_loss_fn(a_hat, a, attention_mask):
            return ((a_hat - a) ** 2)[attention_mask > 0].mean()

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                import d4rl

                env = gym.make(env_name)
                
                if self.delayed_reward_flag > 1: 
                    env = CumulativeRewardWrapper(env, self.delayed_reward_flag)
                if env_name.find("hammer-") != -1:
                    env = TransformReward(env, lambda r: r / HAMMER_REWARD_SCALE_CONST)
                elif env_name.find("pen-") != -1 or env_name.find('relocate-') != -1 or env_name.find('door')!= -1:
                    env = TransformReward(env, lambda r: r / PEN_REWARD_SCALE_CONST)
                elif env_name.find('antmaze-') != -1:
                    env = TransformReward(env, lambda r: r-1)
                elif env_name.find('hopper') != -1 or env_name.find('walker2d') != -1 or env_name.find('ant') != -1 or env_name.find('halfcheetah') != -1:
                    env = TransformReward(env, lambda r: r / MUJOCO_REWARD_SCALE_CONST)
                env.seed(seed)
                if hasattr(env.env, "wrapped_env"):
                    env.env.wrapped_env.seed(seed)
                elif hasattr(env.env, "seed"):
                    env.env.seed(seed)
                else:
                    pass
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")
                return env

            return make_env_fn

        print("\n\nMaking Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )

        self.start_time = time.time()
        if self.variant["max_pretrain_iters"]:
            self.pretrain(eval_envs, (loss_fn if self.variant['stoc'] == 1 else mse_loss_fn))

        if self.variant["max_online_iters"]:
            print("\n\nMaking Online Env.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            self.online_tuning(online_envs, eval_envs, (loss_fn if self.variant['stoc'] == 1 else mse_loss_fn))
            online_envs.close()

        eval_envs.close()


if __name__ == "__main__":

    #device = utils.get_best_gpu()
    #runtime = utils.check_modification()

    args = get_args()
    
    if args.env.find('random') != -1 and args.force_no_minimum == 0:
        if args.env.find('halfcheetah') == -1:
         assert args.minimum_sapairs_per_iter > 0, "there must be lower bound for sapairs in random environment!"
    
    if args.force_no_minimum != 0:
        print("forced no minimum!")
    
    device = torch.device('cuda:'+str(args.force_device)) if args.force_device >= 0 else utils.get_best_gpu()
    runtime = utils.check_modification()

    utils.set_seed_everywhere(args.seed)
    experiment = Experiment(vars(args), device)

    suffix_alg = "-none" if vars(args)["actor_rl_coeff"] < 1e-10 else (vars(args)["rl_algo"]) 
    suffix_det = "-stoc" if vars(args)["stoc"] == 1 else "-det"
    
    EXP_NAME = str(runtime)+"_"+args.env+"-mixed"+suffix_det+suffix_alg+"-seed"+str(args.seed)+(" " if args.delayed_reward == 0 else "delay-"+str(args.delayed_reward))+"-evalc-"+str(args.eval_context_length)+ ("" if args.actor_rl_coeff == 0 else ("both"+str(args.actor_rl_coeff)+"+"+str(args.actor_sup_coeff) if args.actor_sup_coeff > 0 else "pureRL"))+"-delay20bugfixed-LB"+str(args.minimum_sapairs_per_iter)+("-fixed" if args.normalized_rl_coeff == 0 else "-variable")
    
    wandb.init(entity="XXXXXXX",project="odt-TD3branch", name=EXP_NAME)

    print("=" * 50)
    experiment(EXP_NAME)