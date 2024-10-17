"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
import copy
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random
from data import create_dataloader


########################################################################################## 
from decision_transformer.models.decision_transformer import TanhTransform, SquashedNormal
##########################################################################################

class SequenceTrainer:
    def __init__(
        self,
        model,
        critic,
        optimizer,
        critic_optimizer,
        log_temperature_optimizer,
        variant,
        state_dim,
        action_dim,
        action_range,
        reward_scale,
        state_mean,
        state_std,
        actor_update_interval=0,
        scheduler=None,
        entropy_flag=True,
        rl_params=None,
        device="cuda",
        use_grad_clip_flag=True,
        use_lr_scheduler_flag=True
    ):
        self.model = model
        self.model_target = copy.deepcopy(self.model).to(device)
        self.variant=variant
        self.model_target.eval()
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.optimizer = optimizer
        self.critic_optimizer = critic_optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.state_mean, self.state_std = state_mean, state_std
        self.state_dim, self.act_dim = state_dim, action_dim
        self.action_range, self.reward_scale = action_range, reward_scale
        self.use_grad_clip_flag, self.use_lr_scheduler_flag = use_grad_clip_flag, use_lr_scheduler_flag
        
        if self.use_lr_scheduler_flag:
            self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.entropy_flag = entropy_flag
        self.actor_update_interval = actor_update_interval
        assert self.actor_update_interval > 0, "Error!"

        self.algo = rl_params['algo']
        if self.algo == 'TD3': 
            self.policy_noise = rl_params['TD3_policy_noise']
            self.noise_clip = rl_params['TD3_noise_clip']
            self.tau = rl_params['TD3_tau']
        elif self.algo == 'SAC':
            assert self.model.stochastic_policy, "SAC must use stochastic policy!"
            assert not self.entropy_flag, "SAC must not use entropy regularizer (it has an entropy term of itself. Set args.temperature_learnable to be 0 if you want a fixed value)!"
            # self.alpha = rl_params['SAC_alpha']
            self.tau = rl_params['SAC_tau']
        elif self.algo == 'AWAC':
            #self.alpha = rl_params['AWAC_alpha']
            assert not self.entropy_flag, "AWAC, similar to SAC, must not use entropy regularizer (it has an entropy term of itself. Set args.temperature_learnable to be 0 if you want a fixed value)!"
            self.beta = rl_params['AWAC_beta']
            self.tau = rl_params['AWAC_tau']
            self.normalize_adv = rl_params['AWAC_normalize_adv']
            self.soft_flag = rl_params['AWAC_soft_flag']
            #if self.alpha > 0: assert self.model.stochastic_policy, "AWAC with alpha>0 must use stochastic policy!"
        elif self.algo == 'PPO':
            assert self.model.stochastic_policy, "PPO must use stochastic policy!"
            assert not self.entropy_flag, "PPO must not use entropy regularizer (it has an entropy term of itself!)"
            self.eps_clip = rl_params['PPO_eps_clip']
            self.old_logprob_generated_in_training = (rl_params['PPO_old_logprob_generated_in_training'] == 1)
            self.lam = rl_params['PPO_td_lambda']
        elif self.algo == 'AWR':
            assert not self.entropy_flag, "AWR, similar to SAC, must not use entropy regularizer (it has an entropy term of itself. Set args.temperature_learnable to be 0 if you want a fixed value)!"
            self.normalize_adv = rl_params['AWR_normalize_adv']
            self.beta = rl_params['AWR_beta']
            self.lam = rl_params['AWR_td_lambda']
            self.online_ep_count = 0
        elif self.algo == 'IQL':
            self.tau = rl_params['IQL_tau']
            self.beta = rl_params['IQL_beta']
            self.ratio = rl_params['IQL_ratio']
        
        self.gamma = rl_params['gamma']
        self.normalized_rl_coeff = rl_params['normalized_rl_coeff']
        
        if self.gamma == 1: assert self.critic.time_aware != 0, "Error!"

    def update_TD_lambda_by_buffer(self, online_buffer, ignore_exist=False):
        t0 = time.time()
        for i in range(len(online_buffer.trajectories)):
            if ignore_exist and 'target' in online_buffer.trajectories[i]: continue
            
            states = online_buffer.trajectories[i]["observations"]
            rewards = online_buffer.trajectories[i]['rewards']
            
            states = torch.from_numpy((states - self.state_mean) / (self.state_std + 1e-8)).to(self.device).float()
            
            actions = torch.from_numpy(online_buffer.trajectories[i]['actions'].clip(*self.action_range)).to(self.device).float()
            
            
            timesteps = torch.tensor([i for i in range(online_buffer.trajectories[i]['actions'].shape[0])]).to(self.device)
            dones = torch.from_numpy(online_buffer.trajectories[i]['terminals']).to(self.device) if 'terminals' in online_buffer.trajectories[i] else torch.from_numpy(online_buffer.trajectories[i]['dones']).to(self.device)
            
            if self.critic.time_aware == 0: V = self.critic(states, actions)
            else: V = self.critic(states, actions, timesteps)
            
            NS = torch.from_numpy((online_buffer.trajectories[i]['next_observations'][-1] - self.state_mean) / (self.state_std + 1e-8)).to(self.device).float()
            
            def process_segment(reward, value, lam, NS):
                disc_return = np.zeros(reward.shape[0])
                gae = 0
                gamma = self.gamma
                for t in reversed(range(reward.shape[0])):
                    d = dones[t].item() # (t == reward.shape[0] - 1) #???
                    if d:
                        delta = reward[t] - value[t]
                    else:
                        if t + 1 >= reward.shape[0]: VV = self.critic(NS, None).detach().cpu().numpy() # action is never used for the methods that uses TD_lambda
                        else: VV = value[t+1]
                        delta = reward[t] + gamma * VV - value[t]
                    # print("delta:", delta, "gae:", gae, "d:", d, "lam:", lam)
                    gae = delta + gamma * lam * (1 - d) * gae
                    disc_return[t] = gae + value[t]
                return disc_return
            
            online_buffer.trajectories[i]['old_V'] = V.detach().cpu().numpy()
            online_buffer.trajectories[i]['target'] = process_segment(rewards, V.detach().cpu().numpy(), self.lam, NS)
            t1 = time.time()
        print("update buffer time:", t1 - t0)
            
        print("updated!")
        print(online_buffer.trajectories[i].keys())
        
    def train_iteration(
        self,
        loss_fn,
        dataloader,
        update_critic=True,
        rl_coeff=0,
        sup_coeff=1,
        pretrain_flag=False,
        online_buffer=None
    ):
    
        INPUT_RL_COEFF = rl_coeff 
    
        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        _ = 0
        
        if self.algo in ["TD3", "SAC", "AWAC", "IQL"]:
            for trajs in dataloader:
                
                for i in range(len(trajs)): 
                    # print(x)
                    trajs[i] = trajs[i].to(self.device)
                
                if update_critic: 
                    t0 = time.time()
                    if self.algo == "TD3": Q_distr, Q_mean, Q_std, critic_loss = self.critic_step_stochastic_TD3(trajs)
                    elif self.algo == 'SAC': Q_mean, Q_std, critic_loss = self.critic_step_stochastic_SAC(trajs)
                    elif self.algo == 'AWAC': Q_mean, Q_std, critic_loss = self.critic_step_stochastic_AWAC(trajs)
                    elif self.algo == 'IQL': Q_mean, Q_std, V_mean, V_std, v_loss, q_loss, adv, critic_loss = self.critic_step_stochastic_IQL(trajs)
                if _ % self.actor_update_interval == 0:
                    loss, nll, entropy, rl_loss, sup_loss, rl_coeff = self.train_step_stochastic(loss_fn, trajs, self.entropy_flag, rl_coeff=INPUT_RL_COEFF, sup_coeff=sup_coeff, adv=(None if self.algo != 'IQL' else adv))
                    
                    if _ % 50 == 0: print("rl_coeff:", rl_coeff) 
                    
                    losses.append(loss)
                    nlls.append(nll)
                    entropies.append(entropy)
                    
                if pretrain_flag and _ % 100 == 0:
                    logs = {"pretrain/critic_loss": critic_loss, "pretrain/actor_loss": loss, "pretrain/nll": nll, "pretrain/entropy": entropy, 'pretrain/rl_loss': rl_loss, 'pretrain/sup_loss': sup_loss, 'pretrain/Q_mean': Q_mean, "pretrain/Q_std": Q_std}
                    if self.algo == 'TD3': logs['pretrain/Q_distr'] = Q_distr
                    if self.algo == 'IQL': logs['pretrain/V_loss'], logs['pretrain/Q_loss'], logs['pretrain/V_mean'], logs['pretrain/V_std'] = v_loss, q_loss, V_mean, V_std
                    wandb.log(logs)
                
                _ += 1
    
            if not pretrain_flag: 
                logs["training/critic_MSE_loss"] = critic_loss 
                logs["training/Q_mean"] = Q_mean
                logs["training/Q_std"] = Q_std
                if self.algo == 'TD3': logs['training/Q_distr'] = Q_distr
                logs['training/rl_coeff'] = rl_coeff                
                
            logs["time/training"] = time.time() - train_start
            logs["training/train_loss_mean"] = np.mean(losses)
            logs["training/train_loss_std"] = np.std(losses)
            logs["training/rl_loss"] = rl_loss
            logs["training/sup_loss"] = sup_loss
            logs["training/nll"] = nlls[-1]
            logs["training/entropy"] = entropies[-1]
            
            logs["training/temp_value"] = self.model.temperature().detach().cpu().item() if self.model.stochastic_policy else 0
    
        elif self.algo == "PPO":
            if pretrain_flag:
                 assert rl_coeff == 0 and sup_coeff == 1, "Error!" 
                 
                 for trajs in dataloader:
                    # critic only
                    self.V_critic_step_noTDlambda(trajs, logs) # you should not use RTG here because it is scaled
                    self.train_step_stochastic(loss_fn, trajs, self.entropy_flag, rl_coeff=0, sup_coeff=1)
                 
            else:
                nlls, entropies = [], []
                self.update_TD_lambda_by_buffer(online_buffer)
                online_dataloader = create_dataloader(
                    trajectories=online_buffer.trajectories,
                    num_iters=self.variant["num_updates_per_online_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                    logprob_flag=True,
                    V_flag=True
                )
                
                print("len-traj:", len(online_buffer.trajectories))
                
                if self.old_logprob_generated_in_training:
                    self.model_target = copy.deepcopy(self.model).to(self.device)
                
                for offline_trajs, online_trajs in zip(dataloader, online_dataloader):
                   for i in range(len(offline_trajs)): offline_trajs[i] = offline_trajs[i].to(self.device)
                   for i in range(len(online_trajs)): online_trajs[i] = online_trajs[i].to(self.device)
                   (
                       states,
                       actions,
                       next_states,
                       rewards,
                       dones,
                       rtg,
                       timesteps,
                       ordering,
                       padding_mask
                   ) = offline_trajs
                   
                   _, action_preds, _ = self.model.forward(
                       states,
                       actions,
                       rewards,
                       rtg[:, :-1],
                       timesteps,
                       ordering,
                       padding_mask=padding_mask,
                   )
                   
                   assert self.entropy_flag == 0, "Error!"
                   action_target = torch.clone(actions)
                   actor_sup_loss, nll, entropy = loss_fn(
                       action_preds,  # a_hat_dist
                       action_target,
                       padding_mask,
                       self.entropy_flag,
                       self.model.temperature().detach(),  # no gradient taken here
                   )
                   nlls.append(nll)
                   entropies.append(entropy)
                   
                   #self.model.eval()
                   #self.model_target.eval()
                   (
                       states,
                       actions,
                       next_states,
                       rewards,
                       dones,
                       rtg,
                       timesteps,
                       ordering,
                       padding_mask,
                       old_logprob,
                       V_old, 
                       V_target
                   ) = online_trajs 
                   # actor loss
                   _, action_preds, _ = self.model.forward(
                       states,
                       actions,
                       rewards,
                       rtg[:, :-1],
                       timesteps,
                       ordering,
                       padding_mask=padding_mask,
                   )
                   
                   ###################
                   if self.old_logprob_generated_in_training:
                       #self.model_target.eval()
                       recorded_old_logprob = old_logprob 
                       
                       with torch.no_grad():
                           _, old_action_preds, _ = self.model_target.forward(
                               states,
                               actions,
                               rewards,
                               rtg[:, :-1],
                               timesteps,
                               ordering,
                               padding_mask=padding_mask,
                           )
                       old_logprob = old_action_preds.log_likelihood(actions).unsqueeze(-1)
                       print("recorded_old_logprob:", recorded_old_logprob.view(-1))
                       #self.model_target.train()
                   ###################
                   
                   logprob = action_preds.log_likelihood(actions).unsqueeze(-1)
                   # dist_entropy = action_preds.entropy().unsqueeze(-1)
                   #print("shape1:", logprob.shape, old_logprob.shape)
                   assert logprob.shape == old_logprob.shape, "Error!"
                   ratios = torch.exp((logprob - old_logprob.detach()).clamp(max=20))
                   #print("shape2:", rewards.shape, V_target.shape)
                   assert rewards.shape == V_target.shape, "Error!"
                   adv = V_target - V_old
                   #print("shape3:", ratios.shape, adv.shape)
                   surr1 = ratios * adv
                   surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*adv
                   #print("shape4:", surr1.shape, surr2.shape, dist_entropy.shape, padding_mask.shape)
                   actor_rl_loss = (-torch.min(surr1, surr2))[padding_mask.unsqueeze(-1) > 0] #  - 0.01 * dist_entropy
                   #print("shape5:", actor_rl_loss.shape)
                   actor_rl_loss = actor_rl_loss.mean()
                   if self.normalized_rl_coeff == 1: rl_coeff /= V_target.abs().mean().detach()
                   loss = rl_coeff*actor_rl_loss+sup_coeff*actor_sup_loss
                   losses.append(loss.item())
                   self.actor_backward(loss, 0) # dist_entropy.mean()
                   # critic loss
                   if self.critic.time_aware == 0: V = self.critic(states, actions)
                   else: V = self.critic(states, actions, timesteps)
                   critic_loss = 0.5 * ((V - V_target) ** 2)[padding_mask > 0].mean()
                   self.critic_backward(critic_loss)
                   
                   # print("ratios:", ratios, "logprob:", logprob, "old_logprob:", old_logprob)
                   logs['training/rl_coeff'] = rl_coeff
                   logs['training/std'] = action_preds.std.mean().item()
                   logs["training/ratios"] = ratios.mean().item()
                   logs["training/logprob"] = logprob.mean().item()
                   logs['training/old_logprob'] = old_logprob.mean().item()
                   logs["training/surr1"] = surr1.mean().item()
                   logs["training/surr2"] = surr2.mean().item()
                   # backward together
                
        elif self.algo == "AWR":
            
            if pretrain_flag:
                for trajs in dataloader:
                    # critic only
                    self.V_critic_step_noTDlambda(trajs, logs)
                    self.train_step_stochastic(loss_fn, trajs, self.entropy_flag, rl_coeff=0, sup_coeff=1)        
            
            else: 
                offline_data = []
                nlls, entropies = [], []
                
                if self.online_ep_count < 10: rl_coeff = 0 # buffer warmup
                t0 = time.time()
                self.update_TD_lambda_by_buffer(online_buffer, ignore_exist=True) # should only be triggered once
                t1 = time.time()
                online_dataloader = create_dataloader(
                    trajectories=online_buffer.trajectories,
                    num_iters=self.variant["num_updates_per_online_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                    logprob_flag=False,
                    V_flag=True
                )
                t2 = time.time()
                for offline_trajs, online_trajs in zip(dataloader, online_dataloader):
                    # copy data
                    
                    for i in range(len(offline_trajs)): offline_trajs[i] = offline_trajs[i].to(self.device)
                    for i in range(len(online_trajs)): online_trajs[i] = online_trajs[i].to(self.device)
                    
                    offline_data.append(offline_trajs)

                    (
                        states,
                        actions,
                        next_states,
                        rewards,
                        dones,
                        rtg,
                        timesteps,
                        ordering,
                        padding_mask,
                        V_old,
                        V_target
                    ) = online_trajs 
                    assert V_target.shape == V_old.shape, "Error!"
                    # critic steps (online only)
                    if self.critic.time_aware == 0: V = self.critic(states, actions)
                    else: V = self.critic(states, actions, timesteps)
                    critic_loss = 0.5 * ((V - V_target) ** 2).mean()
                    self.critic_backward(critic_loss)
                t3 = time.time() 
                self.update_TD_lambda_by_buffer(online_buffer)
                t4 = time.time()
                online_dataloader = create_dataloader(
                    trajectories=online_buffer.trajectories,
                    num_iters=self.variant["num_updates_per_online_iter"],
                    batch_size=self.variant["batch_size"],
                    max_len=self.variant["K"],
                    state_dim=self.state_dim,
                    act_dim=self.act_dim,
                    state_mean=self.state_mean,
                    state_std=self.state_std,
                    reward_scale=self.reward_scale,
                    action_range=self.action_range,
                    logprob_flag=False,
                    V_flag=True
                ) 
                
                # online_it = iter(online_dataloader)  
                
                t5 = time.time()
                
                for i, online_traj in zip(range(len(offline_data)), online_dataloader): 
                    
                    ta = time.time()
                    (
                        states,
                        actions,
                        next_states,
                        rewards,
                        dones,
                        rtg,
                        timesteps,
                        ordering,
                        padding_mask
                    ) = offline_data[i]
                    
                    _, action_preds, _ = self.model.forward(
                        states,
                        actions,
                        rewards,
                        rtg[:, :-1],
                        timesteps,
                        ordering,
                        padding_mask=padding_mask,
                    )
                    tb = time.time()
                    action_target = torch.clone(actions)
                    if self.model.stochastic_policy: 
                        actor_sup_loss, nll, entropy = loss_fn(
                            action_preds,  # a_hat_dist
                            action_target,
                            padding_mask,
                            self.entropy_flag,
                            self.model.temperature().detach(),  # no gradient taken here
                        )
                    
                    else:
                        actor_sup_loss = loss_fn(action_preds, action_target, padding_mask)
                        nll, entropy = torch.tensor([114.514]), torch.tensor([1919.810]) 
                    tsp1 = time.time()
                    # online_traj = next(online_it)
                    tsp2 = time.time()
                    for j in range(len(online_traj)): online_traj[j] = online_traj[j].to(self.device)
                    (
                        states,
                        actions,
                        next_states,
                        rewards,
                        dones,
                        rtg,
                        timesteps,
                        ordering,
                        padding_mask,
                        V_old,
                        V_target
                    ) = online_traj
                    
                    nlls.append(nll)
                    entropies.append(entropy)
                    tc = time.time()
                    assert V_target.shape == V_old.shape, "Error!"
                    adv = (V_target - V_old).detach()
                    if self.normalize_adv == 1:
                        adv_mean, adv_std = adv.mean(), 1e-8 + adv.std()
                        adv = (adv - adv_mean) / adv_std
                    weight = torch.clamp(torch.exp(adv / self.beta), max=20)
                    # print(weight.shape, adv.shape)
                    td = time.time()
                    logs["training/weight_max"] = weight.max().item()
                    logs["training/weight_mean"] = weight.mean().item()
                    
                    _, action_preds, _ = self.model.forward(
                        states,
                        actions,
                        rewards,
                        rtg[:, :-1],
                        timesteps,
                        ordering,
                        padding_mask=padding_mask,
                    )
                    te = time.time()
                    if self.model.stochastic_policy:
                        policy_logpp = action_preds.log_likelihood(actions).unsqueeze(-1)
                        ent = action_preds.entropy()
                    else:
                        policy_logpp = -((action_preds - actions) ** 2).sum(dim=-1).unsqueeze(-1)
                    #print("w:", weight.shape, policy_logpp.shape)
                    assert weight.shape == policy_logpp.shape, "Error!"
                    actor_rl_loss = -(weight * policy_logpp).mean()
                    tf = time.time()
                    # print("temp:", self.model.temperature(), "ent:", ent)
                    
                    if self.model.stochastic_policy: 
                        # print("temp:", self.model.temperature().item(), "ent:", ent)
                        actor_rl_loss -= (self.model.temperature() * ent).mean()
                    # actor steps
                    
                    if self.normalized_rl_coeff == 1: rl_coeff /= V_target.abs().mean().detach()
                    
                    loss = rl_coeff*actor_rl_loss+sup_coeff*actor_sup_loss
                    losses.append(loss.item())
                    tg = time.time()
                    self.actor_backward(loss, entropy)
                        
                    self.online_ep_count += 1
                    
                    
                    if i == 0: 
                        print("breakdown - 1st forward for sup:", tb - ta, "suploss and ready for rl:", tsp1 - tb, tsp2 - tsp1, tc - tsp2, "weight:", td - tc, "2nd forward for rl:", te - td, "rl loss:", tf - te, "backward:", tg - tf)
                t6 = time.time()
                print("update TD lambda by buffer:", t1 - t0, "dataloader:", t2 - t1, "critic:", t3 - t2, "update TD lambda by buffer:", t4 - t3, "create dataloader:", t5 - t4, "actor:", t6 - t5)
                    
        
        if self.algo in ['PPO', 'AWR']:
            if not pretrain_flag: 
                logs["training/critic_MSE_loss"] = critic_loss
                logs["training/train_loss_mean"] = np.mean(np.array(losses))
                logs["training/train_loss_std"] = np.std(np.array(losses))
                logs["training/rl_loss"] = actor_rl_loss
                logs["training/sup_loss"] = actor_sup_loss
                logs["training/nll"] = nlls[-1]
                logs["training/entropy"] = entropies[-1]
        
        return logs
    
    def V_critic_step_noTDlambda(self, trajs, logs):
        for i in range(len(trajs)):
            trajs[i] = trajs[i].to(self.device)
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        if self.critic.time_aware == 0: 
            V = self.critic(states, None)
            next_V = self.critic(next_states, None)
        else: 
            V = self.critic(states, None, timesteps)
            next_V = self.critic(next_states, None)
        # shape: BS * K * 1
        # print(V.shape, rtg[:, :-1, :].shape)
        # critic_loss = ((V - rtg[:, :-1, :]) ** 2).mean()# note rtg is scaled and no gamma so it should not be used here
        assert V.shape == rewards.shape and rewards.shape == next_V.shape and next_V.shape == V.shape, "Error!"
        # print(V.shape, next_V.shape, rewards.shape, dones.shape)
        # exit(0)
        critic_loss = ((V - (rewards + (1 - dones.unsqueeze(-1)) * next_V)) ** 2).mean()
        
        logs["training/V_mean"] = V.mean().item()
        logs["training/V_std"] = V.std().item()
        logs["training/critic_MSE_loss"] = critic_loss
        self.critic_backward(critic_loss)
    
    def construct_next_action(self, trajs):
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        
        with torch.no_grad():
            BATCH_SIZE, ACT_DIM = actions.shape[0], actions.shape[-1]
            
            next_actions_input = torch.cat([actions, torch.zeros(BATCH_SIZE, 1, ACT_DIM).to(self.device)], dim=1)[:, 1:]
            next_rewards_input = torch.cat([rewards, torch.zeros(BATCH_SIZE, 1, 1).to(self.device)], dim=1)[:, 1:]
            next_target_return_input = rtg[:, 1:]
            next_timesteps_input = torch.cat([timesteps, (timesteps[:, -1] + 1).reshape(BATCH_SIZE ,1)], dim=1)[:, 1:]
            
            state_pred, action_dist, reward_pred = self.model_target.get_predictions(
            next_states.to(dtype=torch.float32),
            next_actions_input.to(dtype=torch.float32),
            next_rewards_input.to(dtype=torch.float32),
            next_target_return_input.to(dtype=torch.float32),
            next_timesteps_input.to(dtype=torch.long),
            num_envs=BATCH_SIZE,
            external_context_len=actions.shape[1]
            )
            
        return action_dist

    def get_target_Q(self, next_states, next_actions, timesteps):
        if self.critic.time_aware == 0: target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
        else: target_Q1, target_Q2 = self.critic_target(next_states, next_actions, timesteps=timesteps)
        target_Q = torch.min(target_Q1, target_Q2)
        return target_Q

    def actor_backward(self, actor_loss, entropy):
        self.optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip_flag: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        
        if self.entropy_flag == 1 or self.algo in ["SAC", "AWAC"]:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        if self.use_lr_scheduler_flag and self.scheduler is not None:
            self.scheduler.step()

    def critic_backward(self, critic_loss):
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def critic_step_stochastic_AWAC(self, trajs):
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        
        action_dist = self.construct_next_action(trajs)
        with torch.no_grad():
            if self.model.stochastic_policy: 
                next_actions = action_dist.sample()
                next_actions_logprob = action_dist.log_likelihood(next_actions) 
                target_Q = self.get_target_Q(next_states, next_actions, timesteps) - self.model.temperature() * next_actions_logprob.unsqueeze(-1) * (self.soft_flag != 0)
                
            else:
                next_actions = action_dist
                target_Q = self.get_target_Q(next_states, next_actions, timesteps)
            
            target_Q = rewards + (1 - dones.unsqueeze(-1)) * self.gamma * target_Q
        
        if self.critic.time_aware == 0: current_Q1, current_Q2 = self.critic(states, actions)
        else: current_Q1, current_Q2 = self.critic(states, actions, timesteps=timesteps)

        assert current_Q1.shape == target_Q.shape and current_Q2.shape == target_Q.shape and padding_mask.unsqueeze(-1).shape == target_Q.shape, "Error!"
        
        critic_loss = (((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2) * (padding_mask.unsqueeze(-1) > 0)).mean()
        self.critic_backward(critic_loss)
        
        return current_Q1.mean().item(), current_Q1.std().item(), critic_loss.item()
            

    def critic_step_stochastic_SAC(self, trajs):
        t0 = time.time()
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        t1 = time.time()
        
        action_dist = self.construct_next_action(trajs)
        with torch.no_grad():
            next_actions = action_dist.sample()
            next_actions_logprob = action_dist.log_likelihood(next_actions) # entropy() ?
            
            t2 = time.time()
            
            ### important for SAC!!! ###
            target_Q = self.get_target_Q(next_states, next_actions, timesteps)
            
            #print(target_Q.shape, next_actions_logprob.shape, rewards.shape, "!!!")
            #exit(0)
            
            target_Q -= self.model.temperature() * next_actions_logprob.unsqueeze(-1)
            ############################
            assert rewards.shape == dones.unsqueeze(-1).shape and rewards.shape == target_Q.shape, "Error!"
            target_Q = rewards + (1 - dones.unsqueeze(-1)) * self.gamma * target_Q
        
            t3 = time.time()
            
        if self.critic.time_aware == 0: current_Q1, current_Q2 = self.critic(states, actions)
        else: current_Q1, current_Q2 = self.critic(states, actions, timesteps=timesteps)

        assert current_Q1.shape == target_Q.shape and current_Q2.shape == target_Q.shape and padding_mask.unsqueeze(-1).shape == target_Q.shape, "Error!"
        
        critic_loss = (((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2) * (padding_mask.unsqueeze(-1) > 0)).mean()
        
        t4 = time.time()
        
        self.critic_backward(critic_loss)

        t5 = time.time()
        if random.random() < 0.005:
            print("critic construct input:", t1 - t0, "inference:", t2 - t1, "gettarget:", t3 - t2, "getcriticloss:", t4 - t3, "backward:", t5 - t4) 

        return current_Q1.mean().item(), current_Q1.std().item(), critic_loss.item()

    def critic_step_stochastic_TD3(self, trajs):
        
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        
        ################### CRITIC UPDATE ###################
        next_action = self.construct_next_action(trajs)
        with torch.no_grad():
            if self.model.stochastic_policy: next_action = next_action.sample() # mean?
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_action + noise).clamp(*self.model_target.action_range)
            
            target_Q = self.get_target_Q(next_states, next_actions, timesteps)
            
            # print("rewards:", rewards.shape, "dones:", dones.shape, "target_Q:", target_Q.shape)
            
            assert rewards.shape == dones.unsqueeze(-1).shape and rewards.shape == target_Q.shape, "Error!"
            target_Q = rewards + (1 - dones.unsqueeze(-1)) * self.gamma * target_Q
        
        if self.critic.time_aware == 0: current_Q1, current_Q2 = self.critic(states, actions)
        else: current_Q1, current_Q2 = self.critic(states, actions, timesteps=timesteps)

        assert current_Q1.shape == target_Q.shape and current_Q2.shape == target_Q.shape and padding_mask.unsqueeze(-1).shape == target_Q.shape, "Error!"
        
        """
        def asymmetric_l2_loss(u, tau):
            return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)
            
        critic_loss = ((asymmetric_l2_loss(current_Q1 - target_Q, 0.9) + asymmetric_l2_loss(current_Q2 - target_Q, 0.9)) * (padding_mask.unsqueeze(-1) > 0)).mean()
        """
        critic_loss = (((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2) * (padding_mask.unsqueeze(-1) > 0)).mean()
        
        # + 0.01 * ((current_Q1 * padding_mask.unsqueeze(-1)) ** 2).mean() + 0.01 * ((current_Q2 * padding_mask.unsqueeze(-1)) ** 2).mean()
        self.critic_backward(critic_loss)
        
        return current_Q1.detach(), current_Q1.mean().item(), current_Q1.std().item(), critic_loss.item()
        
        
        #####################################################
        
    def critic_step_stochastic_IQL(self, trajs):
        
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        
        with torch.no_grad():
            target_Q = self.get_target_Q(states, actions, timesteps).detach()
            next_v = self.critic_target.V(next_states, timesteps).detach()
            
        v = self.critic.V(states, timesteps)
        adv = target_Q - v
        
        def asymmetric_l2_loss(u, tau):
            return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)
        
        v_loss = asymmetric_l2_loss(adv, self.ratio)
        
        assert rewards.shape == dones.unsqueeze(-1).shape and rewards.shape == target_Q.shape, "Error!"
        
        target_Q = rewards + (1 - dones.unsqueeze(-1)) * self.gamma * next_v
        
        if self.critic.time_aware == 0: current_Q1, current_Q2 = self.critic(states, actions)
        else: current_Q1, current_Q2 = self.critic(states, actions, timesteps=timesteps)
        
        assert current_Q1.shape == target_Q.shape and current_Q2.shape == target_Q.shape and padding_mask.unsqueeze(-1).shape == target_Q.shape, "Error!"
        
        q_loss = (((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2) * (padding_mask.unsqueeze(-1) > 0)).mean()
        
        critic_loss = v_loss + q_loss
        
        self.critic_backward(critic_loss)
        
        return current_Q1.mean().item(), current_Q1.std().item(), v.mean().item(), v.std().item(), v_loss.item(), q_loss.item(), adv.detach(), critic_loss.item()

        

    def train_step_stochastic(self, loss_fn, trajs, entropy_flag, rl_coeff=0, sup_coeff=1 ,adv=None):
        t0 = time.time()
        (
            states,
            actions,
            next_states,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs
        #for x in trajs:print("shape!", x.shape)
        #print("----")
        action_target = torch.clone(actions)
        t1 = time.time() 
        
        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )
        
        t2 = time.time() 
        if self.model.stochastic_policy: 
            sup_loss, nll, entropy = loss_fn(
                action_preds,  # a_hat_dist
                action_target,
                padding_mask,
                entropy_flag,
                self.model.temperature().detach(),  # no gradient taken here
            )
        
        else:
            sup_loss = loss_fn(action_preds, action_target, padding_mask)
            nll, entropy = torch.tensor([114.514]), torch.tensor([1919.810]) 
        
        t3 = time.time()
        
        ################ LOSS += RL gradient ################### 
        
        # warning: padding mask?
        
        if self.algo == "TD3":
            action = action_preds.mean if self.model.stochastic_policy else action_preds # this action clearly has a gradient
            
            if self.critic.time_aware == 0: rl_loss = -self.critic.Q1(states, action)
            else: rl_loss = -self.critic.Q1(states, action, timesteps=timesteps)
            
            if self.normalized_rl_coeff == 1: rl_coeff /= rl_loss.abs().mean().detach()
            
            assert padding_mask.unsqueeze(-1).shape == rl_loss.shape, "Error!"
            
            rl_loss = rl_loss[padding_mask.unsqueeze(-1) > 0].mean()
        
        elif self.algo == 'IQL':
            assert adv is not None and adv.shape[0] == states.shape[0], "Error!"
            #########
            EXP_ADV_MAX = 100
            exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX).squeeze(dim=-1)
            action = action_preds.mean if self.model.stochastic_policy else action_preds
            if self.model.stochastic_policy:
                policy_logpp = action_preds.log_likelihood(actions)
            else:
                policy_logpp = -((action_preds - actions) ** 2).sum(dim=-1)
            
            assert policy_logpp.shape == exp_adv.shape and exp_adv.shape == padding_mask.shape, "Error!"+str(exp_adv.shape)+" "+str(policy_logpp.shape)+" "+str(padding_mask.shape)
            rl_loss = -(policy_logpp * exp_adv)[padding_mask > 0].mean()
            
        
        elif self.algo == 'SAC':
            ent = action_preds.entropy().unsqueeze(-1)
            action = action_preds.rsample()
            if self.critic.time_aware == 0: current_Q1, current_Q2 = self.critic(states, action) # it should be "action" that is sampled from the distribution (with gradients!), not "actions" from the buffer!
            else: current_Q1, current_Q2 = self.critic(states, action, timesteps=timesteps)
            Q = torch.min(current_Q1, current_Q2)
            assert Q.shape == ent.shape and Q.shape == padding_mask.unsqueeze(-1).shape, "Error!" 
            rl_loss = -(self.model.temperature() * ent + Q)[padding_mask.unsqueeze(-1) > 0].mean()
            
            if self.normalized_rl_coeff == 1: rl_coeff /= Q.abs().mean().detach()
        
        elif self.algo == 'AWAC':
            pi = action_preds.sample() if self.model.stochastic_policy else action_preds
            ent = action_preds.entropy()
            with torch.no_grad():
                if self.critic.time_aware == 0: 
                    Q1_pi, Q2_pi = self.critic(states, pi)
                    Q1_old_actions, Q2_old_actions = self.critic(states, actions)
                else: 
                    Q1_pi, Q2_pi = self.critic(states, pi, timesteps=timesteps)
                    Q1_old_actions, Q2_old_actions = self.critic(states, actions)
                Q_old_actions = torch.min(Q1_old_actions, Q2_old_actions)
                V_pi = torch.min(Q1_pi, Q2_pi)
                adv_pi = (Q_old_actions - V_pi) / self.beta
                if self.normalize_adv == 1:
                    adv_pi -= adv_pi.max()
                    weights = (torch.exp(adv_pi) / torch.exp(adv_pi).sum()).squeeze(-1)
                else:
                    weights = torch.clamp(torch.exp(adv_pi), max=20).squeeze(-1) 
                
            if self.model.stochastic_policy:
                policy_logpp = action_preds.log_likelihood(actions)
            else:
                policy_logpp = -((action_preds - actions) ** 2).sum(dim=-1)
            # print(policy_logpp.shape, weights.shape, padding_mask.shape)
            #exit(0)
            assert policy_logpp.shape == weights.shape and weights.shape == padding_mask.shape and ent.shape == weights.shape, "Error!"+str(ent.shape)+" "+str(weights.shape)    
            rl_loss = -(self.model.temperature() * ent * (self.soft_flag == 2) + policy_logpp * weights)[padding_mask > 0].mean() 
            
            if self.normalized_rl_coeff == 1: rl_coeff /= V_pi.abs().mean().detach()
            
        else:
            rl_loss = torch.zeros(1).to(self.device)
        
        loss = sup_loss * sup_coeff + rl_loss * rl_coeff
        
        t4 = time.time()
        
        self.actor_backward(loss, entropy)
        
        t5 = time.time()
        if self.algo in ['AWAC', 'SAC', 'TD3', 'IQL']:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
            for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        t6 = time.time()
                
        if random.random() < 0.002: 
            print("construct input:", t1 - t0, "get action pred:", t2 - t1, "get suploss:", t3 - t2, "get rlloss:", t4 - t3, "gradupdate:", t5 - t4, "param update:", t6 - t5, "total actor:", t6 - t0)
            if self.algo == 'IQL': print("exp-adv:", exp_adv.mean(), exp_adv.max())
            # print("weights:", weights.max(), weights.mean())
        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
            rl_loss.detach().cpu().item(),
            sup_loss.detach().cpu().item(),
            rl_coeff.detach().cpu().item() if isinstance(rl_coeff, torch.Tensor) else rl_coeff
        )
