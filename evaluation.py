"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
MAX_EPISODE_LEN = 1000


def create_vec_eval_episodes_fn(
    vec_env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    use_mean=False,
    reward_scale=0.001,
):
    def eval_episodes_fn(model, video_debug=0, name=None):
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        ret = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            online_data_mode='eval',
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            video_debug=video_debug
        )
        
        if video_debug == 0: returns, lengths, trajs = ret
        else: returns, lengths, trajs, video = ret
        
        suffix = "_gm" if use_mean else ""
        
        if video_debug == 1:
            import imageio
            assert name is not None, "Error!"
            imageio.mimsave("video/"+name+".mp4", video, fps=25)
            #exit(0)
        
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
            f"evaluation/return": returns,
            f"evaluation/length": lengths
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    online_data_mode,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
    noise_level=0,
    video_debug=0
):
    assert len(target_return) == vec_env.num_envs
    t0 = time.time()
    if online_data_mode == "eval": model.eval()
    else: model.train()
    model.to(device=device)
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    
    next_states = torch.zeros(0, device=device, dtype=torch.float32)
    if model.stochastic_policy:
        action_log_probs = []
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)
    
    if video_debug == 1:
        video = []
    
    unfinished = np.ones(num_envs).astype(bool)
    for t in range(max_ep_len):
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

        # the return action is a SquashNormal distribution
        if model.stochastic_policy:
            if use_mean:
                action = action_dist.mean.reshape(num_envs, -1, act_dim) 
            else:
                action = action_dist.sample().reshape(num_envs, -1, act_dim)
                
            # print("shape3:", action_dist.mean.shape, action.shape)
            #print(action_dist.log_likelihood(action).shape, "!!!")
            #exit(0)
            action_log_probs.append(action_dist.log_likelihood(action.clamp(*model.action_range))[:, -1])
            
            action = action[:, -1]
            
        else:
            action = action_dist[:, -1] + noise_level * torch.randn_like(action_dist[:, -1])
        
        action = action.clamp(*model.action_range)

        #print("action:") 
        
        if video_debug == 1:
            img = vec_env.render(mode='rgb_array')
            video.append(img)
        #exit(0)
        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())
        #####
        #print("t:", t, action_dist.mean.shape, action_dist.mean[:, -1], action_dist.std[:, -1])
        #print("states:", (states - state_mean) / state_std, "actions:", actions, "rewards:", rewards.view(-1), "target_return:", target_return.view(-1), "timesteps:", timesteps.view(-1))   
        #####
        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action
        
        #print(t, "action dimension:", action.shape)
        
        next_state = _['terminal_observation'] if ('terminal_observation' in _) else state
        
        state = (
            torch.from_numpy(state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        ###########
        
        # print("info:", _)
        
        next_state = (
            torch.from_numpy(next_state).to(device=device).reshape(num_envs, -1, state_dim)
        )
        next_states = torch.cat([next_states.float(), next_state.float()], dim=1)
        ###########
        reward = torch.from_numpy(reward).to(device=device).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            break
    t1 = time.time()
    if model.stochastic_policy: 
        action_log_probs = torch.vstack(action_log_probs).T  

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "next_observations": next_states[ii].detach().cpu().numpy()[:ep_len],
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        if model.stochastic_policy: traj["action_log_probs"] = action_log_probs[ii].detach().cpu().numpy()[:ep_len]
        trajectories.append(traj)
    t2 = time.time()
    print("collecttraj:", t1 - t0, "deal:", t2 - t1)
    
    if video_debug == 0:
        return (
            episode_return.reshape(num_envs),
            episode_length.reshape(num_envs),
            trajectories,
        )
    else:
        return (
            episode_return.reshape(num_envs),
            episode_length.reshape(num_envs),
            trajectories,
            video
        )
