"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import gym
import numpy as np
import collections
import pickle
import d4rl


datasets = []

num = [150]

for idx, env_name in enumerate(['maze2d']): # ["halfcheetah", "hopper", "walker2d", "ant"]:
    for dataset_type in ["open", "umaze", 'medium', 'large']: # ["medium", "medium-expert", "medium-replay", "expert"]:
        name = f"{env_name}-{dataset_type}-" + ("v0" if (dataset_type == "open") else "v1")
        env = gym.make(name)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = False
        if "timeouts" in dataset:
            use_timeouts = True

        episode_step = 0
        paths = []
        for i in range(N - 1):
            done_bool = bool(dataset["terminals"][i])
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = episode_step == num[idx] - 1
            for k in [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "terminals",
                "timeouts"
            ]:
                if k == "next_observations": data_[k].append(dataset["observations"][i+1])
                else: data_[k].append(dataset[k][i])
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

        returns = np.array([np.sum(p["rewards"]) for p in paths])
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        RMXS = d4rl.infos.REF_MAX_SCORE[env_name+'-'+dataset_type+"-"+("v0" if (dataset_type == "open") else "v1")]
        RMNS = d4rl.infos.REF_MIN_SCORE[env_name+'-'+dataset_type+"-"+("v0" if (dataset_type == "open") else "v1")]
        print(f"Number of samples collected: {num_samples}")
        print(
            f"{env_name}-{dataset_type}: Trajectory returns: mean = {(np.mean(returns) - RMNS) * 100 / (RMXS - RMNS)}, std = {np.std(returns) * 100 / (RMXS - RMNS)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(paths, f)
