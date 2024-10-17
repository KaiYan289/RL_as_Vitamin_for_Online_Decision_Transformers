"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.

Part of the code was adapted from the following: 
* https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
* https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py

Both are licensed under the MIT License.
"""

import torch
import torch.nn as nn

import transformers
import time
from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
import math
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """
    Squashed Normal Distribution(s)

    If loc/std is of size (batch_size, sequence length, d),
    this returns batch_size * sequence length * d
    independent squashed univariate normal distributions.
    """

    def __init__(self, loc, std): # warning: loc != mean! mean is loc after tanhtransform!
        self.loc = loc
        self.std = std
        self.base_dist = pyd.Normal(loc, std)

        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self, N=1):
        # sample from the distribution and then compute
        # the empirical entropy:
        x = self.rsample((N,))
        log_p = self.log_prob(x)

        # log_p: (batch_size, context_len, action_dim),
        return -log_p.mean(axis=0).sum(axis=2)

    def log_likelihood(self, x):
        # log_prob(x): (batch_size, context_len, action_dim)
        # sum up along the action dimensions
        # Return tensor shape: (batch_size, context_len)
        return self.log_prob(x).sum(axis=2)


class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, hidden_dim, act_dim, log_std_bounds=[-5.0, 2.0]):
        super().__init__()

        self.mu = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std = torch.nn.Linear(hidden_dim, act_dim)
        self.log_std_bounds = log_std_bounds

        def weight_init(m):
            """Custom weight init for Conv2D and Linear layers."""
            if isinstance(m, torch.nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(weight_init)

    def forward(self, obs):
        mu, log_std = self.mu(obs), self.log_std(obs)
        log_std = torch.tanh(log_std)
        # log_std is the output of tanh so it will be between [-1, 1]
        # map it to be between [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        std = log_std.exp()
        return SquashedNormal(mu, std)

class MLP(nn.Module):
    def __init__(self, input_size, output_size): 
        super().__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, output_size)
    
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class DecisionMLP(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        action_range,
        ordering=0,
        max_length=None,
        eval_context_length=None,
        max_ep_len=4096,
        action_tanh=True,
        stochastic_policy=False,
        init_temperature=0.1,
        target_entropy=None,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        self.action_range = action_range
        self.model = MLP(state_dim, hidden_size)
        
        
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        if stochastic_policy:
            self.predict_action = DiagGaussianActor(hidden_size, self.act_dim)
        else:
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
        self.stochastic_policy = stochastic_policy

        if stochastic_policy:
            self.log_temperature = torch.tensor(np.log(init_temperature))
            self.log_temperature.requires_grad = True
            self.target_entropy = target_entropy

    def temperature(self):
        if self.stochastic_policy:
            return self.log_temperature.exp()
        else:
            return None

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        ordering,
        padding_mask=None,
    ):

        x = self.model(states)
        action_preds = self.predict_action(x)

        return action_preds, action_preds, action_preds[:, 0]

    def get_predictions(
        self, states, actions, rewards, returns_to_go, timesteps, num_envs=1, external_context_len=None, **kwargs
    ):
        x = self.model(states)
        action_preds = self.predict_action(x)

        return action_preds, action_preds, action_preds.reshape(-1)[:num_envs]
