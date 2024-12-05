﻿#  Code for NeurIPS 2024 Spotlight "Reinforcement Learning Gradients as Vitamin for Online Finetuning Decision Transformers"

This repository is the code for NeurIPS 2024 Spotlight "Reinforcement Learning Gradients as Vitamin for Online Finetuning Decision Transformers".

Bibtex: https://kaiyan289.github.io/bibtex/ODTTD3.txt

Website: https://kaiyan289.github.io/jekyll/update/2024/10/16/ODTTD3.html


## File Structure

**./data:** The data for the experiments, generated by python files in this folder.

**./motivation:** The code that reproduces motivation experiment in the method section. 

**./decision_transformer:** Code for decision transformers.

**./main.py:** Entry point and training framework of our algorithm.

**./trainer.py:** main body of our algorithm (and other RL gradients / ODT).

**./data.py, ./replay_buffer.py:** Code for data storage. 

**./recurrent_*.py:** Same as that without "recurrent" prefix, but for recurrent critic.

**./get_args.py:** Code for argument parser.

**./critic.py:** Code for critic.

**./utils.py, ./logger.py, ./delayedreward.py:** Other auxiliary files.

## Dependency

**Mujoco210 is required for all environments,** and we run our experiments with CUDA 11.3. Below are the dependency for python packages:

d4rl == 1.1

dm-control == 1.0.5

gym == 0.23.1

mujoco-py == 2.1.2.14

numpy == 1.20.3

torch == 2.0.1

transformers == 4.11.0

tqdm

wandb

By default, OpenGL is used for the D4RL [1] environments. However, sometimes OpenGL will be problematic on headless machines; to fix this, try to set MUJOCO_GL environment variable to 'egl' or 'osmesa'.

## Running Code

1. Clone the repository.

2. Install the dependencies as stated in the dependency section.

3. Assume you are in the directory. Run the following command:
```
cd data
python download_d4rl_basic.py
python download_antmaze_datasets.py
python download_gym_datasets.py
python download_advanced.py
```
The four scripts will generate Maze2D, Antmaze, MuJoCo and Adroit datasets respectively. 

4. Find in the line in ./main.py 
```
wandb.init(entity=XXXXXXX, project= ...
```
change XXXXXXX to your key and username for wandb. See wandb official website https://docs.wandb.ai/ for this. We use XXXXXXX for anonimity.

5. run the code to reproduce results; see the next section for command.

## Commands for Reproducing Results

### Adroit
```
python main.py --env pen-{human,cloned,expert}-v1 --eval_rtg 120 --online_rtg 120 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 5 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
python main.py --env hammer-{human,cloned,expert}-v1 --eval_rtg 160 --online_rtg 160 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
python main.py --env door-{human,cloned,expert}-v1 --eval_rtg 40 --online_rtg 40 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
python main.py --env relocate-{human,cloned,expert}-v1 --eval_rtg 50 --online_rtg 50 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
```
where {a, b, c} means choose one from a, b or c.

### AntMaze
```
python main.py --env antmaze-umaze{,-diverse}-v2 --eval_rtg -100 --online_rtg -100 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 5000 --max_pretrain_iters 40 --eval_context_length 1 --K 1 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 2000 --RL_from_start 1 --gamma 0.998
python main.py --env antmaze-medium{-play,-diverse}-v2 --eval_rtg -200 --online_rtg -200 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 5000 --max_pretrain_iters 40 --eval_context_length 1 --K 1 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 2000 --RL_from_start 1 --gamma 0.998
python main.py --env antmaze-large{-play,-diverse}-v2 --eval_rtg -500 --online_rtg -500 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 5000 --max_pretrain_iters 40 --eval_context_length 1 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 2000 --RL_from_start 1 --gamma 0.998
```
where {a, b} means choose one from a, b (a, b can be empty). Use --RL_from_start flag to control whether apply alpha (--actor_rl_coeff) to pretrain or not. Note: for antmaze, following CQL and IQL, we use a reward shaping which is to apply a -1 reward to every step. Thus, a successful trajectory is one that has >-1000 reward. 


### MuJoCo
```
python main.py --env hopper-medium-v2 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --seed 16 
python main.py --env hopper-medium-replay-v2 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --seed 16 
python main.py --env hopper-medium-random-v2 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --seed 16 --minimum_sapairs_per_iter 1000
python main.py --env halfcheetah-medium-v2 --eval_rtg 6000 --online_rtg 12000 --weight_decay 0.0001 --seed 16 --actor_rl_coeff 0.1 --actor_sup_coeff 1 --stoc 0 --use_entropy_reg 0
python main.py --env halfcheetah-medium-replay-v2 --eval_rtg 6000 --online_rtg 12000 --weight_decay 0.0001 --seed 16 --actor_rl_coeff 0.1 --actor_sup_coeff 1 --stoc 0 --use_entropy_reg 0
python main.py --env halfcheetah-random-v2 --eval_rtg 6000 --online_rtg 12000 --weight_decay 0.0001 --seed 16 --actor_rl_coeff 0.1 --actor_sup_coeff 1 --stoc 0 --use_entropy_reg 0 --minimum_sapairs_per_iter 1000
python main.py --env walker2d-medium-v2 --eval_rtg 5000 --online_rtg 10000 --num_updates_per_pretrain_iter 10000 --learning_rate 0.001 --weight_decay 0.001 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --seed 16
python main.py --env walker2d-medium-replay-v2 --eval_rtg 5000 --online_rtg 10000 --num_updates_per_pretrain_iter 10000 --learning_rate 0.001 --weight_decay 0.001 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --seed 16
python main.py --env walker2d-random-v2 --eval_rtg 5000 --online_rtg 10000 --num_updates_per_pretrain_iter 10000 --learning_rate 0.001 --weight_decay 0.001 --actor_rl_coeff 0.1 --gamma 0.99 --use_entropy_reg 0 --stoc 0 --seed 16 --minimum_sapairs_per_iter 1000
python main.py --env ant-random-v2 --minimum_sapairs_per_iter 1000 --eval_rtg 5000 --online_rtg 10000 --weight_decay 0.0001 --actor_rl_coeff 0.1 --gamma 0.99 --actor_sup_coeff 1 --use_entropy_reg 0 --stoc 0 --seed 16
python main.py --env ant-medium-v2 --eval_rtg 5000 --online_rtg 10000 --weight_decay 0.0001 --actor_rl_coeff 0.1 --gamma 0.99 --actor_sup_coeff 1 --use_entropy_reg 0 --stoc 0 --seed 16
python main.py --env ant-medium-replay-v2 --eval_rtg 5000 --online_rtg 10000 --weight_decay 0.0001 --actor_rl_coeff 0.1 --gamma 0.99 --actor_sup_coeff 1 --use_entropy_reg 0 --stoc 0 --seed 16
python main.py --env ant-random-v2 --eval_rtg 5000 --online_rtg 10000 --weight_decay 0.0001 --actor_rl_coeff 0.1 --gamma 0.99 --actor_sup_coeff 1 --use_entropy_reg 0 --stoc 0 --seed 16
```
for delayed reward, add argument "-\-delayed_reward 5".

### Maze2D
```
python main.py --env maze2d-open-v0 --eval_rtg 120 --online_rtg 120 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 10000 --eval_context_length 1 --K 1 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 10000
python main.py --env maze2d-umaze-v1 --eval_rtg 60 --online_rtg 60 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 1 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 2500
python main.py --env maze2d-medium-v1 --eval_rtg 60 --online_rtg 60 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 1 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
python main.py --env maze2d-large-v1 --eval_rtg 60 --online_rtg 60 --weight_decay 0.0001 --rl_algo TD3 --num_actor_update_interval 1 --actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1 --minimum_sapairs_per_iter 1000 --num_updates_per_pretrain_iter 40000 --eval_context_length 1 --K 5 --critic_learning_rate 0.0002 --seed 16 --max_online_iters 99999 --replay_size 5000
```
To reproduce ODT and pure RL result (denoted as "TD3" in the paper), change the argument "--actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 1" to "--actor_rl_coeff 0 --use_entropy_reg 1 --stoc 1 --actor_sup_coeff 1" for ODT and "--actor_rl_coeff 0.1 --use_entropy_reg 0 --stoc 0 --actor_sup_coeff 0" for TD3.

## Reference

[1] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine. D4rl: Datasets for deep data-driven reinforcement learning.  ArXiv:2004.07219, 2020.


