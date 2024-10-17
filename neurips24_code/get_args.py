import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    
    parser.add_argument("--force_device", type=int, default=-1)
    
    # model options
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    
    parser.add_argument("--remove_trivial_trajs", type=int, default=1)
     
    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature_learnable", type=int, default=1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--lr_scheduler', type=int, default=1)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500) 
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=10)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--video_debug", type=int, default=0)
    ########################## our own #######################################
    
    parser.add_argument('--override_layernorm', type=int, default=0) # 1 = must have layernorm, -1 = must not have layernorm
    
    parser.add_argument('--critic_time_aware', type=int, default=0)
    parser.add_argument('--critic_time_dim', type=int, default=10)
    parser.add_argument('--critic_learning_rate', type=float, default=0.001)
    
    parser.add_argument('--minimum_sapairs_per_iter', type=int, default=0)
    parser.add_argument('--force_no_minimum', type=int, default=0)
    parser.add_argument('--use_entropy_reg', type=int, default=1)
    
    parser.add_argument("--actor_sup_coeff", type=float, default=1)
    parser.add_argument("--actor_rl_coeff", type=float, default=0.1)

    parser.add_argument('--rl_algo', type=str, default='TD3')

    parser.add_argument('--RL_from_start', type=int, default=0)

    parser.add_argument("--gamma", type=float, default=0.99) 
    parser.add_argument('--stoc', type=int, default=1)
    parser.add_argument('--num_actor_update_interval', type=int, default=2)
    parser.add_argument('--normalized_rl_coeff', type=int, default=0)
    parser.add_argument('--expl_noise', type=float, default=0.1)
    parser.add_argument('--critic_activation', type=str, default="relu")
    parser.add_argument('--critic_normalization', type=str, default='layernorm')
    ############################# SAC #######################################
    parser.add_argument('--SAC_tau', type=float, default=0.005)
    # parser.add_argument('--SAC_alpha', type=float, default=0.2) SAC_alpha = init_temperature
    ############################# TD3 #######################################
    parser.add_argument('--TD3_policy_noise', type=float, default=0.2)
    parser.add_argument("--TD3_noise_clip", type=float, default=0.5)
    parser.add_argument("--TD3_tau", type=float, default=0.005)
    
    ############################# AWAC ######################################
    # parser.add_argument('--AWAC_alpha', type=float, default=0) AWAC_alpha = init_temperature
    parser.add_argument('--AWAC_beta', type=float, default=1)
    parser.add_argument('--AWAC_tau', type=float, default=0.005)
    parser.add_argument('--AWAC_soft_flag', type=int)
    parser.add_argument('--AWAC_normalize_adv', type=int)
    ############################# AWR #######################################
    parser.add_argument('--AWR_beta', type=float, default=1)
    parser.add_argument('--AWR_normalize_adv', type=int, default=0)
    parser.add_argument('--AWR_buffer_size', type=int, default=50000)
    parser.add_argument('--AWR_td_lambda', type=float, default=0.98)
    ############################# PPO #######################################
    parser.add_argument('--PPO_eps_clip', type=float, default=0.2)
    parser.add_argument('--PPO_td_lambda', type=float, default=0.98)
    parser.add_argument('--PPO_old_logprob_generated_in_training', type=int, default=0)
    
    ############################# IQL #######################################
    parser.add_argument('--IQL_ratio', type=float, default=0.7) # adroit: 0.9
    parser.add_argument('--IQL_beta', type=float, default=3) # adroit:  10
    parser.add_argument('--IQL_tau', type=float, default=0.005)
    
    ############################ collection eval or train? #####################
    parser.add_argument('--online_data_mode', type=str, default='eval')
    ############################ custom dataset? #########################
    parser.add_argument('--custom_dataset', type=int, default=0)
    parser.add_argument('--suffix', type=str, default="")
    
    parser.add_argument('--delayed_reward', type=int, default=0) # should not be set to 1; 1 is meaningless!
    
    args = parser.parse_args()
    
    return args