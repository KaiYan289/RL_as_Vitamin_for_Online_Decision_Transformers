import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib
import copy

matplotlib.use('Agg')

seed = 59 # 24, 31 
torch.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)
np.random.seed(seed) 
random.seed(seed) 
torch.backends.cudnn.deterministic = True

dataset_size, pretrain_epoch, finetune_epoch, batch_size, expl_noise = 128, 5, 16, 32, 0.02 

# Generate colors from the Blues colormap
colors_red = plt.cm.Reds(np.linspace(0.2, 1, finetune_epoch))
colors_green = plt.cm.Greens(np.linspace(0.2, 1, finetune_epoch))
colors_blue = plt.cm.Blues(np.linspace(0.2, 1, finetune_epoch))
colors_black = plt.cm.Greys(np.linspace(0.2, 1, finetune_epoch))

def env(a):
    a0 = torch.clamp(a, min=-1, max=1)
    return (a0 + 1) ** 2 * (a0 < 0) + (1 - 2 * a0) * (a0 >= 0) # (a0 + 1) ** 2 * (a0 < 0.8) + (3.24 - 32.4 * (a0 - 0.8)) * (a0 >= 0.8)

# action between [-1, 1]
# reward = (a ** 2)

class Net(nn.Module):
    def __init__(self, is_critic=False, input_size=1):
        super().__init__()
        self.is_critic = is_critic
        width = 128
        self.l1 = nn.Linear(input_size, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, 1)
        
    def forward(self, x):
        #print("x:", x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.is_critic: return self.l3(x)
        else: return F.tanh(self.l3(x)) 

CONST_INPUT = 1 # serve as constant input for RL and RTG for ODT

action = torch.cat([torch.from_numpy(0.05 * np.random.random(100) - 1), torch.from_numpy(0.5 * np.random.random(28) + 0.5)])
reward = env(action)

primitive_action = action.clone()
primitive_reward = reward.clone()

def train_epoch_odt(net, action, reward, optimizer):
    size = action.shape[0]
    idx = torch.randperm(size)
    tot_step = 0
    for j in range(size // batch_size):
        tot_step += 1
        r, a = reward[idx[j * batch_size:(j+1)*batch_size]], action[idx[j * batch_size:(j+1)*batch_size]]
        #print(r.shape, a.shape)
        # print(r.reshape(-1, 1).shape,"!")
        a_pred = net(r.reshape(-1, 1))
        loss = ((a.reshape(-1, 1) - a_pred) ** 2).mean()
        #print("loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tot_step
    
def train_epoch_both(net, critic, action, reward, optimizer, optimizer_c):
    size = action.shape[0]
    idx = torch.randperm(size)
    tot_step = 0
    for j in range(size // batch_size):
        tot_step += 1
        r, a = reward[idx[j * batch_size:(j+1)*batch_size]], action[idx[j * batch_size:(j+1)*batch_size]]
        
        q_pred=critic(torch.cat([r.reshape(-1, 1), a.reshape(-1, 1)], dim=-1))
        q_loss=((q_pred - r.reshape(-1, 1)) ** 2).mean() 
        
        optimizer_c.zero_grad()
        q_loss.backward()
        optimizer_c.step()

        a_pred = net(r.reshape(-1, 1))
        q_of_a_pred = critic(torch.cat([r.reshape(-1, 1), a_pred.reshape(-1,1)], dim=-1))
        
        loss = 0.02 * ((a.reshape(-1, 1) - a_pred) ** 2).mean() - q_of_a_pred.mean() #((a.reshape(-1, 1) - a_pred) ** 2).mean() - q_of_a_pred.mean()
        #print("loss:", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tot_step

def train_epoch_ddpg(actor, critic, action, reward, optimizer_a, optimizer_c):
    
    size = action.shape[0]
    
    idx = torch.randperm(size)
    tot_step = 0
    for j in range(size // batch_size):
        tot_step += 1
        r, a = reward[idx[j * batch_size:(j+1)*batch_size]], action[idx[j * batch_size:(j+1)*batch_size]]
        q_pred = critic(a.reshape(-1, 1))
        q_loss = ((q_pred - r.reshape(-1, 1)) ** 2).mean()
        
        optimizer_c.zero_grad()
        q_loss.backward()
        optimizer_c.step()
        
        a_pred = actor(torch.ones_like(r.reshape(-1, 1)).double())
        q_of_a_pred = critic(a_pred)
        actor_loss = -q_of_a_pred.mean()
        
        optimizer_a.zero_grad()
        actor_loss.backward()
        optimizer_a.step()
    return tot_step 

def rollout(net, action, reward, batch_size=64):
    r_s, a_s = [], []
    for j in range(batch_size):
        a = expl_noise * (np.random.random() - 0.5) * torch.ones(1).double() + net(torch.tensor([CONST_INPUT]).double())
        # print("a:", a)
        r = env(a)
        a_s.append(a.item())
        r_s.append(r.item())
    print('r_s:', np.array(r_s).mean())
    return torch.cat([action, torch.tensor(a_s)]), torch.cat([reward, torch.tensor(r_s)]), np.array(r_s).mean(), np.array(r_s).std(), np.array(a_s) 
 
def DDPG(action, reward):
    ddpg_actor, ddpg_critic = Net().double(), Net(is_critic=True).double()
    ddpg_optim_a = torch.optim.Adam(ddpg_actor.parameters(), lr=1e-3)
    ddpg_optim_c = torch.optim.Adam(ddpg_critic.parameters(), lr=1e-3)
    
    step_vs_reward = []
    
    tot_grad_step = 0
    
    for i in range(pretrain_epoch):
        grad_step = train_epoch_ddpg(ddpg_actor, ddpg_critic, action, reward, ddpg_optim_a, ddpg_optim_c)
        tot_grad_step += grad_step
        
    ddpg_optim_a = torch.optim.Adam(ddpg_actor.parameters(), lr=1e-3)
    ddpg_optim_c = torch.optim.Adam(ddpg_critic.parameters(), lr=1e-3)
    
    new_actions, new_models = [], []
    
    for i in range(finetune_epoch):
        action, reward, avg_now_reward, std_now_reward, new_action = rollout(ddpg_actor, action, reward)
        new_actions.append(new_action)
        new_models.append([copy.deepcopy(ddpg_actor), copy.deepcopy(ddpg_critic)])
        step_vs_reward.append((tot_grad_step, avg_now_reward, std_now_reward))
        grad_step = train_epoch_ddpg(ddpg_actor, ddpg_critic, action, reward, ddpg_optim_a, ddpg_optim_c)
        tot_grad_step += grad_step
        
    #print("step_vs_reward:", step_vs_reward)
    
    return step_vs_reward, new_actions, new_models
    
        
def ODT(action, reward):
    odt_net = Net().double()
    odt_optim = torch.optim.Adam(odt_net.parameters(), lr=1e-3)
    
    step_vs_reward = []
         
    tot_grad_step = 0
    
    for i in range(pretrain_epoch):
        grad_step = train_epoch_odt(odt_net, action, reward, odt_optim) #idx = torch.randperm(dataset_size)
        tot_grad_step += grad_step
    odt_optim = torch.optim.Adam(odt_net.parameters(), lr=1e-3)
    
    new_actions, new_models = [], []
    
    for i in range(finetune_epoch):
        action, reward, avg_now_reward, std_now_reward, new_action = rollout(odt_net, action, reward)
        new_actions.append(new_action)
        new_models.append(copy.deepcopy(odt_net))
        step_vs_reward.append((tot_grad_step, avg_now_reward, std_now_reward))
        grad_step = train_epoch_odt(odt_net, action, reward, odt_optim)
        tot_grad_step += grad_step
        # collect data
    
    return step_vs_reward, new_actions, new_models
    
def Both(action, reward):
    odt_net, odt_critic = Net().double(), Net(is_critic=True, input_size=2).double() 
    odt_optim_a = torch.optim.Adam(odt_net.parameters(), lr=1e-3)
    odt_optim_c = torch.optim.Adam(odt_critic.parameters(), lr=1e-3)
    step_vs_reward = []
         
    tot_grad_step = 0
    
    for i in range(pretrain_epoch):
        grad_step = train_epoch_both(odt_net, odt_critic, action, reward, odt_optim_a, odt_optim_c) 
        tot_grad_step += grad_step
        
    odt_optim_a = torch.optim.Adam(odt_net.parameters(), lr=1e-3)
    odt_optim_c = torch.optim.Adam(odt_critic.parameters(), lr=1e-3)
    
    new_actions, new_models = [], []
    
    for i in range(finetune_epoch):
        action, reward, avg_now_reward, std_now_reward, new_action = rollout(odt_net, action, reward)
        new_actions.append(new_action)
        new_models.append([copy.deepcopy(odt_net), copy.deepcopy(odt_critic)])
        step_vs_reward.append((tot_grad_step, avg_now_reward, std_now_reward))
        grad_step = train_epoch_both(odt_net, odt_critic, action, reward, odt_optim_a, odt_optim_c) 
        tot_grad_step += grad_step
        # collect data
    
    return step_vs_reward, new_actions, new_models


matplotlib.rcParams.update({'font.size': 12})
svr_both, na_both, nm_both = Both(action, reward)
print("------------------------------------------")
action = primitive_action.clone()
reward = primitive_reward.clone()

svr_ddpg, na_ddpg, nm_ddpg = DDPG(action, reward)
print("------------------------------------------")
action = primitive_action.clone()
reward = primitive_reward.clone()

svr_odt, na_odt, nm_odt = ODT(action, reward)

plt.plot([x[0] for x in svr_odt], [x[1] for x in svr_odt], color='black', label='ODT')
plt.plot([x[0] for x in svr_ddpg], [x[1] for x in svr_ddpg], color='blue', label='DDPG')
plt.plot([x[0] for x in svr_both], [x[1] for x in svr_both], color='red', label='ODT+DDPG')
plt.fill_between([x[0] for x in svr_odt], [x[1]-x[2] for x in svr_odt], [x[1]+x[2] for x in svr_odt], alpha=0.5, color='black')
plt.fill_between([x[0] for x in svr_ddpg], [x[1]-x[2] for x in svr_ddpg], [x[1]+x[2] for x in svr_ddpg], alpha=0.5, color='blue')
plt.fill_between([x[0] for x in svr_both], [x[1]-x[2] for x in svr_both], [x[1]+x[2] for x in svr_both], alpha=0.5, color='red')
plt.xlabel('Gradient Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.savefig('reward.png', bbox_inches='tight', pad_inches=0.05)
plt.cla()

#plt.figure(figsize=(3,3))
pts_odt, pts_ddpg, pts_both = [], [], []
for i in range(finetune_epoch):
    
    for j in range(na_odt[i].shape[0]):
        pts_odt.append((i, na_odt[i][j]))
    for j in range(na_ddpg[i].shape[0]):
        pts_ddpg.append((i, na_ddpg[i][j]))
    
    for j in range(na_both[i].shape[0]): 
        pts_both.append((i, na_both[i][j]))

plt.scatter([x[0] for x in pts_odt], [x[1] for x in pts_odt], color='black', label='ODT')
plt.scatter([x[0] for x in pts_ddpg], [x[1] for x in pts_ddpg], color='blue', label='DDPG')
plt.scatter([x[0] for x in pts_both], [x[1] for x in pts_both], color='red', label='ODT+DDPG')
plt.xlabel('Epoch')
plt.ylabel('Action')
plt.legend()
plt.savefig('action.png', bbox_inches='tight', pad_inches=0.05)
plt.cla()


#plt.figure(figsize=(3,3))
for i in range(finetune_epoch):
    
    model = nm_odt[i]
    x = torch.from_numpy(np.arange(0, CONST_INPUT + 0.01, 0.01)).double().reshape(-1, 1)
    y = model(x)
    plt.xlim(-1, 1)
    if i == finetune_epoch // 2: plt.plot(y.detach().numpy().reshape(-1), x.detach().numpy().reshape(-1), color=colors_black[i], label='ODT')
    else: plt.plot(y.detach().numpy().reshape(-1), x.detach().numpy().reshape(-1), color=colors_black[i])
    
    model = nm_both[i][0]
    x = torch.from_numpy(np.arange(0, CONST_INPUT + 0.01, 0.01)).double().reshape(-1, 1)
    y = model(x)
    plt.xlim(-1, 1)
    if i == finetune_epoch // 2: plt.plot(y.detach().numpy().reshape(-1), x.detach().numpy().reshape(-1), color=colors_red[i], label='ODT+DDPG')
    else: plt.plot(y.detach().numpy().reshape(-1), x.detach().numpy().reshape(-1), color=colors_red[i])
plt.legend()
plt.ylabel('Estimated RTG')
plt.xlabel('Action')
plt.savefig('ODT-actor.png', bbox_inches='tight', pad_inches=0.05)

plt.cla()

#plt.figure(figsize=(3,3))
for i in range(finetune_epoch):
    
    model = nm_ddpg[i][1]
    x = torch.from_numpy(np.arange(-1, 1.01, 0.01)).double().reshape(-1, 1)
    y = model(x)
    if i == finetune_epoch // 2: plt.plot(x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), color=colors_blue[i], label='DDPG')
    else: plt.plot(x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), color=colors_blue[i])
    
    
    model = nm_both[i][1]
    x = torch.from_numpy(np.arange(-1, 1.01, 0.01)).double().reshape(-1, 1)
    y = model(torch.cat([env(x), x], dim=-1))

    if i == finetune_epoch // 2: plt.plot(x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), color=colors_red[i], label='ODT+DDPG')
    else: plt.plot(x.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1), color=colors_red[i])


x = torch.from_numpy(np.arange(-1, 1.01, 0.01)).double().reshape(-1, 1)

y2 = env(x)
plt.plot(x.detach().numpy().reshape(-1), y2.detach().numpy().reshape(-1), color='green', label='ground truth')
# exit(0)
plt.xlabel('Action')
plt.ylabel('Estimated RTG')
plt.legend()
plt.savefig('DDPG-critic.png', bbox_inches='tight', pad_inches=0.05)

plt.cla()