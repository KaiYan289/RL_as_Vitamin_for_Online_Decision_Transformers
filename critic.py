import torch.nn.functional as F
import torch.nn as nn
import torch

def Adanorm(x, shape):
    y = F.layer_norm(x, shape)
    C, k = 1, 0.1
    return C * (1 - k * y) * y

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024):
        super(Critic, self).__init__()
        assert time_aware in [0, 1, 2], "Error!"
        self.time_dim, self.time_aware = time_dim, time_aware
        if self.time_aware == 1:
            self.time_embed = nn.Embedding(max_timestep, time_dim)
            
    def binarize(self, timestep):
        res = torch.zeros(size=tuple(list(timestep.shape)+[self.time_dim]))
        ts = timestep.clone()
        
        for j in range(1, self.time_dim + 1):
            cur_val = 1 << (self.time_dim - j)
            is_one = (ts >= cur_val).int()
            res[..., j-1] = is_one
            ts -= is_one * cur_val
        return res
    
    def forward(self, state_action, timesteps=None):
        pass

class V_Critic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu'):
      super(V_Critic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      self.l1 = nn.Linear(state_dim, 256)
      self.l2 = nn.Linear(256, 256)    
      self.l3 = nn.Linear(256, 1)
      self.with_layernorm = with_layernorm
      print("activation:", activation)
      assert activation in ['relu', 'tanh', 'leakyrelu', 'swish'] and normalization in ['layernorm', 'adanorm', 'affinenorm'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      if activation == 'relu': self.A = F.relu
      elif activation == 'tanh': self.A = F.tanh
      elif activation == 'leakyrelu': self.A = F.leaky_relu
      elif activation == 'swish': self.A = F.silu
      self.N = None
      self.normalization = normalization
      if self.with_layernorm:
          if normalization == "layernorm": self.N = lambda x: F.layer_norm(x, (256,))
          elif normalization == 'affinenorm': self.N = [nn.LayerNorm((256,)) for _ in range(2)]
          elif normalization == "adanorm": self.N = lambda x: Adanorm(x, (256,))
      print("with layernorm:", with_layernorm)
      exit(0)
      
    def forward(self, state, action, timesteps=None):
       if self.time_aware == 1: sa = torch.cat([state, self.time_embed(timesteps)], -1)
       elif self.time_aware == 2: sa = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
       else: sa = state
       if not self.with_layernorm:
           q1 = self.A(self.l1(sa))
           q1 = self.A(self.l2(q1))
           q1 = self.l3(q1)
       else:
           if self.normalization == "affinenorm":
               q1 = self.A(self.N[0](self.l1(sa)))
               q1 = self.A(self.N[1](self.l2(q1)))
               q1 = self.l3(q1)
           else:
               q1 = self.A(self.N(self.l1(sa)))
               q1 = self.A(self.N(self.l2(q1)))
               q1 = self.l3(q1)
       
       return q1

class Q_Critic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu'):
      super(Q_Critic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      
      self.with_layernorm = with_layernorm
      print("activation:", activation)
      assert activation in ['relu', 'tanh', 'leakyrelu', 'swish'] and normalization in ['layernorm', 'adanorm', 'affinenorm'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      if activation == 'relu': self.A = F.relu
      elif activation == 'tanh': self.A = F.tanh
      elif activation == 'leakyrelu': self.A = F.leaky_relu
      elif activation == 'swish': self.A = F.silu
      self.N = None
      self.normalization = normalization
      if self.with_layernorm:
          if normalization == "layernorm": self.N = lambda x: F.layer_norm(x, (256,))
          elif normalization == 'affinenorm': self.N = [nn.LayerNorm((256,)) for _ in range(4)]
          elif normalization == "adanorm": self.N = lambda x: Adanorm(x, (256,))
      
  		# Q1 architecture
      self.l1 = nn.Linear(state_dim + action_dim + (0 if time_aware == 0 else time_dim), 256)
      self.l2 = nn.Linear(256, 256)
      self.l3 = nn.Linear(256, 1)
  
  		# Q2 architecture
      self.l4 = nn.Linear(state_dim + action_dim + (0 if time_aware == 0 else time_dim), 256)
      self.l5 = nn.Linear(256, 256)
      self.l6 = nn.Linear(256, 1)
      
      #print("with layernorm:", with_layernorm)
      #exit(0)
  
    def forward(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      if not self.with_layernorm:
          q1 = self.A(self.l1(sa))
          q1 = self.A(self.l2(q1))
          q1 = self.l3(q1)
      
          q2 = self.A(self.l4(sa))
          q2 = self.A(self.l5(q2))
          q2 = self.l6(q2)
      else:
          if self.normalization == "affinenorm":
              q1 = self.A(self.N[0](self.l1(sa)))
              q1 = self.A(self.N[1](self.l2(q1)))
              q1 = self.l3(q1)
              		    
              q2 = self.A(self.N[2](self.l4(sa)))
              q2 = self.A(self.N[3](self.l5(q2)))
              q2 = self.l6(q2)
          else:
              q1 = self.A(self.N(self.l1(sa)))
              q1 = self.A(self.N(self.l2(q1)))
              q1 = self.l3(q1)
              		    
              q2 = self.A(self.N(self.l4(sa)))
              q2 = self.A(self.N(self.l5(q2)))
              q2 = self.l6(q2)
      return q1, q2
  
    def Q1(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      if not self.with_layernorm:
         q1 = self.A(self.l1(sa))
         q1 = self.A(self.l2(q1))
         q1 = self.l3(q1)
      else:
         if self.normalization == "affinenorm":
              q1 = self.A(self.N[0](self.l1(sa)))
              q1 = self.A(self.N[1](self.l2(q1)))
              q1 = self.l3(q1)
         else:
             q1 = self.A(self.N(self.l1(sa)))
             q1 = self.A(self.N(self.l2(q1))) 
             q1 = self.l3(q1)
      return q1
   
class VQ_Critic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu'):
      super(VQ_Critic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      
      self.with_layernorm = with_layernorm
      print("activation:", activation)
      assert activation in ['relu', 'tanh', 'leakyrelu', 'swish'] and normalization in ['layernorm', 'adanorm', 'affinenorm'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      if activation == 'relu': self.A = F.relu
      elif activation == 'tanh': self.A = F.tanh
      elif activation == 'leakyrelu': self.A = F.leaky_relu
      elif activation == 'swish': self.A = F.silu
      self.N = None
      self.normalization = normalization
      if self.with_layernorm:
          if normalization == "layernorm": self.N = lambda x: F.layer_norm(x, (256,))
          elif normalization == 'affinenorm': self.N = [nn.LayerNorm((256,)) for _ in range(6)]
          elif normalization == "adanorm": self.N = lambda x: Adanorm(x, (256,))
  		# Q1 architecture
      self.Q_l1 = nn.Linear(state_dim + action_dim + (0 if time_aware == 0 else time_dim), 256)
      self.Q_l2 = nn.Linear(256, 256)
      self.Q_l3 = nn.Linear(256, 1)
  
  		# Q2 architecture
      self.Q_l4 = nn.Linear(state_dim + action_dim + (0 if time_aware == 0 else time_dim), 256)
      self.Q_l5 = nn.Linear(256, 256)
      self.Q_l6 = nn.Linear(256, 1)
      
      self.V_l1 = nn.Linear(state_dim, 256)
      self.V_l2 = nn.Linear(256, 256)    
      self.V_l3 = nn.Linear(256, 1)
      
      #print("with layernorm:", with_layernorm)
      #exit(0)
  
    def forward(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      if not self.with_layernorm:
          q1 = self.A(self.Q_l1(sa))
          q1 = self.A(self.Q_l2(q1))
          q1 = self.Q_l3(q1)
      
          q2 = self.A(self.Q_l4(sa))
          q2 = self.A(self.Q_l5(q2))
          q2 = self.Q_l6(q2)
          
      else:
          if self.normalization == "affinenorm":
              q1 = self.A(self.N[0](self.Q_l1(sa)))
              q1 = self.A(self.N[1](self.Q_l2(q1)))
              q1 = self.Q_l3(q1)
              		    
              q2 = self.A(self.N[2](self.Q_l4(sa)))
              q2 = self.A(self.N[3](self.Q_l5(q2)))
              q2 = self.Q_l6(q2)
          else:
              q1 = self.A(self.N(self.Q_l1(sa)))
              q1 = self.A(self.N(self.Q_l2(q1)))
              q1 = self.Q_l3(q1)
              		    
              q2 = self.A(self.N(self.Q_l4(sa)))
              q2 = self.A(self.N(self.Q_l5(q2)))
              q2 = self.Q_l6(q2)
      return q1, q2
      
    def V(self, state, timesteps=None):
      if self.time_aware == 1: s = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
      elif self.time_aware == 2: s = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
      else: s = state
      
      if not self.with_layernorm:
         v = self.A(self.V_l1(s))
         v = self.A(self.V_l2(v))
         v = self.V_l3(v)
      else:
         if self.normalization == 'affinenorm':
             v = self.A(self.N[4](self.V_l1(s)))
             v = self.A(self.N[5](self.V_l2(v)))
             v = self.V_l3(v)
         else:
             v = self.A(self.N(self.V_l1(s)))
             v = self.A(self.N(self.V_l2(v)))
             v = self.V_l3(v) 
      return v
  
    def Q1(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      if not self.with_layernorm:
         q1 = self.A(self.Q_l1(sa))
         q1 = self.A(self.Q_l2(q1))
         q1 = self.Q_l3(q1)
      else:
         if self.normalization == "affinenorm":
              q1 = self.A(self.N[0](self.Q_l1(sa)))
              q1 = self.A(self.N[1](self.Q_l2(q1)))
              q1 = self.Q_l3(q1)
         else:
             q1 = self.A(self.N(self.Q_l1(sa)))
             q1 = self.A(self.N(self.Q_l2(q1))) 
             q1 = self.Q_l3(q1)
      return q1