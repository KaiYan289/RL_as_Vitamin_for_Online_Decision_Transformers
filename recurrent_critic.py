import torch.nn.functional as F
import torch.nn as nn
import torch
from critic import Critic

def Adanorm(x, shape):
    y = F.layer_norm(x, shape)
    C, k = 1, 0.1
    return C * (1 - k * y) * y

class LayerNormRNN(nn.Module):
    pass
        
class LayerNormLSTM(nn.Module):
    pass


class V_RecurrentCritic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu', typ='RNN'):
      super(V_RecurrentCritic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      print("activation:", activation) 
      batch_first=True
      assert activation in ['relu', 'tanh'] and normalization == 'layernorm' and typ in ['RNN', 'LSTM'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      self.with_layernorm, self.activation = with_layernorm, activation
      if with_layernorm:
          if typ == 'RNN':
              self.net = LayerNormRNN(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':
              self.net = LayerNormLSTM(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0)
      else:
          if typ == 'RNN':
              self.net = nn.RNN(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':  
              self.net = nn.LSTM(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0) 
      self.v_l1 = nn.Linear(256, 1)

    def forward(self, state, action, timesteps=None):
    
       assert len(state.shape) == 3, "not suitable for recurrent!"
       
       if self.time_aware == 1: sa = torch.cat([state, self.time_embed(timesteps)], -1)
       elif self.time_aware == 2: sa = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
       else: sa = state
       
       q1, hidden = self.net(sa)
       
       if self.with_layernorm: q1 = F.layer_norm(q1, (256,))
       
       q1 = self.v_l1(F.relu(q1)) if self.activation == 'relu' else self.v_l1(F.tanh(q1)) 
       
       return q1

class Q_RecurrentCritic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu', typ='RNN'):
      super(Q_RecurrentCritic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      print("activation:", activation)
      assert activation in ['relu', 'tanh'] and normalization in ['layernorm'] and typ in ['RNN', 'LSTM'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      self.with_layernorm, self.activation = with_layernorm, activation
      SADIM = state_dim + action_dim + (0 if time_aware == 0 else time_dim)
      batch_first=True
      if False: #self.with_layernorm:
          if typ == 'RNN':
              self.Q_1, self.Q_2 = LayerNormRNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation), LayerNormRNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':
              self.Q_1, self.Q_2 = LayerNormLSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0), LayerNormLSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0)
      else:
          if typ == 'RNN':
              self.Q_1, self.Q_2 = nn.RNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation), nn.RNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':
              self.Q_1, self.Q_2 = nn.LSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0), nn.LSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0)
      self.l1 = nn.Linear(256, 1)
      self.l2 = nn.Linear(256, 1)
  
    def forward(self, state, action, timesteps=None):
    
      assert len(state.shape) == 3, "not suitable for recurrent!"
      
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      
      
      q1, _ = self.Q_1(sa)
      q2, _ = self.Q_2(sa)
          
      if self.with_layernorm: q1, q2 = F.layer_norm(q1, (256, )), F.layer_norm(q2, (256,))
      
      q1 = self.l1(F.relu(q1)) if self.activation == 'relu' else self.l1(F.tanh(q1))
      q2 = self.l2(F.relu(q2)) if self.activation == 'relu' else self.l2(F.tanh(q2))
      
      return q1, q2
  
    def Q1(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      
      q1, _ = self.Q_1(sa)
      
      if self.with_layernorm: q1 = F.layer_norm(q1, (256, ))
      
      q1 = self.l1(F.relu(q1)) if self.activation == 'relu' else self.l1(F.tanh(q1))
      
      return q1
   
class VQ_RecurrentCritic(Critic):
    def __init__(self, state_dim, action_dim, time_dim, time_aware, max_timestep=1024, with_layernorm=True, normalization='layernorm', activation='relu', typ='RNN'):
      super(VQ_RecurrentCritic, self).__init__(state_dim, action_dim, time_dim, time_aware, max_timestep)
      print("activation:", activation)
      assert activation in ['relu', 'tanh'] and normalization in ['layernorm'] and typ in ['RNN', 'LSTM'], "Error!" # layernorm = simplenorm, affinenorm = real layernorm
      self.with_layernorm, self.activation = with_layernorm, activation
      self.normalization = normalization
      batch_first=True
      if False: #with_layernorm:
          if typ == 'RNN':
              self.Vnet = LayerNormRNN(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
              self.Q_1, self.Q_2 = LayerNormRNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation), LayerNormRNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':
              self.Vnet = LayerNormLSTM(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
              self.Q_1, self.Q_2 = LayerNormLSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0), LayerNormLSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0)
      else:
          if typ == 'RNN':
              self.Vnet = nn.RNN(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
              self.Q_1, self.Q_2 = nn.RNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation), nn.RNN(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
          elif typ == 'LSTM':
              self.Vnet = nn.RNN(input_size=state_dim, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation)
              self.Q_1, self.Q_2 = nn.LSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0, nonlinearity=activation), nn.LSTM(input_size=SADIM, hidden_size=256, num_layers=3, batch_first=batch_first, dropout=0) 
      self.v_l1 = nn.Linear(256, 1)
      self.q_l1 = nn.Linear(256, 1)
      self.q_l2 = nn.Linear(256, 1)
  
    def forward(self, state, action, timesteps=None):
    
      assert len(state.shape) == 3, "not suitable for recurrent!" 
    
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      
      q1, _ = self.Q_1(sa)
      q2, _ = self.Q_2(sa)
          
      if self.with_layernorm: q1, q2 = F.layer_norm(q1, (256, )), F.layer_norm(q2, (256,))
      
      q1 = self.l1(F.relu(q1)) if self.activation == 'relu' else self.l1(F.tanh(q1))
      q2 = self.l2(F.relu(q2)) if self.activation == 'relu' else self.l2(F.tanh(q2))
      
      return q1, q2
      
    def V(self, state, timesteps=None):
      if self.time_aware == 1: s = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
      elif self.time_aware == 2: s = torch.cat([state, self.binarize(timesteps).to(state.device)], -1)
      else: s = state
      
      v, hidden = self.Vnet(s)
       
      if self.with_layernorm: v = F.layer_norm(v, (256,))
       
      v = self.v_l1(F.relu(v)) if self.activation == 'relu' else self.v_l1(F.tanh(vs))  
      
      return v
  
    def Q1(self, state, action, timesteps=None):
      if self.time_aware == 1: sa = torch.cat([state, action, self.time_embed(timesteps)], -1)
      elif self.time_aware == 2: sa = torch.cat([state, action, self.binarize(timesteps).to(state.device)], -1)
      else: sa = torch.cat([state, action], -1)
      
      q1, _ = self.Q_1(sa)
      if self.with_layernorm: q1 = F.layer_norm(q1, (256, )) 
      q1 = self.l1(F.relu(q1)) if self.activation == 'relu' else self.l1(F.tanh(q1))
      
      return q1